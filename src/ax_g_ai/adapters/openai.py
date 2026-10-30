"""OpenAI SDK 기반 AI Provider Adapter.

서비스 계층은 ``ProviderClient``의 ``respond``/``stream``만 사용한다. OpenAI는 Responses
API, vLLM 등 자체 추론 서버는 OpenAI 호환 Chat Completions API로 호출한다. HTTP 연결
풀링·SSE 파싱·일시 오류 재시도는 SDK가 담당하고, 이 모듈은 요청 조립과 응답 완결성
검증만 맡는다.

환자 문맥은 호출자가 만든 비식별 프롬프트만 전달한다. API Key와 Provider 원문 오류는
이 모듈 밖으로 노출하지 않으며, ``OpenAIProviderError`` 메시지는 로그에 남겨도 되는
분류 문자열만 담는다. Responses API의 기본 보관을 피하기 위해 ``store: false``를 보낸다.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Mapping, Protocol

import openai
from openai import AsyncOpenAI


class OpenAIProviderError(RuntimeError):
    """안전하게 분류된 Provider 호출 또는 응답 검증 실패다."""


@dataclass(frozen=True)
class OpenAIRequest:
    """Provider에 전달할 고정 지시와 비식별 입력이다."""

    instructions: str
    input: str
    output_schema: Mapping[str, object] | None = None
    output_schema_name: str = "patient_context_summary"


@dataclass(frozen=True)
class ProviderAnswer:
    """검증된 모델 텍스트 응답이다."""

    answer: str


class ProviderClient(Protocol):
    """서비스 계층이 의존하는 AI Provider 호출 경계."""

    async def respond(self, request: OpenAIRequest) -> ProviderAnswer: ...

    def stream(self, request: OpenAIRequest) -> AsyncIterator[str]: ...


class OpenAIResponsesClient:
    """OpenAI Responses API 요청 조립과 텍스트 응답 검증을 담당한다."""

    _LABEL = "OpenAI"

    def __init__(
        self,
        client: AsyncOpenAI,
        model: str,
        *,
        reasoning_effort: str | None = None,
        max_output_tokens: int | None = None,
    ) -> None:
        self._client = client
        self._model = model
        self._reasoning_effort = reasoning_effort
        self._max_output_tokens = max_output_tokens

    async def respond(self, request: OpenAIRequest) -> ProviderAnswer:
        params = self._params(request)
        with _classified_errors(self._LABEL):
            response = await self._client.responses.create(**params)
        return _answer(response.output_text, self._LABEL)

    async def stream(self, request: OpenAIRequest) -> AsyncIterator[str]:
        """Yield only ``response.output_text.delta`` text; event names stay here."""
        params = self._params(request)
        completed = yielded = False
        with _classified_errors(self._LABEL):
            async with await self._client.responses.create(**params, stream=True) as events:
                async for event in events:
                    if event.type == "response.output_text.delta":
                        if event.delta:
                            yielded = True
                            yield event.delta
                    elif event.type == "response.completed":
                        completed = True
                        break
                    elif event.type in {"error", "response.failed", "response.incomplete"}:
                        raise OpenAIProviderError("OpenAI streaming response failed")
        if not completed or not yielded:
            raise OpenAIProviderError("OpenAI returned an incomplete or empty streaming answer")

    def _params(self, request: OpenAIRequest) -> dict[str, Any]:
        _require_text(request, self._LABEL)
        params: dict[str, Any] = {
            "model": self._model,
            "instructions": request.instructions,
            "input": request.input,
            "store": False,
        }
        if self._reasoning_effort is not None:
            # Reasoning tokens are generated before any visible text, so the
            # effort level directly controls time-to-first-token.
            params["reasoning"] = {"effort": self._reasoning_effort}
        if self._max_output_tokens is not None:
            params["max_output_tokens"] = self._max_output_tokens
        if request.output_schema is not None:
            params["text"] = {"format": {
                "type": "json_schema",
                "name": request.output_schema_name,
                "strict": True,
                "schema": request.output_schema,
            }}
        return params


class OpenAICompatibleChatClient:
    """vLLM 등 OpenAI 호환 Chat Completions 요청 조립과 응답 검증을 담당한다.

    지시문은 ``system``, 비식별 입력은 ``user`` 메시지로 보낸다. 구조화 출력은
    ``response_format``의 JSON Schema로 요청하며 vLLM은 이를 guided decoding으로 강제한다.
    ``disable_thinking``은 Qwen3 등 사고 모드가 있는 chat template에
    ``enable_thinking=false``를 전달해 사고 토큰 생성 대기를 없앤다.
    """

    _LABEL = "Chat Completions"

    def __init__(
        self,
        client: AsyncOpenAI,
        model: str,
        *,
        disable_thinking: bool = True,
        max_output_tokens: int | None = None,
    ) -> None:
        self._client = client
        self._model = model
        self._disable_thinking = disable_thinking
        self._max_output_tokens = max_output_tokens

    async def respond(self, request: OpenAIRequest) -> ProviderAnswer:
        params = self._params(request)
        with _classified_errors(self._LABEL):
            completion = await self._client.chat.completions.create(**params)
        content = completion.choices[0].message.content if completion.choices else None
        return _answer(content, self._LABEL)

    async def stream(self, request: OpenAIRequest) -> AsyncIterator[str]:
        """Yield ``choices[].delta.content`` until the model reports ``stop``."""
        params = self._params(request)
        completed = yielded = False
        with _classified_errors(self._LABEL):
            async with await self._client.chat.completions.create(**params, stream=True) as chunks:
                async for chunk in chunks:
                    for choice in chunk.choices:
                        if choice.delta is not None and choice.delta.content:
                            yielded = True
                            yield choice.delta.content
                        if choice.finish_reason == "length":
                            raise OpenAIProviderError("Chat Completions answer was truncated")
                        if choice.finish_reason == "stop":
                            completed = True
        if not completed or not yielded:
            raise OpenAIProviderError("Chat Completions returned an incomplete or empty streaming answer")

    def _params(self, request: OpenAIRequest) -> dict[str, Any]:
        _require_text(request, self._LABEL)
        params: dict[str, Any] = {
            "model": self._model,
            # The system message stays byte-identical across calls so the
            # inference server can reuse its prefix cache.
            "messages": [
                {"role": "system", "content": request.instructions},
                {"role": "user", "content": request.input},
            ],
        }
        if self._max_output_tokens is not None:
            params["max_tokens"] = self._max_output_tokens
        if self._disable_thinking:
            params["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
        if request.output_schema is not None:
            params["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": request.output_schema_name,
                    "strict": True,
                    "schema": request.output_schema,
                },
            }
        return params


def _require_text(request: OpenAIRequest, label: str) -> None:
    if not request.instructions.strip() or not request.input.strip():
        raise OpenAIProviderError(f"{label} request requires instructions and input")


def _answer(text: str | None, label: str) -> ProviderAnswer:
    if not isinstance(text, str) or not text.strip():
        raise OpenAIProviderError(f"{label} returned an empty or invalid answer")
    return ProviderAnswer(text.strip())


@contextmanager
def _classified_errors(label: str) -> Iterator[None]:
    """Replace SDK errors, which may carry provider bodies, with safe categories."""
    try:
        yield
    except OpenAIProviderError:
        raise
    except openai.APIStatusError as error:
        raise OpenAIProviderError(f"{label} returned a non-success status (status={error.status_code})") from error
    except openai.APITimeoutError as error:
        raise OpenAIProviderError(f"{label} request timed out") from error
    except Exception as error:
        raise OpenAIProviderError(f"{label} transport failed") from error
