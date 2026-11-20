"""Pydantic AI 기반 AI Provider Adapter.

서비스 계층은 ``ProviderClient``의 ``respond``·``respond_structured``·``stream``만 사용한다. 요청 조립(Responses API의
``instructions``·Chat Completions의 ``system`` 메시지, JSON Schema 구조화 출력), SSE 파싱, OpenTelemetry
계측은 Pydantic AI가 담당하고, 이 모듈은 응답 완결성 검증과 오류 분류만 맡는다.

환자 문맥은 호출자가 만든 비식별 프롬프트만 전달한다. API Key와 Provider 원문 오류는
이 모듈 밖으로 노출하지 않으며, ``OpenAIProviderError`` 메시지는 로그에 남겨도 되는
분류 문자열만 담는다.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from contextlib import contextmanager
from typing import Any

import openai
from pydantic_ai import Agent, InstrumentationSettings, NativeOutput
from pydantic_ai.capabilities import Instrumentation
from pydantic_ai.exceptions import ModelAPIError, ModelHTTPError, UnexpectedModelBehavior
from pydantic_ai.messages import ModelResponse
from pydantic_ai.models import Model

# The port is defined by the service layer; re-exported here for callers that build clients.
from ax_g_ai.services.ports import (
    OpenAIProviderError,
    OpenAIRequest,
    OutputT,
    ProviderAnswer,
    ProviderClient,
    ProviderOutputError,
)

__all__ = [
    "OpenAIProviderError", "OpenAIRequest", "ProviderAnswer", "ProviderClient", "ProviderOutputError", "PydanticAIProviderClient",
]


class PydanticAIProviderClient:
    """Pydantic AI ``Agent``로 ``ProviderClient``를 구현한다.

    Provider별 차이(Responses·Chat Completions, 저장 안 함, 사고 모드)는 ``model``의 설정이 맡는다.
    호출 하나는 모델 요청 하나다: 출력 재시도(``retries=0``)를 하지 않고, ``stop``으로 끝나지 않은
    응답(토큰 한도로 잘림, 실패·중단된 스트림)은 오류로 바꾼다. 작업(``operation``)마다 이름이 같은
    Agent를 두어 추적이 ``invoke_agent <operation>``으로 남는다.
    """

    def __init__(self, model: Model, *, instrumentation: InstrumentationSettings | None = None) -> None:
        self._model = model
        self._capabilities = [Instrumentation(settings=instrumentation)] if instrumentation is not None else []
        self._agents: dict[str, Agent[None, Any]] = {}
        self._label = type(model).__name__

    async def respond(self, request: OpenAIRequest) -> ProviderAnswer:
        text = await self._run(request, str)
        if not text.strip():
            raise ProviderOutputError(f"{self._label} returned an empty answer")
        return ProviderAnswer(text.strip())

    async def respond_structured(self, request: OpenAIRequest, output_type: type[OutputT]) -> OutputT:
        # Native JSON Schema output: Responses ``text.format`` / Chat ``response_format`` (vLLM guided decoding).
        return await self._run(request, NativeOutput(output_type))

    async def stream(self, request: OpenAIRequest) -> AsyncIterator[str]:
        """Yield text deltas as they arrive (no debounce, so the first sentence shows at once)."""
        yielded = False
        with _classified_errors(self._label):
            async with self._agent(request).run_stream(request.input, **self._run_options(request)) as result:
                async for delta in result.stream_text(delta=True, debounce_by=None):
                    if delta:
                        yielded = True
                        yield delta
                response = result.response
        _require_complete(response, self._label)
        if not yielded:
            raise ProviderOutputError(f"{self._label} returned an empty streaming answer")

    async def _run(self, request: OpenAIRequest, output_type: Any) -> Any:
        with _classified_errors(self._label):
            result = await self._agent(request).run(request.input, output_type=output_type, **self._run_options(request))
        _require_complete(result.response, self._label)
        return result.output

    def _agent(self, request: OpenAIRequest) -> Agent[None, Any]:
        agent = self._agents.get(request.operation)
        if agent is None:
            agent = self._agents[request.operation] = Agent(
                self._model, name=request.operation, retries=0, capabilities=self._capabilities,
            )
        return agent

    def _run_options(self, request: OpenAIRequest) -> dict[str, Any]:
        if not request.instructions.strip() or not request.input.strip():
            raise OpenAIProviderError(f"{self._label} request requires instructions and input")
        metadata = {"operation": request.operation}
        if request.prompt_version:
            metadata["prompt_version"] = request.prompt_version
        return {"instructions": request.instructions, "metadata": metadata}


def _require_complete(response: ModelResponse, label: str) -> None:
    if response.finish_reason != "stop":
        raise ProviderOutputError(f"{label} answer was incomplete (finish_reason={response.finish_reason})")


@contextmanager
def _classified_errors(label: str) -> Iterator[None]:
    """Replace SDK/framework errors, which may carry provider bodies, with safe categories."""
    try:
        yield
    except OpenAIProviderError:
        raise
    except ModelHTTPError as error:
        raise OpenAIProviderError(f"{label} returned a non-success status (status={error.status_code})") from error
    except UnexpectedModelBehavior as error:
        raise ProviderOutputError(f"{label} returned an empty or invalid answer") from error
    except ModelAPIError as error:
        if isinstance(error.__cause__, openai.APITimeoutError):
            raise OpenAIProviderError(f"{label} request timed out") from error
        raise OpenAIProviderError(f"{label} transport failed") from error
    except Exception as error:
        raise OpenAIProviderError(f"{label} transport failed") from error
