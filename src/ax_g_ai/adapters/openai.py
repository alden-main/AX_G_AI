"""Pydantic AI 기반 AI Provider 어댑터(``ProviderClient`` 구현).

설계상 위치: 어댑터 계층의 AI Provider 경계. ``adapters.provider.build_provider_client``가 Client를 만들고,
서비스 계층은 ``services.ports.ProviderClient``로만 이 구현에 의존한다.

역할: 서비스 계층은 ``ProviderClient``의 ``respond``·``respond_structured``·``stream``만 사용한다. 요청 조립(Responses API의
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

# 포트는 서비스 계층이 정의한다. Client를 만드는 호출자를 위해 여기서 다시 내보낸다.
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

    역할: Provider별 차이(Responses·Chat Completions, 저장 안 함, 사고 모드)는 ``model``의 설정이 맡는다.
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
        # 네이티브 JSON Schema 출력: Responses ``text.format`` / Chat ``response_format``(vLLM guided decoding).
        return await self._run(request, NativeOutput(output_type))

    async def stream(self, request: OpenAIRequest) -> AsyncIterator[str]:
        """텍스트 조각을 도착하는 대로 내보낸다(디바운스가 없어 첫 문장이 바로 보인다).

        역할: 작업별 Agent로 스트림을 열어 비어 있지 않은 조각만 내보내고, 끝난 뒤 종료 사유가 ``stop``인지와 조각이
        하나라도 있었는지 확인한다. SDK·프레임워크 오류는 안전한 분류 오류로 바꾼다.
        입력:
            request: 지시문·입력·작업 이름·프롬프트 판을 담은 요청.
        출력: 텍스트 조각(``str``)을 내보내는 비동기 제너레이터.
        예외:
            ProviderOutputError: 응답이 잘렸거나(``finish_reason``이 ``stop``이 아님) 조각이 하나도 없을 때.
            OpenAIProviderError: 지시문·입력이 비었거나 HTTP·전송 오류가 났을 때.
        """
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
    """SDK·프레임워크 오류(Provider 응답 본문을 담을 수 있음)를 안전한 분류 오류로 바꾼다.

    역할: ``with`` 블록 안의 예외를 분류한다. ``OpenAIProviderError``(하위 클래스 포함)는 그대로 두고, HTTP 오류는 상태
    코드만 남기며, ``UnexpectedModelBehavior``는 ``ProviderOutputError``로, 시간 초과는 별도 메시지로, 그 밖의 예외는
    전송 실패로 바꾼다. 원래 예외는 ``__cause__``로 연결된다.
    입력:
        label: 오류 메시지에 넣을 모델 클래스 이름.
    출력: 컨텍스트 관리자(값을 내보내지 않음).
    예외:
        OpenAIProviderError: 분류된 호출 실패.
        ProviderOutputError: 비었거나 잘못된 응답.
    """
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
