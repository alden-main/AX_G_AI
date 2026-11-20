"""서비스 계층이 정의하고 어댑터가 구현하는 AI Provider 경계(port).

``adapters.openai``의 Pydantic AI 기반 Client가 ``ProviderClient``를 구현한다(Responses·Chat Completions).
서비스는 이 모듈에만 의존하므로 Provider SDK를 바꿔도 서비스 코드는 바뀌지 않는다.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Protocol, TypeVar

from pydantic import BaseModel

OutputT = TypeVar("OutputT", bound=BaseModel)


class OpenAIProviderError(RuntimeError):
    """안전하게 분류된 Provider 호출 또는 응답 검증 실패다. 메시지는 로그에 남겨도 되는 분류 문자열만 담는다."""


class ProviderOutputError(OpenAIProviderError):
    """호출은 끝났지만 답이 비었거나 잘렸거나 요청한 구조와 맞지 않는다."""


@dataclass(frozen=True)
class OpenAIRequest:
    """Provider에 전달할 고정 지시와 비식별 입력이다.

    ``operation``(예: ``clinical_recommendation``)과 ``prompt_version``은 추적 span의 이름·메타데이터로
    남아 어느 작업·어느 프롬프트 판이 답을 만들었는지 구분한다. 둘 다 환자 데이터를 담지 않는다.
    """

    instructions: str
    input: str
    operation: str = "provider_call"
    prompt_version: str | None = None


@dataclass(frozen=True)
class ProviderAnswer:
    """검증된 모델 텍스트 응답이다."""

    answer: str


class ProviderClient(Protocol):
    """서비스 계층이 의존하는 AI Provider 호출 경계."""

    async def respond(self, request: OpenAIRequest) -> ProviderAnswer: ...

    async def respond_structured(self, request: OpenAIRequest, output_type: type[OutputT]) -> OutputT:
        """``output_type``의 JSON Schema로 구조화 출력을 요청하고 그 모델로 검증해 돌려준다."""
        ...

    def stream(self, request: OpenAIRequest) -> AsyncIterator[str]: ...
