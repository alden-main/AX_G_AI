"""서비스 계층이 정의하고 어댑터가 구현하는 AI Provider 경계(port).

``adapters.openai``의 Responses·Chat Completions Client가 ``ProviderClient``를 구현한다.
서비스는 이 모듈에만 의존하므로 Provider SDK를 바꿔도 서비스 코드는 바뀌지 않는다.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Mapping, Protocol


class OpenAIProviderError(RuntimeError):
    """안전하게 분류된 Provider 호출 또는 응답 검증 실패다. 메시지는 로그에 남겨도 되는 분류 문자열만 담는다."""


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
