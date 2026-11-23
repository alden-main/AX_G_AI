"""외부 시스템과 내부 도메인 사이의 경계(어댑터 패키지).

설계상 위치: 어댑터 계층의 패키지 진입점. 서비스 계층이 정의한 포트(``services.ports``)를 외부 시스템(Bridge EMR,
AI Provider, Redis, SQLite 참조 DB, OpenTelemetry)에 맞춰 구현하는 모듈들을 묶는다.

역할: 자주 쓰는 ``RawEmrPayloadMapper``(Bridge 원문 EMR → 비식별 문맥)와 Provider 호출 타입
(``ProviderClient``, ``PydanticAIProviderClient``, ``OpenAIRequest``, ``OpenAIProviderError``)을 다시 내보낸다.
"""

from .main_bridge import RawEmrPayloadMapper
from .openai import (
    OpenAIProviderError,
    OpenAIRequest,
    ProviderClient,
    PydanticAIProviderClient,
)

__all__ = [
    "OpenAIProviderError",
    "OpenAIRequest",
    "ProviderClient",
    "PydanticAIProviderClient",
    "RawEmrPayloadMapper",
]
