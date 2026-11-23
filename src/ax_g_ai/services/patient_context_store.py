"""준비된 환자 EMR 문맥과 그 저장소.

설계상 위치: 서비스 계층. EMR 매퍼(``adapters.main_bridge``)가 만든 문맥을 API의 문맥 준비 엔드포인트가 저장하고,
채팅 요청이 꺼내 쓴다(``clinical_grounding``이 ``rule_source``로 규칙 입력을 다시 만든다).

역할: ``ContextStore``는 비동기 인터페이스다. 기본 구현 ``PatientContextStore``는 이 프로세스 메모리(가장 빠름,
워커 1개용)이고, 여러 워커·인스턴스가 문맥을 공유해야 하면 ``adapters.redis_context_store``를 쓴다.
외부 저장소에는 ``context_to_json``의 JSON만 저장하며, 규칙 매핑 결과(``rule_input``)는 조회 후
``rule_source``로 다시 계산한다(약 0.1 ms).
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol

from cachetools import TTLCache

from ax_g_ai.errors import PatientContextNotReadyError
from ax_g_ai.services.clinical_recommendation import RuleInput

_FORMAT = 1


@dataclass(frozen=True)
class PreparedRawEmrContext:
    """원문에서 허용 목록만 추린, Provider에 그대로 보낼 수 있는 비식별 문맥이다.

    역할: ``adapters.main_bridge.RawEmrPayloadMapper``가 만들고 저장소가 보관한다.
    """

    data: Mapping[str, object]
    updated_at: datetime
    # 2단계 규칙 평가용 말뭉치 개념 사실(개념 ID·값·KCD 코드만). 말뭉치 릴리스 ``rule_release``의 규칙으로
    # 매핑했다(None: 파일 말뭉치).
    rule_input: RuleInput | None = None
    # 규칙 매핑이 읽는 허용 목록 EMR 필드(``adapters.main_bridge.rule_source``): 더 새 말뭉치 릴리스가 활성화되면
    # 사실을 다시 매핑할 수 있게 한다. Provider에는 절대 보내지 않는다.
    rule_source: Mapping[str, Any] = field(default_factory=dict)
    rule_release: int | None = None


def context_to_json(context: PreparedRawEmrContext) -> bytes:
    """저장 형식: 비식별 데이터·갱신 시각·규칙 원천을 담은 UTF-8 JSON 바이트(파생값 ``rule_input``은 빼고)."""
    document = {"v": _FORMAT, "data": context.data, "updated_at": context.updated_at.isoformat(), "rule_source": context.rule_source}
    return json.dumps(document, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def context_from_json(raw: bytes | str) -> PreparedRawEmrContext:
    """``context_to_json``의 JSON을 문맥으로 되돌린다. 다른 형식이거나 잘못된 문서이면 ``ValueError``를 낸다
    (필수 키가 없거나 타입이 틀리면 ``KeyError``·``TypeError``도 날 수 있다)."""
    document = json.loads(raw)
    if not isinstance(document, dict) or document.get("v") != _FORMAT:
        raise ValueError("unsupported patient context format")
    return PreparedRawEmrContext(
        data=document["data"], updated_at=datetime.fromisoformat(document["updated_at"]), rule_source=document.get("rule_source") or {},
    )


class ContextStore(Protocol):
    """환자 ID별 문맥 보관 인터페이스.

    역할: ``put``·``get``·``discard``·``close``를 정의한다. ``get``은 없거나 만료되면 ``PatientContextNotReadyError``를 낸다.
    ``backend``(``memory``·``redis``)와 ``ping``은 준비 상태 확인(``GET /ready``)용이다.
    """

    backend: str

    async def ping(self) -> bool:
        """저장소에 지금 읽고 쓸 수 있으면 True. 예외를 내지 않는다."""
        ...

    async def put(self, patient_id: str, context: PreparedRawEmrContext) -> None: ...

    async def get(self, patient_id: str) -> PreparedRawEmrContext: ...

    async def discard(self, patient_id: str) -> None: ...

    async def close(self) -> None: ...


class PatientContextStore:
    """비식별 문맥을 환자 ID별로 이 프로세스 메모리에만, 보관 기간 동안 둔다.

    역할: 보관 기간이 지나거나 최대 개수를 넘으면 오래된 문맥부터 사라지고, 이후 채팅은
    ``PATIENT_CONTEXT_NOT_READY``로 재준비를 요구한다. 엔드포인트가 이벤트 루프 한
    스레드에서 실행되므로 별도 잠금은 두지 않는다. 매핑된 ``rule_input``도 그대로 보관한다.
    """

    backend = "memory"

    def __init__(self, ttl_seconds: float = 12 * 60 * 60, max_entries: int = 1000) -> None:
        self._contexts: TTLCache[str, PreparedRawEmrContext] = TTLCache(maxsize=max_entries, ttl=ttl_seconds)

    async def ping(self) -> bool:
        return True

    async def put(self, patient_id: str, context: PreparedRawEmrContext) -> None:
        self._contexts[patient_id] = context

    async def get(self, patient_id: str) -> PreparedRawEmrContext:
        try:
            return self._contexts[patient_id]
        except KeyError as error:
            raise PatientContextNotReadyError() from error

    async def discard(self, patient_id: str) -> None:
        self._contexts.pop(patient_id, None)

    async def close(self) -> None:
        return None
