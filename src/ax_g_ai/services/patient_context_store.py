"""준비된 환자 EMR 문맥과 그 저장소.

``ContextStore``는 비동기 인터페이스다. 기본 구현 ``PatientContextStore``는 이 프로세스 메모리(가장 빠름,
워커 1개용)이고, 여러 워커·인스턴스가 문맥을 공유해야 하면 ``adapters.redis_context_store``를 쓴다.
외부 저장소에는 ``context_to_json``의 JSON만 저장하며, 규칙 매핑 결과(``rule_input``)는 조회 후
``rule_source``로 다시 계산한다(약 0.1 ms).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Mapping, Protocol

from cachetools import TTLCache

from ax_g_ai.errors import PatientContextNotReadyError
from ax_g_ai.services.clinical_recommendation import RuleInput

_FORMAT = 1


@dataclass(frozen=True)
class PreparedRawEmrContext:
    """원문에서 허용 목록만 추린, Provider에 그대로 보낼 수 있는 비식별 문맥이다.

    ``adapters.main_bridge.RawEmrPayloadMapper``가 만들고 저장소가 보관한다.
    """

    data: Mapping[str, object]
    updated_at: datetime
    # Corpus concept facts for stage-2 rule evaluation (concept ids, values, KCD codes only), mapped with
    # the rules of corpus release ``rule_release`` (None: file corpus).
    rule_input: RuleInput | None = None
    # Whitelisted EMR fields the rule mapping reads (``adapters.main_bridge.rule_source``): lets the facts be
    # mapped again when a newer corpus release is active. Never sent to the provider.
    rule_source: Mapping[str, Any] = field(default_factory=dict)
    rule_release: int | None = None


def context_to_json(context: PreparedRawEmrContext) -> bytes:
    """Stored form: the de-identified data, timestamp and rule source (not the derived ``rule_input``)."""
    document = {"v": _FORMAT, "data": context.data, "updated_at": context.updated_at.isoformat(), "rule_source": context.rule_source}
    return json.dumps(document, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def context_from_json(raw: bytes | str) -> PreparedRawEmrContext:
    """Raises ``ValueError`` for another format or a malformed document."""
    document = json.loads(raw)
    if not isinstance(document, dict) or document.get("v") != _FORMAT:
        raise ValueError("unsupported patient context format")
    return PreparedRawEmrContext(
        data=document["data"], updated_at=datetime.fromisoformat(document["updated_at"]), rule_source=document.get("rule_source") or {},
    )


class ContextStore(Protocol):
    """환자 ID별 문맥 보관. ``get``은 없거나 만료되면 ``PatientContextNotReadyError``를 낸다."""

    async def put(self, patient_id: str, context: PreparedRawEmrContext) -> None: ...

    async def get(self, patient_id: str) -> PreparedRawEmrContext: ...

    async def discard(self, patient_id: str) -> None: ...

    async def close(self) -> None: ...


class PatientContextStore:
    """비식별 문맥을 환자 ID별로 이 프로세스 메모리에만, 보관 기간 동안 둔다.

    보관 기간이 지나거나 최대 개수를 넘으면 오래된 문맥부터 사라지고, 이후 채팅은
    ``PATIENT_CONTEXT_NOT_READY``로 재준비를 요구한다. 엔드포인트가 이벤트 루프 한
    스레드에서 실행되므로 별도 잠금은 두지 않는다. 매핑된 ``rule_input``도 그대로 보관한다.
    """

    def __init__(self, ttl_seconds: float = 12 * 60 * 60, max_entries: int = 1000) -> None:
        self._contexts: TTLCache[str, PreparedRawEmrContext] = TTLCache(maxsize=max_entries, ttl=ttl_seconds)

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
