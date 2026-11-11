"""준비된 환자 EMR 문맥의 프로세스 메모리 저장소."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Mapping

from cachetools import TTLCache

from ax_g_ai.errors import PatientContextNotReadyError
from ax_g_ai.services.clinical_recommendation import RuleInput


@dataclass(frozen=True)
class PreparedRawEmrContext:
    """원문에서 허용 목록만 추린, Provider에 그대로 보낼 수 있는 비식별 문맥이다.

    ``adapters.main_bridge.RawEmrPayloadMapper``가 만들고 이 저장소가 보관한다.
    """

    data: Mapping[str, object]
    updated_at: datetime
    # Corpus concept facts for stage-2 rule evaluation (concept ids, values, KCD codes only).
    rule_input: RuleInput | None = None


class PatientContextStore:
    """비식별 문맥을 환자 ID별로 이 프로세스 메모리에만, 보관 기간 동안 둔다.

    보관 기간이 지나거나 최대 개수를 넘으면 오래된 문맥부터 사라지고, 이후 채팅은
    ``PATIENT_CONTEXT_NOT_READY``로 재준비를 요구한다. 엔드포인트가 이벤트 루프 한
    스레드에서 실행되므로 별도 잠금은 두지 않는다.
    """

    def __init__(self, ttl_seconds: float = 12 * 60 * 60, max_entries: int = 1000) -> None:
        self._contexts: TTLCache[str, PreparedRawEmrContext] = TTLCache(maxsize=max_entries, ttl=ttl_seconds)

    def put(self, patient_id: str, context: PreparedRawEmrContext) -> None:
        self._contexts[patient_id] = context

    def get(self, patient_id: str) -> PreparedRawEmrContext:
        try:
            return self._contexts[patient_id]
        except KeyError as error:
            raise PatientContextNotReadyError() from error

    def discard(self, patient_id: str) -> None:
        self._contexts.pop(patient_id, None)
