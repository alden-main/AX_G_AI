"""D-09의 최소 구조화 감사 이벤트 계약.

감사 이벤트는 접근·문맥 전환·오류 결과를 추적하되, 원문 PHI, 인증 토큰, API key,
질문·답변 원문을 저장하지 않는다. 영속 저장소와 보존 기간은 보안 승인이 난 뒤
``AuditSink`` 구현체로 추가한다.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Protocol


class AuditEventType(str, Enum):
    ACCESS_GRANTED = "access_granted"
    ACCESS_DENIED = "access_denied"
    CONTEXT_SWITCHED = "context_switched"
    CONTEXT_CLEARED = "context_cleared"
    AI_REQUEST_STARTED = "ai_request_started"
    AI_REQUEST_SUCCEEDED = "ai_request_succeeded"
    AI_REQUEST_FAILED = "ai_request_failed"


@dataclass(frozen=True)
class AuditEvent:
    """민감 원문을 포함하지 않는 접근·문맥 감사 레코드.

    Attributes:
        event_type: 감사 대상 행위 종류를 나타내는 ``AuditEventType``이다.
        occurred_at: UTC 시간대가 포함된 이벤트 발생 ``datetime``이다.
        actor_id: 접근을 시도한 의료진 또는 시스템 사용자 식별자다.
        organization_id: 기관 격리와 추적에 사용하는 기관 식별자다.
        patient_id: 대상 환자 식별자다. 감사 목적 외 화면 로그에 노출하면 안 된다.
        encounter_id: 대상 진료 에피소드 식별자다.
        request_id: 상위 요청과 연계하는 추적 식별자다.
        outcome: ``granted``, ``denied``, ``cleared`` 중 처리 결과 문자열이다.
        reason_code: 원문 오류 대신 기록하는 안전한 분류 코드다.
    """

    event_type: AuditEventType
    occurred_at: datetime
    actor_id: str
    organization_id: str
    patient_id: str
    encounter_id: str
    request_id: str
    outcome: str
    reason_code: str


class AuditSink(Protocol):
    def record(self, event: AuditEvent) -> None: ...


class InMemoryAuditSink:
    """단위 시험과 로컬 개발 전용의 비영속 감사 저장소."""

    def __init__(self) -> None:
        self.events: list[AuditEvent] = []

    def record(self, event: AuditEvent) -> None:
        self.events.append(event)


def audit_event(
    event_type: AuditEventType,
    *,
    actor_id: str,
    organization_id: str,
    patient_id: str,
    encounter_id: str,
    request_id: str,
    outcome: str,
    reason_code: str,
) -> AuditEvent:
    """현재 UTC 시각을 포함한 원문 비포함 감사 이벤트를 만든다."""
    return AuditEvent(
        event_type=event_type,
        occurred_at=datetime.now(timezone.utc),
        actor_id=actor_id,
        organization_id=organization_id,
        patient_id=patient_id,
        encounter_id=encounter_id,
        request_id=request_id,
        outcome=outcome,
        reason_code=reason_code,
    )
