from __future__ import annotations

from datetime import datetime
import unittest

from ax_g_ai.adapters.emr import EmrReadRequest
from ax_g_ai.domain.patient_context import Actor, Encounter
from ax_g_ai.services.access_policy import (
    AccessDeniedError,
    AccessRequest,
    AuthorizedEmrReader,
    ContextGuard,
)
from ax_g_ai.services.audit import AuditEventType, InMemoryAuditSink


class FixedAuthorizer:
    def __init__(self, allowed: bool) -> None:
        self.allowed = allowed

    def is_allowed(self, request: AccessRequest) -> bool:
        return self.allowed


class RecordingSessionState:
    def __init__(self) -> None:
        self.cleared_sessions: list[str] = []

    def clear(self, session_id: str) -> None:
        self.cleared_sessions.append(session_id)


class CountingAdapter:
    def __init__(self) -> None:
        self.calls = 0

    def read_patient_context(self, request: EmrReadRequest) -> object:
        self.calls += 1
        return object()


class ContextGuardTest(unittest.TestCase):
    """D-03의 권한·환자 전환·하위 EMR 호출 차단 수용 기준을 검증한다."""

    def setUp(self) -> None:
        self.actor = Actor("clinician-1", "hospital-a", "physician", "treatment")
        self.encounter = Encounter("encounter-1", "outpatient", "open")
        self.audit = InMemoryAuditSink()
        self.state = RecordingSessionState()
        self.authorizer = FixedAuthorizer(True)
        self.guard = ContextGuard(self.authorizer, self.state, self.audit)

    def request(self, patient_id: str = "patient-1") -> AccessRequest:
        return AccessRequest(self.actor, patient_id, self.encounter, "request-1")

    def emr_request(self, patient_id: str = "patient-1") -> EmrReadRequest:
        return EmrReadRequest(
            patient_id=patient_id,
            encounter_id="encounter-1",
            from_at=datetime.fromisoformat("2026-09-01T00:00:00+09:00"),
            to_at=datetime.fromisoformat("2026-09-17T00:00:00+09:00"),
            timezone="Asia/Seoul",
            request_id="request-1",
        )

    def test_authorized_context_is_issued_and_audited(self) -> None:
        context = self.guard.activate("session-1", self.request())

        self.guard.require_current(
            context,
            session_id="session-1",
            patient_id="patient-1",
            encounter_id="encounter-1",
        )

        self.assertEqual(self.audit.events[-1].event_type, AuditEventType.ACCESS_GRANTED)
        self.assertEqual(self.audit.events[-1].outcome, "granted")

    def test_denied_access_clears_state_and_records_a_safe_event(self) -> None:
        denied_guard = ContextGuard(FixedAuthorizer(False), self.state, self.audit)

        with self.assertRaises(AccessDeniedError):
            denied_guard.activate("session-1", self.request())

        self.assertEqual(self.state.cleared_sessions, ["session-1"])
        self.assertEqual(self.audit.events[-1].event_type, AuditEventType.ACCESS_DENIED)
        self.assertEqual(self.audit.events[-1].reason_code, "access_denied")

    def test_patient_switch_invalidates_prior_context_before_emr_read(self) -> None:
        old_context = self.guard.activate("session-1", self.request("patient-1"))
        self.guard.activate("session-1", self.request("patient-2"))
        adapter = CountingAdapter()
        reader = AuthorizedEmrReader(self.guard, adapter)

        with self.assertRaises(AccessDeniedError):
            reader.read(old_context, self.emr_request("patient-1"))

        self.assertEqual(adapter.calls, 0)
        self.assertEqual(self.state.cleared_sessions, ["session-1"])
        self.assertTrue(
            any(event.event_type is AuditEventType.CONTEXT_SWITCHED for event in self.audit.events)
        )


if __name__ == "__main__":
    unittest.main()
