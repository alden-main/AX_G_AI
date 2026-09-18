from __future__ import annotations

from datetime import date
import unittest

from ax_g_ai.adapters.ai_tank import AiTankProviderError, ProviderAnswer
from ax_g_ai.domain.patient_context import Actor, CodeMapping, Encounter, PatientContextBuilder
from ax_g_ai.services.access_policy import AccessRequest, ContextGuard
from ax_g_ai.services.audit import AuditEventType, InMemoryAuditSink
from ax_g_ai.services.clinical_chat import (
    ClinicalChatRequest,
    ClinicalChatService,
    ClinicalChatUnavailableError,
)
from ax_g_ai.services.evaluation import ChatVersion, EvaluationGate, EvaluationRecord
from ax_g_ai.services.knowledge import ApprovedKnowledgeStore, KnowledgeDocument
from tests.domain.test_patient_context import payload


class AllowedAuthorizer:
    def is_allowed(self, request: AccessRequest) -> bool:
        return True


class SessionState:
    def clear(self, session_id: str) -> None:
        pass


class FailingAiTank:
    def consult(self, request: object) -> ProviderAnswer:
        raise AiTankProviderError("provider failed")


class RecordingAiTank:
    def __init__(self) -> None:
        self.request = None

    def consult(self, request: object) -> ProviderAnswer:
        self.request = request
        return ProviderAnswer("요약 답변")


class ClinicalChatSafetyTest(unittest.TestCase):
    """P-07의 Provider 장애 안전 실패와 감사 추적을 검증한다."""

    def test_provider_failure_records_no_question_or_answer_and_returns_safe_error(self) -> None:
        audit = InMemoryAuditSink()
        actor = Actor("clinician-1", "hospital-a", "physician", "treatment")
        encounter = Encounter("encounter-1", "outpatient", "open")
        guard = ContextGuard(AllowedAuthorizer(), SessionState(), audit)
        verified = guard.activate("session-1", AccessRequest(actor, "patient-1", encounter, "access-1"))
        context = PatientContextBuilder(
            [CodeMapping("hospital-lab-v1", "GLU", "LOINC", "2345-7")]
        ).build(payload())
        knowledge = ApprovedKnowledgeStore([
            KnowledgeDocument("guide-1", "당뇨 지침", "2026.1", date(2026, 1, 1), "3장", True, date(2026, 1, 1), None)
        ])
        version = ChatVersion("ai-tank", "test-model", "prompt-1", "knowledge-1")
        gate = EvaluationGate([
            EvaluationRecord(version, "eval-1", date(2026, 9, 1), "reviewer-1", True, "approval-1")
        ])
        service = ClinicalChatService(guard, knowledge, FailingAiTank(), audit, gate, version)

        with self.assertRaises(ClinicalChatUnavailableError):
            service.answer(
                verified,
                context,
                ClinicalChatRequest("환자 이름을 포함한 질문", "ko", ("guide-1",), "chat-1"),
                on_date=date(2026, 9, 17),
            )

        self.assertEqual(audit.events[-1].event_type, AuditEventType.AI_REQUEST_FAILED)
        self.assertNotIn("환자 이름", str(audit.events[-1]))

    def test_sends_the_deidentified_emr_summary_with_the_clinician_question(self) -> None:
        audit = InMemoryAuditSink()
        actor = Actor("clinician-1", "hospital-a", "physician", "treatment")
        encounter = Encounter("encounter-1", "outpatient", "open")
        guard = ContextGuard(AllowedAuthorizer(), SessionState(), audit)
        verified = guard.activate("session-1", AccessRequest(actor, "patient-1", encounter, "access-1"))
        context = PatientContextBuilder([
            CodeMapping("hospital-lab-v1", "GLU", "LOINC", "2345-7")
        ]).build(payload())
        knowledge = ApprovedKnowledgeStore([
            KnowledgeDocument("guide-1", "당뇨 지침", "2026.1", date(2026, 1, 1), "3장", True, date(2026, 1, 1), None)
        ])
        version = ChatVersion("ai-tank", "test-model", "prompt-1", "knowledge-1")
        gate = EvaluationGate([
            EvaluationRecord(version, "eval-1", date(2026, 9, 1), "reviewer-1", True, "approval-1")
        ])
        ai_tank = RecordingAiTank()
        service = ClinicalChatService(guard, knowledge, ai_tank, audit, gate, version)  # type: ignore[arg-type]

        response = service.answer(
            verified, context,
            ClinicalChatRequest("최근 혈당을 요약해 주세요.", "ko", ("guide-1",), "chat-1"),
            on_date=date(2026, 9, 17),
        )

        self.assertEqual(response.answer, "요약 답변")
        self.assertIn("최근 혈당을 요약해 주세요.", ai_tank.request.question)
        self.assertIn('"code":"2345-7"', ai_tank.request.question)
        self.assertNotIn("patient-1", ai_tank.request.question)
        self.assertNotIn("observation-1", ai_tank.request.question)


if __name__ == "__main__":
    unittest.main()
