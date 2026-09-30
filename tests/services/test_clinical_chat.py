from __future__ import annotations

from datetime import date
import unittest

from ax_g_ai.adapters.openai import OpenAIProviderError, ProviderAnswer
from ax_g_ai.domain.patient_context import Actor, CodeMapping, Encounter, PatientContextBuilder
from ax_g_ai.services.access_policy import AccessRequest, ContextGuard
from ax_g_ai.services.audit import AuditEventType, InMemoryAuditSink
from ax_g_ai.services.clinical_chat import (
    ClinicalChatRequest,
    ClinicalChatService,
    ClinicalChatUnavailableError,
    compose_prepared_snapshot_question,
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


class FailingOpenAI:
    def respond(self, request: object) -> ProviderAnswer:
        raise OpenAIProviderError("provider failed")


class RecordingOpenAI:
    def __init__(self) -> None:
        self.request = None

    def respond(self, request: object) -> ProviderAnswer:
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
        version = ChatVersion("openai", "test-model", "prompt-1", "knowledge-1")
        gate = EvaluationGate([
            EvaluationRecord(version, "eval-1", date(2026, 9, 1), "reviewer-1", True, "approval-1")
        ])
        service = ClinicalChatService(guard, knowledge, FailingOpenAI(), audit, gate, version)  # type: ignore[arg-type]

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
        version = ChatVersion("openai", "test-model", "prompt-1", "knowledge-1")
        gate = EvaluationGate([
            EvaluationRecord(version, "eval-1", date(2026, 9, 1), "reviewer-1", True, "approval-1")
        ])
        openai = RecordingOpenAI()
        service = ClinicalChatService(guard, knowledge, openai, audit, gate, version)  # type: ignore[arg-type]

        response = service.answer(
            verified, context,
            ClinicalChatRequest("최근 혈당을 요약해 주세요.", "ko", ("guide-1",), "chat-1"),
            on_date=date(2026, 9, 17),
        )

        self.assertEqual(response.answer, "요약 답변")
        self.assertIn("최근 혈당을 요약해 주세요.", openai.request.input)
        self.assertIn('"code":"2345-7"', openai.request.input)
        self.assertNotIn("patient-1", openai.request.input)
        self.assertNotIn("observation-1", openai.request.input)

    def test_prepared_snapshot_prompt_is_korean_and_excludes_patient_identifiers(self) -> None:
        prompt = compose_prepared_snapshot_question(
            "최근 혈당을 확인해 주세요.",
            {
                "patient": {"patient_id": "patient-1", "name": "홍길동"},
                "blood_sugar": [{"measured_at": "2026-09-22T10:30:00+09:00", "value": 120, "unit": "mg/dL"}],
                "lab_results": [], "prescriptions": [],
            },
        )

        self.assertIn("반드시 한국어로 답변", prompt)
        self.assertIn('"blood_sugar"', prompt)
        self.assertNotIn("patient-1", prompt)
        self.assertNotIn("홍길동", prompt)
        self.assertNotIn("obs-1", prompt)


if __name__ == "__main__":
    unittest.main()
