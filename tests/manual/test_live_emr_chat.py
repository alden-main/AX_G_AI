"""`emr_test.json`을 OpenAI에 보내 실제 답변을 눈으로 확인하는 수동 테스트.

기본 테스트 묶음에서는 절대 네트워크를 호출하지 않는다. 실행하려면
``AX_G_RUN_LIVE_EMR_CHAT=true``와 OpenAI 환경 변수를 명시적으로 제공해야 한다.
질문은 ``AX_G_EMR_CHAT_QUESTION``으로 바꿀 수 있다.
"""

from __future__ import annotations

from datetime import date
import json
import os
from pathlib import Path
import unittest

from ax_g_ai.adapters.openai import OpenAIResponsesClient, UrllibHttpTransport
from ax_g_ai.adapters.main_bridge import MainBridgePayloadMapper
from ax_g_ai.config import OpenAISettings
from ax_g_ai.domain.patient_context import Encounter, PatientContextBuilder
from ax_g_ai.services.access_policy import AccessRequest, ContextGuard
from ax_g_ai.services.audit import InMemoryAuditSink
from ax_g_ai.services.clinical_chat import ClinicalChatRequest, ClinicalChatService
from ax_g_ai.services.evaluation import ChatVersion, EvaluationGate, EvaluationRecord
from ax_g_ai.services.knowledge import ApprovedKnowledgeStore, KnowledgeDocument


class _AllowDemoAccess:
    def is_allowed(self, request: AccessRequest) -> bool:
        return True


class _DiscardSessionState:
    def clear(self, session_id: str) -> None:
        pass


@unittest.skipUnless(
    os.getenv("AX_G_RUN_LIVE_EMR_CHAT", "").lower() == "true",
    "Set AX_G_RUN_LIVE_EMR_CHAT=true to send the dummy fixture to OpenAI.",
)
class LiveEmrChatTest(unittest.TestCase):
    """더미 EMR 문맥으로 OpenAI의 실제 자연어 답변을 출력한다."""

    def test_prints_answer_for_the_fixture_question(self) -> None:
        fixture = Path(__file__).parents[2] / "emr_test.json"
        bridge_payload = json.loads(fixture.read_text(encoding="utf-8"))
        mapper = MainBridgePayloadMapper("local-demo-hospital")
        patient_context = PatientContextBuilder(mapper.code_mappings()).build(
            mapper.to_patient_context_payload(bridge_payload)
        )

        today = date.today()
        settings = OpenAISettings.from_environment()
        version = ChatVersion("openai", settings.model, "fixture-demo-v1", "fixture-demo-guide-v1")
        audit = InMemoryAuditSink()
        guard = ContextGuard(_AllowDemoAccess(), _DiscardSessionState(), audit)
        verified = guard.activate(
            "local-fixture-demo",
            AccessRequest(
                patient_context.actor,
                patient_context.patient_id,
                Encounter(
                    patient_context.encounter.encounter_id,
                    patient_context.encounter.encounter_type,
                    patient_context.encounter.status,
                ),
                "fixture-demo-request",
            ),
        )
        knowledge_id = "fixture-demo-safety-guide"
        service = ClinicalChatService(
            guard,
            ApprovedKnowledgeStore((
                KnowledgeDocument(knowledge_id, "더미 EMR 요약 시험", "v1", today,
                                  "시험 데이터 검토", True, today, None),
            )),
            OpenAIResponsesClient(UrllibHttpTransport(), settings.api_key, settings.model, settings.timeout_seconds),
            audit,
            EvaluationGate((
                EvaluationRecord(version, "fixture-demo-set", today, "local-demo", True, "local-only"),
            )),
            version,
        )
        question = os.getenv(
            "AX_G_EMR_CHAT_QUESTION", "해당 환자의 최근 vital 정보를 요약해줘."
        )

        response = service.answer(
            verified,
            patient_context,
            ClinicalChatRequest(question, "ko", (knowledge_id,), "fixture-demo-request"),
            on_date=today,
        )

        print(f"\n[질문]\n{question}\n\n[AI 답변]\n{response.answer}\n")
        self.assertTrue(response.answer.strip())


if __name__ == "__main__":
    unittest.main(verbosity=2)
