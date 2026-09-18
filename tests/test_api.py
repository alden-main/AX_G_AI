from __future__ import annotations

from datetime import date, datetime
import unittest

from fastapi.testclient import TestClient

from ax_g_ai.api import (
    BridgeAuthenticationError,
    ResolvedBridgeChatContext,
    ResolvedChatContext,
    SnapshotInfo,
    create_app,
)
from ax_g_ai.services.clinical_chat import (
    ClinicalChatProviderUnavailableError,
    ClinicalChatResponse,
    KnowledgeEvidence,
)


class FixedContextProvider:
    def resolve(self, request: object, request_id: str) -> ResolvedChatContext:
        return ResolvedChatContext(verified_context=None, patient_context=None)  # type: ignore[arg-type]


class FixedBridgeContextProvider:
    def resolve_internal(self, request: object, body: object, request_id: str) -> ResolvedBridgeChatContext:
        return ResolvedBridgeChatContext(
            chat_context=ResolvedChatContext(verified_context=None, patient_context=None),  # type: ignore[arg-type]
            context_id="ctx-1",
            context_info=SnapshotInfo(
                datetime.fromisoformat("2026-09-18T10:30:00+09:00"), "partial", ("lab_results",)
            ),
        )


class DeniedBridgeContextProvider:
    def resolve_internal(self, request: object, body: object, request_id: str) -> ResolvedBridgeChatContext:
        raise BridgeAuthenticationError()


class SuccessfulService:
    def __init__(self) -> None:
        self.request = None

    def answer(self, verified_context: object, patient_context: object, request: object, *, on_date: date) -> ClinicalChatResponse:
        self.request = request
        return ClinicalChatResponse(
            answer="혈당 관리 원칙입니다.",
            evidence=(KnowledgeEvidence("guide-2026-01", "지침명", "2026.1", date(2026, 1, 1), "3장 2절"),),
            limitations=("의료진 판단을 보조합니다.",),
        )


class FailingProviderService:
    def answer(self, *args: object, **kwargs: object) -> ClinicalChatResponse:
        raise ClinicalChatProviderUnavailableError("provider failed")


class ClinicalChatApiTest(unittest.TestCase):
    def test_preserves_the_documented_request_and_response_envelope(self) -> None:
        service = SuccessfulService()
        client = TestClient(create_app(service, FixedContextProvider()))

        response = client.post("/api/clinical-chat", json={
            "question": "당뇨 환자 혈당 관리 원칙을 알려주세요.",
            "language": "ko",
            "knowledge_ids": ["guide-2026-01"],
            "request_id": "trace-1",
            "history": [{"inputs": "이전 질문", "outputs": "이전 답변"}],
        })

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {
            "answer": "혈당 관리 원칙입니다.",
            "evidence": [{
                "document_id": "guide-2026-01", "title": "지침명", "version": "2026.1",
                "published_on": "2026-01-01", "citation_location": "3장 2절",
            }],
            "limitations": ["의료진 판단을 보조합니다."],
        })
        self.assertEqual(service.request.question, "당뇨 환자 혈당 관리 원칙을 알려주세요.")
        self.assertEqual(service.request.history[0].inputs, "이전 질문")

    def test_provider_failure_is_a_safe_gateway_error(self) -> None:
        client = TestClient(create_app(FailingProviderService(), FixedContextProvider()))  # type: ignore[arg-type]

        response = client.post("/api/clinical-chat", json={
            "question": "질문", "language": "ko", "knowledge_ids": ["guide-1"], "request_id": "trace-1",
        })

        self.assertEqual(response.status_code, 502)
        self.assertEqual(response.json()["detail"], "The AI provider is temporarily unavailable.")

    def test_default_app_does_not_accept_chat_without_runtime_wiring(self) -> None:
        client = TestClient(create_app())

        response = client.post("/api/clinical-chat", json={
            "question": "질문", "language": "ko", "knowledge_ids": ["guide-1"], "request_id": "trace-1",
        })

        self.assertEqual(response.status_code, 503)

    def test_internal_bridge_api_returns_context_bound_snapshot_metadata(self) -> None:
        client = TestClient(create_app(SuccessfulService(), FixedContextProvider(), FixedBridgeContextProvider()))

        response = client.post("/internal/v1/clinical-chat", headers={"X-Request-ID": "trace-1"}, json={
            "context_id": "ctx-1", "patient_id": "patient-1", "encounter_id": "encounter-1",
            "question": "질문", "language": "ko", "knowledge_ids": ["guide-1"],
        })

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["request_id"], "trace-1")
        self.assertEqual(response.json()["context_id"], "ctx-1")
        self.assertEqual(response.json()["context_info"], {
            "snapshot_at": "2026-09-18T10:30:00+09:00",
            "assembly_status": "partial", "missing_data": ["lab_results"],
        })

    def test_internal_bridge_api_never_treats_an_unauthenticated_request_as_a_context(self) -> None:
        client = TestClient(create_app(SuccessfulService(), FixedContextProvider(), DeniedBridgeContextProvider()))

        response = client.post("/internal/v1/clinical-chat", headers={"X-Request-ID": "trace-1"}, json={
            "context_id": "ctx-1", "patient_id": "patient-1", "encounter_id": "encounter-1",
            "question": "질문", "language": "ko", "knowledge_ids": ["guide-1"],
        })

        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.json(), {
            "code": "UNAUTHENTICATED", "message": "Bridge authentication failed.", "request_id": "trace-1",
        })


if __name__ == "__main__":
    unittest.main()
