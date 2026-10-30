from __future__ import annotations

import unittest

from fastapi.testclient import TestClient

from ax_g_ai.adapters.main_bridge import RawEmrPayloadMapper
from ax_g_ai.api import create_app
from ax_g_ai.errors import ClinicalChatProviderUnavailableError, PatientContextNotReadyError
from ax_g_ai.services.patient_context_store import PatientContextStore
from ax_g_ai.services.prepared_patient_summary import PreparedPatientSummary


class Service:
    def __init__(self) -> None:
        self.called = False
        self.fail_summary = False
        self.question = ""
        self.reference_record: str | None = None

    async def summarize_prepared_snapshot(self, *, snapshot: object) -> PreparedPatientSummary:
        self.called = True
        if self.fail_summary:
            raise ClinicalChatProviderUnavailableError()
        return PreparedPatientSummary(
            message="EMR 데이터가 로드되었습니다.",
            sections=(),
            guidance="현재 기록을 함께 확인하세요.",
        )

    async def answer_prepared_snapshot(
        self, *, question: str, snapshot: object, reference_record: str | None = None
    ) -> str:
        self.called = True
        self.question = question
        self.reference_record = reference_record
        return "최근 혈당 추이를 확인하세요."

    async def stream_prepared_snapshot(
        self, *, question: str, snapshot: object, reference_record: str | None = None
    ):
        self.called = True
        self.question = question
        self.reference_record = reference_record
        yield "최근 혈당 "
        yield "추이를 확인하세요."

    async def stream_summary_once(self, *, snapshot: object):
        self.called = True
        yield "혈당 기록이 "
        yield "있습니다."


def emr_payload(patient_id: str = "pid-4356") -> dict[str, object]:
    return {
        "patient": {"patientId": patient_id},
        "bloodSugarList": [{"measuredAt": "2026-09-22 10:30:00", "glucoseValue": 120}],
    }


def app_client(service: object | None, store: PatientContextStore | None = None) -> TestClient:
    return TestClient(create_app(service, RawEmrPayloadMapper(), store))  # type: ignore[arg-type]


def is_ready(store: PatientContextStore, patient_id: str) -> bool:
    try:
        store.get(patient_id)
    except PatientContextNotReadyError:
        return False
    return True


class ClinicalChatApiTest(unittest.TestCase):
    def setUp(self) -> None:
        self.service = Service()
        self.client = app_client(self.service)
        self.headers: dict[str, str] = {}

    def test_prepare_then_chat_uses_only_the_new_contract(self) -> None:
        prepared = self.client.post("/internal/v1/patient-context", headers=self.headers, json={"patient_id": "pid-4356", "emr_payload": emr_payload()})
        self.assertEqual(prepared.status_code, 200)
        self.assertEqual(prepared.json(), {"patient_id": "pid-4356", "status": "ready", "patient_summary": {"message": "EMR 데이터가 로드되었습니다.", "sections": [], "guidance": "현재 기록을 함께 확인하세요."}, "emr_updated_at": "2026-09-22T10:30:00+09:00"})
        response = self.client.post("/internal/v1/clinical-chat", headers=self.headers, json={"patient_id": "pid-4356", "question": "최근 혈당은 어떤가요?"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"answer": "최근 혈당 추이를 확인하세요.", "source": "EMR 데이터베이스", "emr_updated_at": "2026-09-22T10:30:00+09:00"})
        self.assertEqual(self.service.question, "최근 혈당은 어떤가요?")

    def test_chat_before_prepare_returns_not_ready_without_provider_call(self) -> None:
        response = self.client.post("/internal/v1/clinical-chat", headers=self.headers, json={"patient_id": "pid-4356", "question": "질문"})
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["code"], "PATIENT_CONTEXT_NOT_READY")
        self.assertFalse(self.service.called)

    def test_stream_chat_emits_normalized_sse_events(self) -> None:
        self.client.post("/internal/v1/patient-context", json={"patient_id": "pid-4356", "emr_payload": emr_payload()})

        response = self.client.post("/internal/v1/clinical-chat/stream", json={"patient_id": "pid-4356", "question": "복약지도 가이드를 생성해줘"})

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.headers["content-type"].startswith("text/event-stream"))
        self.assertEqual(response.headers["x-accel-buffering"], "no")
        self.assertEqual(response.text, (
            'event: meta\ndata: {"source":"EMR 데이터베이스","emr_updated_at":"2026-09-22T10:30:00+09:00"}\n\n'
            'event: delta\ndata: {"text":"최근 혈당 "}\n\n'
            'event: delta\ndata: {"text":"추이를 확인하세요."}\n\n'
            'event: done\ndata: {}\n\n'
        ))

    def test_stream_chat_before_prepare_returns_json_not_ready(self) -> None:
        response = self.client.post("/internal/v1/clinical-chat/stream", json={"patient_id": "pid-4356", "question": "질문"})
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["code"], "PATIENT_CONTEXT_NOT_READY")

    def test_stream_provider_failure_before_and_after_first_delta(self) -> None:
        class FailingService(Service):
            after_first = False

            async def stream_prepared_snapshot(self, **kwargs):
                if self.after_first:
                    yield "부분 답변"
                raise ClinicalChatProviderUnavailableError()

        service = FailingService()
        client = app_client(service)
        client.post("/internal/v1/patient-context", json={"patient_id": "pid-4356", "emr_payload": emr_payload()})
        body = {"patient_id": "pid-4356", "question": "질문"}
        response = client.post("/internal/v1/clinical-chat/stream", json=body)
        self.assertEqual(response.status_code, 502)
        self.assertEqual(response.json()["code"], "AI_PROVIDER_UNAVAILABLE")
        service.after_first = True
        response = client.post("/internal/v1/clinical-chat/stream", json=body)
        self.assertEqual(response.status_code, 200)
        self.assertIn('event: delta\ndata: {"text":"부분 답변"}', response.text)
        self.assertIn('event: error\ndata: {"code":"AI_PROVIDER_UNAVAILABLE"', response.text)
        self.assertNotIn("event: done", response.text)

    def test_stream_prepare_makes_context_ready_before_the_summary_finishes(self) -> None:
        response = self.client.post("/internal/v1/patient-context/stream", json={"patient_id": "pid-4356", "emr_payload": emr_payload()})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.text, (
            'event: meta\ndata: {"patient_id":"pid-4356","status":"ready","source":"EMR 데이터베이스","emr_updated_at":"2026-09-22T10:30:00+09:00"}\n\n'
            'event: delta\ndata: {"text":"혈당 기록이 "}\n\n'
            'event: delta\ndata: {"text":"있습니다."}\n\n'
            'event: done\ndata: {}\n\n'
        ))
        chat = self.client.post("/internal/v1/clinical-chat", json={"patient_id": "pid-4356", "question": "질문"})
        self.assertEqual(chat.status_code, 200)

    def test_stream_prepare_discards_context_when_the_summary_fails(self) -> None:
        class FailingSummaryService(Service):
            after_first = False

            async def stream_summary_once(self, *, snapshot: object):
                if self.after_first:
                    yield "부분 요약"
                raise ClinicalChatProviderUnavailableError()

        service = FailingSummaryService()
        store = PatientContextStore()
        client = app_client(service, store)
        body = {"patient_id": "pid-4356", "emr_payload": emr_payload()}

        response = client.post("/internal/v1/patient-context/stream", json=body)
        self.assertEqual(response.status_code, 502)
        self.assertEqual(response.json()["code"], "AI_PROVIDER_UNAVAILABLE")
        self.assertFalse(is_ready(store, "pid-4356"))

        service.after_first = True
        response = client.post("/internal/v1/patient-context/stream", json=body)
        self.assertEqual(response.status_code, 200)
        self.assertIn("event: error", response.text)
        self.assertFalse(is_ready(store, "pid-4356"))

    def test_stream_prepare_and_one_shot_summary_reject_invalid_emr_as_json(self) -> None:
        for path in ("/internal/v1/patient-context/stream", "/internal/v1/emr-summary/stream"):
            mismatch = self.client.post(path, json={"patient_id": "pid-4356", "emr_payload": emr_payload("other")})
            unavailable = self.client.post(path, json={"patient_id": "pid-4356", "emr_payload": {}})
            self.assertEqual((mismatch.status_code, mismatch.json()["code"]), (403, "PATIENT_MISMATCH"))
            self.assertEqual((unavailable.status_code, unavailable.json()["code"]), (502, "EMR_UNAVAILABLE"))

    def test_one_shot_summary_stream_does_not_prepare_chat_context(self) -> None:
        store = PatientContextStore()
        client = app_client(self.service, store)

        response = client.post("/internal/v1/emr-summary/stream", json={"patient_id": "pid-4356", "emr_payload": emr_payload()})

        self.assertEqual(response.status_code, 200)
        self.assertIn('event: delta\ndata: {"text":"혈당 기록이 "}', response.text)
        self.assertTrue(response.text.endswith("event: done\ndata: {}\n\n"))
        self.assertFalse(is_ready(store, "pid-4356"))

    def test_unauthenticated_internal_network_request_and_invalid_request(self) -> None:
        response = self.client.post("/internal/v1/patient-context", json={"patient_id": "pid-4356", "emr_payload": emr_payload()})
        self.assertEqual(response.status_code, 200)
        with self.assertLogs("uvicorn.error", level="WARNING") as logs:
            invalid = self.client.post(
                "/internal/v1/clinical-chat",
                headers={"X-Request-ID": "diagnostic-request-1"},
                json={"patient_id": "pid-4356", "question": " "},
            )
        self.assertEqual(invalid.status_code, 400)
        self.assertEqual(invalid.json()["code"], "INVALID_REQUEST")
        self.assertEqual(invalid.headers["X-Request-ID"], "diagnostic-request-1")
        self.assertTrue(any("body.question:string_too_short" in line for line in logs.output))
        self.assertTrue(any("request_id=diagnostic-request-1 status=400" in line for line in logs.output))
        extra = self.client.post("/internal/v1/clinical-chat", headers=self.headers, json={"patient_id": "pid-4356", "question": "질문", "language": "ko"})
        self.assertEqual(extra.status_code, 400)

    def test_prepare_requires_raw_payload_rejects_legacy_snapshot_and_patient_mismatch(self) -> None:
        missing = self.client.post("/internal/v1/patient-context", headers=self.headers, json={"patient_id": "pid-4356"})
        self.assertEqual(missing.status_code, 400)
        legacy = self.client.post("/internal/v1/patient-context", headers=self.headers, json={"patient_id": "pid-4356", "emr_snapshot": {}})
        self.assertEqual(legacy.status_code, 400)
        mismatch = self.client.post("/internal/v1/patient-context", headers=self.headers, json={"patient_id": "pid-4356", "emr_payload": emr_payload("other")})
        self.assertEqual(mismatch.status_code, 403)
        self.assertEqual(mismatch.json()["code"], "PATIENT_MISMATCH")

    def test_emr_schema_failure_logs_the_missing_field_without_logging_payload(self) -> None:
        with self.assertLogs("uvicorn.error", level="WARNING") as logs:
            response = self.client.post(
                "/internal/v1/patient-context",
                headers={"X-Request-ID": "emr-schema-request-1"},
                json={"patient_id": "pid-4356", "emr_payload": {}},
            )

        self.assertEqual(response.status_code, 502)
        self.assertTrue(
            any(
                "request_id=emr-schema-request-1" in line
                and "failure=emr_payload.patient:missing_or_not_object" in line
                for line in logs.output
            )
        )

    def test_soap_summary_uses_the_existing_clinical_chat_endpoint(self) -> None:
        self.client.post("/internal/v1/patient-context", json={"patient_id": "pid-4356", "emr_payload": emr_payload()})
        response = self.client.post("/internal/v1/clinical-chat", json={
            "patient_id": "pid-4356",
            "question": "현재 EMR을 SOAP로 요약해줘",
        })
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["answer"], "최근 혈당 추이를 확인하세요.")
        self.assertEqual(self.service.question, "현재 EMR을 SOAP로 요약해줘")

    def test_chat_passes_a_request_only_reference_record_without_storing_it(self) -> None:
        self.client.post("/internal/v1/patient-context", json={"patient_id": "pid-4356", "emr_payload": emr_payload()})
        response = self.client.post("/internal/v1/clinical-chat", json={
            "patient_id": "pid-4356",
            "question": "이전 SOAP를 참고해 SOAP 초안을 작성해줘",
            "reference_record": "S: 야간 기침 지속\nO: 체온 37.2℃\nA: 경과 관찰\nP: 활력징후 재확인",
        })

        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.service.reference_record, "S: 야간 기침 지속\nO: 체온 37.2℃\nA: 경과 관찰\nP: 활력징후 재확인")

    def test_the_separate_soap_draft_endpoint_is_not_exposed(self) -> None:
        response = self.client.post("/internal/v1/soap-drafts", json={})
        self.assertEqual(response.status_code, 404)


    def test_failed_refresh_discards_the_previous_context(self) -> None:
        store = PatientContextStore()
        client = app_client(self.service, store)
        body = {"patient_id": "pid-4356", "emr_payload": emr_payload()}
        self.assertEqual(client.post("/internal/v1/patient-context", json=body).status_code, 200)

        self.service.fail_summary = True
        response = client.post("/internal/v1/patient-context", json=body)

        self.assertEqual((response.status_code, response.json()["code"]), (502, "AI_PROVIDER_UNAVAILABLE"))
        self.assertFalse(is_ready(store, "pid-4356"))

    def test_unconfigured_app_fails_closed_without_storing_context(self) -> None:
        store = PatientContextStore()
        client = app_client(None, store)

        prepared = client.post("/internal/v1/patient-context", json={"patient_id": "pid-4356", "emr_payload": emr_payload()})
        streamed = client.post("/internal/v1/patient-context/stream", json={"patient_id": "pid-4356", "emr_payload": emr_payload()})

        self.assertEqual((prepared.status_code, prepared.json()["code"]), (502, "AI_PROVIDER_UNAVAILABLE"))
        self.assertEqual((streamed.status_code, streamed.json()["code"]), (502, "AI_PROVIDER_UNAVAILABLE"))
        self.assertFalse(is_ready(store, "pid-4356"))
        self.assertEqual(client.get("/health").json(), {"status": "ok"})


if __name__ == "__main__":
    unittest.main()
