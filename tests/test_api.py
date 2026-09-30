from __future__ import annotations

from datetime import datetime
import unittest

from fastapi.testclient import TestClient

from ax_g_ai.api import PatientContextNotReadyError, PatientContextUnavailableError, PatientMismatchError, create_app
from ax_g_ai.services.prepared_patient_summary import PreparedPatientSummary


class Store:
    def __init__(self) -> None:
        self.prepared: dict[str, tuple[dict[str, object], datetime]] = {}

    def prepare(self, patient_id: str, emr_payload: dict[str, object]) -> tuple[PreparedPatientSummary, datetime]:
        if patient_id == "unavailable":
            raise PatientContextUnavailableError("raw EMR payload has no patient object")
        if patient_id == "mismatch":
            raise PatientMismatchError()
        updated_at = datetime.fromisoformat("2026-09-22T10:30:00+09:00")
        self.prepared[patient_id] = ({"patient": {"patient_id": patient_id}, "observations": []}, updated_at)
        return PreparedPatientSummary(
            message="EMR 데이터가 로드되었습니다.",
            sections=(),
            guidance="현재 기록을 함께 확인하세요.",
        ), updated_at

    def get(self, patient_id: str) -> tuple[dict[str, object], datetime]:
        if patient_id not in self.prepared:
            raise PatientContextNotReadyError()
        return self.prepared[patient_id]


class Service:
    def __init__(self) -> None:
        self.called = False
        self.question = ""

    def answer_prepared_snapshot(self, *, patient_id: str, question: str, snapshot: object) -> str:
        self.called = True
        self.question = question
        return "최근 혈당 추이를 확인하세요."


def emr_payload(patient_id: str = "pid-4356") -> dict[str, object]:
    return {"patient": {"patientId": patient_id}, "bloodPressureList": []}


class ClinicalChatApiTest(unittest.TestCase):
    def setUp(self) -> None:
        self.service = Service()
        self.client = TestClient(create_app(self.service, Store()))  # type: ignore[arg-type]
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
        self.assertTrue(any("body.question:value_error" in line for line in logs.output))
        self.assertTrue(any("request_id=diagnostic-request-1 status=400" in line for line in logs.output))
        extra = self.client.post("/internal/v1/clinical-chat", headers=self.headers, json={"patient_id": "pid-4356", "question": "질문", "language": "ko"})
        self.assertEqual(extra.status_code, 400)

    def test_prepare_requires_raw_payload_rejects_legacy_snapshot_and_patient_mismatch(self) -> None:
        missing = self.client.post("/internal/v1/patient-context", headers=self.headers, json={"patient_id": "pid-4356"})
        self.assertEqual(missing.status_code, 400)
        legacy = self.client.post("/internal/v1/patient-context", headers=self.headers, json={"patient_id": "pid-4356", "emr_snapshot": {}})
        self.assertEqual(legacy.status_code, 400)
        mismatch = self.client.post("/internal/v1/patient-context", headers=self.headers, json={"patient_id": "mismatch", "emr_payload": emr_payload("other")})
        self.assertEqual(mismatch.status_code, 403)
        self.assertEqual(mismatch.json()["code"], "PATIENT_MISMATCH")

    def test_emr_schema_failure_logs_the_missing_field_without_logging_payload(self) -> None:
        with self.assertLogs("uvicorn.error", level="WARNING") as logs:
            response = self.client.post(
                "/internal/v1/patient-context",
                headers={"X-Request-ID": "emr-schema-request-1"},
                json={"patient_id": "unavailable", "emr_payload": {}},
            )

        self.assertEqual(response.status_code, 502)
        self.assertTrue(
            any(
                "request_id=emr-schema-request-1" in line
                and "failure=emr_payload.patient:missing_or_not_object" in line
                for line in logs.output
            )
        )


if __name__ == "__main__":
    unittest.main()
