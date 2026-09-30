from __future__ import annotations

from unittest import TestCase

from ax_g_ai.api import (
    PatientContextNotReadyError,
    PatientContextUnavailableError,
    PatientMismatchError,
)
from ax_g_ai.runtime import PatientContextStore
from ax_g_ai.services.clinical_chat import ClinicalChatProviderUnavailableError
from ax_g_ai.services.prepared_patient_summary import PreparedPatientSummary


def raw_payload(patient_id: str = "patient-1") -> dict[str, object]:
    return {"patient": {"patientId": patient_id}, "bloodSugarList": [{"measuredAt": "2026-09-22 10:30:00", "glucoseValue": 120}]}


class SummaryService:
    def __init__(self) -> None:
        self.fail = False
        self.received: object | None = None

    def summarize_prepared_snapshot(self, *, patient_id: str, snapshot: object) -> PreparedPatientSummary:
        self.received = snapshot
        if self.fail:
            raise ClinicalChatProviderUnavailableError()
        return PreparedPatientSummary("EMR 데이터가 로드되었습니다.", (), "현재 기록을 확인하세요.")


class PatientContextStoreTest(TestCase):
    def setUp(self) -> None:
        self.service = SummaryService()
        self.store = PatientContextStore(self.service)  # type: ignore[arg-type]

    def test_prepares_the_supplied_snapshot_without_external_read(self) -> None:
        summary, updated_at = self.store.prepare("patient-1", raw_payload())

        self.assertEqual(summary.message, "EMR 데이터가 로드되었습니다.")
        self.assertEqual(updated_at.isoformat(), "2026-09-22T10:30:00+09:00")
        self.assertIsNotNone(self.service.received)

    def test_failed_refresh_discards_previous_context(self) -> None:
        self.store.prepare("patient-1", raw_payload())
        self.service.fail = True

        with self.assertRaises(ClinicalChatProviderUnavailableError):
            self.store.prepare("patient-1", raw_payload())
        with self.assertRaises(PatientContextNotReadyError):
            self.store.get("patient-1")

    def test_invalid_or_cross_patient_snapshot_is_not_prepared(self) -> None:
        with self.assertRaises(PatientContextUnavailableError):
            self.store.prepare("patient-1", {"success": False})
        with self.assertRaises(PatientMismatchError):
            self.store.prepare("patient-1", raw_payload("patient-2"))
