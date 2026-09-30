from __future__ import annotations

from datetime import datetime
import unittest

from ax_g_ai.adapters.emr import EmrAdapter, EmrAdapterError, EmrReadRequest
from ax_g_ai.domain.patient_context import CodeMapping, PatientContextBuilder
from tests.domain.test_patient_context import payload


class StaticEmrClient:
    def __init__(self, response: dict) -> None:
        self.response = response

    def fetch_patient_snapshot(self, request: EmrReadRequest) -> dict:
        return self.response


class EmrAdapterTest(unittest.TestCase):
    def setUp(self) -> None:
        self.request = EmrReadRequest(
            patient_id="patient-1",
            encounter_id="encounter-1",
            from_at=datetime.fromisoformat("2026-09-01T00:00:00+09:00"),
            to_at=datetime.fromisoformat("2026-09-17T00:00:00+09:00"),
            timezone="Asia/Seoul",
            request_id="request-1",
        )
        self.builder = PatientContextBuilder(
            [CodeMapping("hospital-lab-v1", "GLU", "LOINC", "2345-7")]
        )

    def test_returns_only_the_context_matching_the_requested_patient_and_encounter(self) -> None:
        adapter = EmrAdapter(StaticEmrClient(payload()), self.builder)

        context = adapter.read_patient_context(self.request)

        self.assertEqual(context.patient_id, self.request.patient_id)
        self.assertEqual(context.encounter.encounter_id, self.request.encounter_id)

    def test_rejects_a_cross_patient_response_instead_of_returning_it(self) -> None:
        response = payload()
        response["patient"]["patient_id"] = "patient-2"
        adapter = EmrAdapter(StaticEmrClient(response), self.builder)

        with self.assertRaisesRegex(EmrAdapterError, "patient does not match"):
            adapter.read_patient_context(self.request)

    def test_rejects_an_invalid_provider_payload(self) -> None:
        response = payload()
        response["encounter"] = {}
        adapter = EmrAdapter(StaticEmrClient(response), self.builder)

        with self.assertRaisesRegex(EmrAdapterError, "invalid patient-data"):
            adapter.read_patient_context(self.request)

    def test_bridge_snapshot_requires_a_matching_context_id_and_valid_metadata(self) -> None:
        response = payload()
        response.update({
            "context_id": "ctx-1", "snapshot_id": "snapshot-1",
            "snapshot_at": "2026-09-17T10:00:00+09:00", "schema_version": "patient-context-v1",
            "assembly_status": "partial", "missing_data": ["lab_results"], "delayed_data": [],
        })
        request = EmrReadRequest(**{**self.request.__dict__, "context_id": "ctx-1"})

        snapshot = EmrAdapter(StaticEmrClient(response), self.builder).read_bridge_snapshot(request)

        self.assertEqual(snapshot.context_id, "ctx-1")
        self.assertEqual(snapshot.snapshot_id, "snapshot-1")
        self.assertEqual(snapshot.schema_version, "patient-context-v1")
        self.assertEqual(snapshot.assembly_status, "partial")
        self.assertEqual(snapshot.missing_data, ("lab_results",))

    def test_bridge_snapshot_rejects_a_late_response_for_another_context(self) -> None:
        response = payload()
        response.update({
            "context_id": "ctx-old", "snapshot_id": "snapshot-old",
            "snapshot_at": "2026-09-17T10:00:00+09:00", "schema_version": "patient-context-v1",
            "assembly_status": "complete", "missing_data": [], "delayed_data": [],
        })
        request = EmrReadRequest(**{**self.request.__dict__, "context_id": "ctx-current"})

        with self.assertRaisesRegex(EmrAdapterError, "context does not match"):
            EmrAdapter(StaticEmrClient(response), self.builder).read_bridge_snapshot(request)

    def test_bridge_snapshot_rejects_missing_or_wrong_contract_metadata(self) -> None:
        request = EmrReadRequest(**{**self.request.__dict__, "context_id": "ctx-1"})
        for key, value in (("snapshot_id", None), ("schema_version", "patient-context-v2"), ("delayed_data", None)):
            with self.subTest(key=key):
                response = payload()
                response.update({
                    "context_id": "ctx-1", "snapshot_id": "snapshot-1",
                    "snapshot_at": "2026-09-17T10:00:00+09:00", "schema_version": "patient-context-v1",
                    "assembly_status": "complete", "missing_data": [], "delayed_data": [],
                })
                if value is None:
                    response.pop(key)
                else:
                    response[key] = value
                with self.assertRaises(EmrAdapterError):
                    EmrAdapter(StaticEmrClient(response), self.builder).read_bridge_snapshot(request)

    def test_prepared_snapshot_validates_only_patient_centered_contract(self) -> None:
        response = payload()
        response.update({
            "snapshot_id": "snapshot-1", "snapshot_at": "2026-09-17T10:00:00+09:00",
            "schema_version": "patient-context-v1", "assembly_status": "complete",
        })
        request = EmrReadRequest("patient-1", "", None, None, "", "request-1")  # type: ignore[arg-type]

        snapshot = EmrAdapter(StaticEmrClient(response), self.builder).read_prepared_snapshot(request)

        self.assertEqual(snapshot.patient_id, "patient-1")
        self.assertEqual(snapshot.snapshot_at, datetime.fromisoformat("2026-09-17T10:00:00+09:00"))


if __name__ == "__main__":
    unittest.main()
