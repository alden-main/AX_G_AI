from __future__ import annotations

from datetime import datetime
import json
from unittest.mock import patch
import unittest

from ax_g_ai.adapters.emr import EmrAdapterError, EmrReadRequest, UrllibEmrReadClient
from ax_g_ai.config import BridgeSnapshotSettings


class FakeResponse:
    def __init__(self, body: bytes) -> None:
        self.headers = {"Content-Type": "application/json; charset=utf-8"}
        self._body = body

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None


class UrllibEmrReadClientTest(unittest.TestCase):
    def test_post_uses_configured_address_and_only_request_contract_fields(self) -> None:
        settings = BridgeSnapshotSettings("https://bridge.example", "/contexts/{patient_id}/{encounter_id}", "POST", 3, "token")
        client = UrllibEmrReadClient(settings)
        request = EmrReadRequest("patient 1", "encounter-1", datetime.fromisoformat("2026-09-01T00:00:00+09:00"), datetime.fromisoformat("2026-09-02T00:00:00+09:00"), "Asia/Seoul", "trace-1")

        with patch("ax_g_ai.adapters.emr.urlopen", return_value=FakeResponse(b"{}")) as urlopen:
            client.fetch_patient_snapshot(request)

        sent = urlopen.call_args.args[0]
        self.assertEqual(sent.full_url, "https://bridge.example/contexts/patient%201/encounter-1")
        self.assertEqual(sent.get_header("Authorization"), "Bearer token")
        self.assertEqual(json.loads(sent.data), {
            "patient_id": "patient 1", "encounter_id": "encounter-1",
            "from": "2026-09-01T00:00:00+09:00", "to": "2026-09-02T00:00:00+09:00",
            "timezone": "Asia/Seoul",
        })

    def test_bridge_snapshot_requires_the_success_data_wrapper(self) -> None:
        settings = BridgeSnapshotSettings("https://bridge.example", "/internal/v1/patient-context/snapshot", "POST", 3, "token")
        client = UrllibEmrReadClient(settings)
        request = EmrReadRequest(
            "patient-1", "encounter-1", datetime.fromisoformat("2026-09-01T00:00:00+09:00"),
            datetime.fromisoformat("2026-09-02T00:00:00+09:00"), "Asia/Seoul", "trace-1", "ctx-1",
        )

        with patch("ax_g_ai.adapters.emr.urlopen", return_value=FakeResponse(b'{"context_id":"ctx-1"}')):
            with self.assertRaisesRegex(EmrAdapterError, "no usable patient data"):
                client.fetch_patient_snapshot(request)

    def test_simplified_snapshot_request_sends_patient_id_only(self) -> None:
        settings = BridgeSnapshotSettings("https://bridge.example", "/internal/v1/patient-context/snapshot", "POST", 3, "token")
        client = UrllibEmrReadClient(settings)
        request = EmrReadRequest("patient-1", "", None, None, "", "trace-1")  # type: ignore[arg-type]

        with patch("ax_g_ai.adapters.emr.urlopen", return_value=FakeResponse(b'{"success":true,"data":{}}')) as urlopen:
            client.fetch_patient_snapshot(request)

        sent = urlopen.call_args.args[0]
        self.assertEqual(json.loads(sent.data), {"patient_id": "patient-1"})


if __name__ == "__main__":
    unittest.main()
