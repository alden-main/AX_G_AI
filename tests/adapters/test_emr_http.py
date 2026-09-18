from __future__ import annotations

from datetime import datetime
import json
from unittest.mock import patch
import unittest

from ax_g_ai.adapters.emr import EmrReadRequest, UrllibEmrReadClient
from ax_g_ai.config import EmrApiSettings


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
        settings = EmrApiSettings("https://emr.example", "/contexts/{patient_id}/{encounter_id}", "POST", 3, "token")
        client = UrllibEmrReadClient(settings)
        request = EmrReadRequest("patient 1", "encounter-1", datetime.fromisoformat("2026-09-01T00:00:00+09:00"), datetime.fromisoformat("2026-09-02T00:00:00+09:00"), "Asia/Seoul", "trace-1")

        with patch("ax_g_ai.adapters.emr.urlopen", return_value=FakeResponse(b"{}")) as urlopen:
            client.fetch_patient_snapshot(request)

        sent = urlopen.call_args.args[0]
        self.assertEqual(sent.full_url, "https://emr.example/contexts/patient%201/encounter-1")
        self.assertEqual(sent.get_header("Authorization"), "Bearer token")
        self.assertEqual(json.loads(sent.data), {
            "patient_id": "patient 1", "encounter_id": "encounter-1",
            "from": "2026-09-01T00:00:00+09:00", "to": "2026-09-02T00:00:00+09:00",
            "timezone": "Asia/Seoul",
        })


if __name__ == "__main__":
    unittest.main()
