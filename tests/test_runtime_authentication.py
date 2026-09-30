from __future__ import annotations

import os
from unittest import TestCase
from unittest.mock import patch

from fastapi.testclient import TestClient

from ax_g_ai.runtime import create_runtime_app


class InternalNetworkAuthenticationTest(TestCase):
    def test_runtime_allows_unauthenticated_internal_network_requests(self) -> None:
        environment = {
            "OPENAI_API_KEY": "test-key",
        }
        with patch.dict(os.environ, environment, clear=True):
            client = TestClient(create_runtime_app())
            response = client.post(
                "/internal/v1/patient-context",
                json={"patient_id": "patient-1", "emr_payload": {}},
            )

        self.assertEqual(response.status_code, 502)
        self.assertEqual(response.json()["code"], "EMR_UNAVAILABLE")
