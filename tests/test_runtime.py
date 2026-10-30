from __future__ import annotations

from datetime import datetime
import os
import time
from unittest import TestCase
from unittest.mock import patch

from fastapi.testclient import TestClient

from ax_g_ai.adapters.main_bridge import PreparedRawEmrContext
from ax_g_ai.errors import PatientContextNotReadyError
from ax_g_ai.runtime import create_runtime_app
from ax_g_ai.services.patient_context_store import PatientContextStore


def context() -> PreparedRawEmrContext:
    return PreparedRawEmrContext({"blood_sugar": [{"value": 120}]}, datetime.fromisoformat("2026-09-22T10:30:00+09:00"))


class PatientContextStoreTest(TestCase):
    def test_context_is_available_until_discarded(self) -> None:
        store = PatientContextStore()
        store.put("patient-1", context())

        self.assertEqual(store.get("patient-1"), context())
        store.discard("patient-1")
        with self.assertRaises(PatientContextNotReadyError):
            store.get("patient-1")

    def test_expired_or_evicted_context_requires_a_new_prepare(self) -> None:
        expiring = PatientContextStore(ttl_seconds=0.01)
        expiring.put("patient-1", context())
        time.sleep(0.02)
        with self.assertRaises(PatientContextNotReadyError):
            expiring.get("patient-1")

        bounded = PatientContextStore(max_entries=1)
        bounded.put("patient-1", context())
        bounded.put("patient-2", context())
        with self.assertRaises(PatientContextNotReadyError):
            bounded.get("patient-1")
        self.assertEqual(bounded.get("patient-2"), context())


class RuntimeAssemblyTest(TestCase):
    def test_invalid_configuration_fails_closed_without_calling_a_provider(self) -> None:
        for environment in (
            {},
            {"OPENAI_API_KEY": "test-key", "AX_G_REFERENCE_DATA_DB_PATH": "/nonexistent/reference.sqlite"},
        ):
            with self.subTest(environment=environment), patch.dict(os.environ, environment, clear=True):
                with self.assertLogs("uvicorn.error", level="WARNING") as logs:
                    client = TestClient(create_runtime_app())
                response = client.post(
                    "/internal/v1/patient-context",
                    json={"patient_id": "patient-1", "emr_payload": {"patient": {"patientId": "patient-1"}}},
                )

            self.assertEqual(response.json()["code"], "AI_PROVIDER_UNAVAILABLE")
            self.assertTrue(any("runtime_configuration_invalid" in line for line in logs.output))
            self.assertFalse(any("test-key" in line for line in logs.output))
