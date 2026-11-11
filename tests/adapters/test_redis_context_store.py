from __future__ import annotations

import asyncio
import os
import time
import unittest
import uuid

from fastapi.testclient import TestClient

from ax_g_ai.adapters.main_bridge import RawEmrPayloadMapper
from ax_g_ai.api import create_app
from ax_g_ai.errors import PatientContextNotReadyError
from ax_g_ai.services.clinical_chat import ClinicalChatService
from ax_g_ai.services.clinical_evidence import InMemoryEvidenceIndex
from ax_g_ai.services.patient_context_store import context_from_json, context_to_json
from tests.services.test_clinical_recommendation import RULES, recommender, source
from tests.test_clinical_recommendation_api import ScriptedProvider, payload

TEST_REDIS_URL = os.getenv("AX_G_TEST_REDIS_URL")


class ContextCodecTest(unittest.TestCase):
    def test_round_trip_keeps_data_time_and_rule_source_but_not_derived_facts(self) -> None:
        context = RawEmrPayloadMapper().map("pid-1", {"data": source()})
        restored = context_from_json(context_to_json(context))
        self.assertEqual((restored.data, restored.updated_at, restored.rule_source), (context.data, context.updated_at, context.rule_source))
        self.assertIsNone(restored.rule_input)
        engine = recommender()
        self.assertEqual(engine.prepare(restored.rule_source), engine.prepare(source()))  # facts are re-derived exactly

    def test_other_format_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            context_from_json(b'{"v": 99}')


@unittest.skipUnless(TEST_REDIS_URL, "AX_G_TEST_REDIS_URL not set (needs a disposable Redis)")
class RedisContextStoreTest(unittest.TestCase):
    def setUp(self) -> None:
        from ax_g_ai.adapters.redis_context_store import RedisPatientContextStore

        self.prefix = f"axg:test:{uuid.uuid4().hex}:"
        self.make = lambda ttl=60.0: RedisPatientContextStore(  # noqa: E731
            RedisPatientContextStore.connect(TEST_REDIS_URL, ttl_seconds=ttl)._client, ttl_seconds=ttl, prefix=self.prefix,  # type: ignore[arg-type]
        )
        self.context = RawEmrPayloadMapper().map("pid-1", {"data": source()})

    def run_async(self, coroutine):
        return asyncio.run(coroutine)

    def test_put_get_discard_and_hashed_keys(self) -> None:
        async def scenario():
            store = self.make()
            await store.put("pid-1", self.context)
            restored = await store.get("pid-1")
            keys = [key async for key in store._client.scan_iter(match=self.prefix + "*")]
            await store.discard("pid-1")
            with self.assertRaises(PatientContextNotReadyError):
                await store.get("pid-1")
            await store.close()
            return restored, keys

        restored, keys = self.run_async(scenario())
        self.assertEqual((restored.data, restored.updated_at), (self.context.data, self.context.updated_at))
        self.assertEqual(len(keys), 1)
        self.assertNotIn(b"pid-1", keys[0])

    def test_contexts_expire_with_the_ttl(self) -> None:
        async def scenario():
            store = self.make(ttl=0.2)
            await store.put("pid-1", self.context)
            await asyncio.sleep(0.4)
            try:
                await store.get("pid-1")
            finally:
                await store.close()

        with self.assertRaises(PatientContextNotReadyError):
            self.run_async(scenario())

    def test_unreadable_document_asks_for_a_new_prepare(self) -> None:
        async def scenario():
            store = self.make()
            await store._client.set(store._key("pid-1"), b'{"v": 99}')
            try:
                await store.get("pid-1")
            finally:
                await store.close()

        with self.assertLogs("uvicorn.error", level="WARNING"), self.assertRaises(PatientContextNotReadyError):
            self.run_async(scenario())

    def test_two_app_instances_share_contexts_like_two_workers(self) -> None:
        def app():
            evidence = InMemoryEvidenceIndex.from_records(RULES["chunks.jsonl"], {}, include_unreviewed=True)
            return TestClient(create_app(ClinicalChatService(ScriptedProvider(), evidence=evidence), RawEmrPayloadMapper(),  # type: ignore[arg-type]
                                         self.make(), recommender()))

        with app() as first, app() as second:
            self.assertEqual(first.post("/internal/v1/patient-context", json=payload()).status_code, 200)
            chat = second.post("/internal/v1/clinical-chat", json={"patient_id": "pid-1", "question": "추천 검사 알려줘"})
            on_first = first.post("/internal/v1/clinical-recommendations", json={"patient_id": "pid-1"}).json()["assessment"]
            on_second = second.post("/internal/v1/clinical-recommendations", json={"patient_id": "pid-1"}).json()["assessment"]
        self.assertEqual(chat.status_code, 200)
        self.assertEqual(on_second, on_first)  # rule facts re-derived from the stored source give the same result


class RedisRuntimeTest(unittest.TestCase):
    def test_unreachable_redis_falls_back_to_memory_without_logging_the_password(self) -> None:
        from ax_g_ai.config import RuntimeSettings
        from ax_g_ai.runtime import _context_store
        from ax_g_ai.services.patient_context_store import PatientContextStore

        settings = RuntimeSettings(patient_context_redis_url="redis://:very-secret@127.0.0.1:1/0", patient_context_store_timeout_seconds=0.2)
        started = time.monotonic()
        with self.assertLogs("uvicorn.error", level="WARNING") as logs:
            store = _context_store(settings)
        self.assertIsInstance(store, PatientContextStore)
        self.assertLess(time.monotonic() - started, 3)
        self.assertTrue(any("fallback=memory" in line for line in logs.output))
        self.assertFalse(any("very-secret" in line for line in logs.output))


if __name__ == "__main__":
    unittest.main()
