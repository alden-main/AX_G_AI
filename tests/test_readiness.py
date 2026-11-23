"""GET /ready: 503 when AI calls cannot be served (no AI settings, context store down); corpus state reported only."""

from __future__ import annotations

import asyncio
import contextlib
import unittest

import psycopg
from fastapi.testclient import TestClient

from ax_g_ai.api import create_app
from ax_g_ai.services.clinical_evidence import InMemoryEvidenceIndex
from ax_g_ai.services.corpus_snapshot import CorpusSnapshot, StaticCorpus
from ax_g_ai.services.patient_context_store import PatientContextStore
from tests.services.test_clinical_recommendation import RULES, recommender
from tests.test_api import Service


class DownStore(PatientContextStore):
    backend = "redis"

    async def ping(self) -> bool:
        return False


class DatabaseCorpus:
    """A database-backed source (like ``LiveCorpus``) whose database answers or not."""

    def __init__(self, snapshot: CorpusSnapshot, *, reachable: bool) -> None:
        self.current = snapshot
        self.reachable = reachable

    def database_reachable(self) -> bool | None:
        return self.reachable

    def close(self) -> None:
        return None


def evidence() -> InMemoryEvidenceIndex:
    return InMemoryEvidenceIndex.from_records(RULES["chunks.jsonl"], {}, include_unreviewed=True)


class ReadinessTest(unittest.TestCase):
    def test_app_without_ai_settings_is_not_ready(self) -> None:
        with TestClient(create_app()) as client:
            response = client.get("/ready")
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json(), {
            "status": "not_ready", "ai_provider": "not_configured",
            "patient_context_store": {"backend": "memory", "reachable": True},
            "corpus": {"backend": "none", "database_reachable": None, "release_id": None, "evidence_chunks": 0, "rules": 0,
                       "dur_dataset_id": None},
        })

    def test_configured_app_with_the_file_corpus_is_ready(self) -> None:
        engine = recommender()
        corpus = StaticCorpus(CorpusSnapshot(evidence(), engine))
        with TestClient(create_app(Service(), corpus=corpus)) as client:  # type: ignore[arg-type]
            response = client.get("/ready")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual((body["status"], body["ai_provider"]), ("ready", "configured"))
        self.assertEqual(body["corpus"]["backend"], "files")
        self.assertEqual(body["corpus"]["evidence_chunks"], len(evidence()))
        self.assertEqual(body["corpus"]["rules"], len(engine))

    def test_unreachable_context_store_is_not_ready(self) -> None:
        with TestClient(create_app(Service(), store=DownStore())) as client:  # type: ignore[arg-type]
            response = client.get("/ready")
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["patient_context_store"], {"backend": "redis", "reachable": False})

    def test_unreachable_corpus_database_is_reported_but_still_ready(self) -> None:
        # The snapshot in memory keeps answering; only semantic search and release refresh stop.
        corpus = DatabaseCorpus(CorpusSnapshot(evidence(), recommender(), release_id=7, dur_dataset_id=3), reachable=False)
        with TestClient(create_app(Service(), corpus=corpus)) as client:  # type: ignore[arg-type]
            response = client.get("/ready")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            {key: response.json()["corpus"][key] for key in ("backend", "database_reachable", "release_id", "dur_dataset_id")},
            {"backend": "postgres", "database_reachable": False, "release_id": 7, "dur_dataset_id": 3},
        )

    def test_repeated_probes_log_only_the_change_of_state(self) -> None:
        store = DownStore()
        with TestClient(create_app(Service(), store=store)) as client, self.assertLogs("uvicorn.error", level="INFO") as logs:  # type: ignore[arg-type]
            for _ in range(3):
                client.get("/ready")
            store.ping = PatientContextStore().ping  # type: ignore[method-assign]
            client.get("/ready")
        readiness = [line for line in logs.output if "readiness" in line]
        self.assertEqual(len(readiness), 2)
        self.assertIn("status=not_ready", readiness[0])
        self.assertIn("status=ready", readiness[1])
        self.assertFalse([line for line in logs.output if "api_request_failed" in line])


class StorePingTest(unittest.TestCase):
    def test_redis_ping_failure_is_false_without_the_password_in_the_logs(self) -> None:
        import redis.asyncio as aioredis

        from ax_g_ai.adapters.redis_context_store import RedisPatientContextStore

        async def scenario() -> bool:
            client = aioredis.Redis.from_url("redis://:secret-password@127.0.0.1:1/0", socket_connect_timeout=0.5, socket_timeout=0.5)
            store = RedisPatientContextStore(client, ttl_seconds=60)
            try:
                return await store.ping()
            finally:
                await store.close()

        with self.assertLogs("uvicorn.error", level="WARNING") as logs:
            reachable = asyncio.run(scenario())
        self.assertFalse(reachable)
        self.assertNotIn("secret-password", "\n".join(logs.output))


class CorpusDatabasePingTest(unittest.TestCase):
    def test_live_corpus_reports_an_unreachable_database(self) -> None:
        from ax_g_ai.corpus_store.live import LiveCorpus

        class DownPool:
            def connection(self):
                raise psycopg.OperationalError("connection refused")

        live = LiveCorpus.__new__(LiveCorpus)  # only the ping is under test: no refresh thread
        live._pool = DownPool()  # type: ignore[assignment]
        with self.assertLogs("uvicorn.error", level="WARNING"):
            self.assertFalse(live.database_reachable())

    def test_live_corpus_reports_a_reachable_database(self) -> None:
        from ax_g_ai.corpus_store.live import LiveCorpus

        class Connection:
            def execute(self, query: str) -> None:
                self.query = query

        class UpPool:
            def connection(self):
                return contextlib.nullcontext(Connection())

        live = LiveCorpus.__new__(LiveCorpus)
        live._pool = UpPool()  # type: ignore[assignment]
        self.assertTrue(live.database_reachable())


if __name__ == "__main__":
    unittest.main()
