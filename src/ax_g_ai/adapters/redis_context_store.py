"""Patient contexts in Redis, shared by every worker and instance (``AX_G_PATIENT_CONTEXT_REDIS_URL``).

Each context is one key holding ``context_to_json`` (de-identified data only), with the same TTL as
the in-memory store; Redis evicts by ``maxmemory`` instead of a fixed count. Keys are SHA-256 of
the patient id, so ids are not readable from a key listing. Calls use the asyncio client and
never block the event loop. A Redis failure is reported as ``EMR_UNAVAILABLE`` (never stale data).
"""

from __future__ import annotations

import hashlib
import logging

import redis
import redis.asyncio as aioredis

from ax_g_ai.errors import PatientContextNotReadyError, PatientContextUnavailableError
from ax_g_ai.services.patient_context_store import PreparedRawEmrContext, context_from_json, context_to_json

logger = logging.getLogger("uvicorn.error")
_PREFIX = "axg:patient-context:"


class RedisPatientContextStore:
    def __init__(self, client: aioredis.Redis, *, ttl_seconds: float, prefix: str = _PREFIX) -> None:
        self._client = client
        self._ttl_ms = max(1, int(ttl_seconds * 1000))
        self._prefix = prefix

    @classmethod
    def connect(cls, url: str, *, ttl_seconds: float, timeout_seconds: float = 1.0) -> "RedisPatientContextStore":
        """Check the server once (synchronously, at start-up) and return a store using a pooled async client.

        Raises ``redis.RedisError`` when the server is unreachable or rejects the credentials.
        """
        options = {"socket_timeout": timeout_seconds, "socket_connect_timeout": timeout_seconds}
        with redis.Redis.from_url(url, **options) as probe:
            probe.ping()
        return cls(aioredis.Redis.from_url(url, health_check_interval=30, **options), ttl_seconds=ttl_seconds)

    def _key(self, patient_id: str) -> str:
        return self._prefix + hashlib.sha256(patient_id.encode("utf-8")).hexdigest()

    async def put(self, patient_id: str, context: PreparedRawEmrContext) -> None:
        try:
            await self._client.set(self._key(patient_id), context_to_json(context), px=self._ttl_ms)
        except redis.RedisError as error:
            raise _unavailable(error) from error

    async def get(self, patient_id: str) -> PreparedRawEmrContext:
        try:
            raw = await self._client.get(self._key(patient_id))
        except redis.RedisError as error:
            raise _unavailable(error) from error
        if raw is None:
            raise PatientContextNotReadyError()
        try:
            return context_from_json(raw)
        except (ValueError, KeyError, TypeError) as error:
            # Another format (e.g. a newer release during a rolling update): ask for a fresh prepare.
            logger.warning("patient_context_unreadable error_type=%s", type(error).__name__)
            raise PatientContextNotReadyError() from error

    async def discard(self, patient_id: str) -> None:
        try:
            await self._client.delete(self._key(patient_id))
        except redis.RedisError as error:
            raise _unavailable(error) from error

    async def close(self) -> None:
        await self._client.aclose()


def _unavailable(error: Exception) -> PatientContextUnavailableError:
    # Error class only: the message may contain the server address.
    logger.warning("patient_context_store_failed error_type=%s", type(error).__name__)
    return PatientContextUnavailableError("patient_context_store:unavailable")
