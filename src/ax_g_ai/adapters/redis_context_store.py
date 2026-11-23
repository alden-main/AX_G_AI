"""모든 워커·인스턴스가 공유하는 Redis 환자 문맥 저장소(``AX_G_PATIENT_CONTEXT_REDIS_URL``).

설계상 위치: 어댑터 계층의 ``services.patient_context_store.ContextStore`` 구현. Redis URL이 설정되면 ``runtime``이
``RedisPatientContextStore.connect``로 만들고, API의 문맥 준비·채팅 요청이 이 저장소를 쓴다.

역할: 문맥 하나는 ``context_to_json``(비식별 데이터만)을 담은 키 하나이며, TTL은 메모리 저장소와 같다. Redis는 고정 개수
대신 ``maxmemory``로 축출한다. 키는 환자 ID의 SHA-256이라 키 목록에서 ID를 읽을 수 없다. 호출은 asyncio Client를 써서
이벤트 루프를 막지 않는다. Redis 장애는 ``EMR_UNAVAILABLE``로 보고한다(오래된 데이터를 돌려주지 않는다).
"""

from __future__ import annotations

import hashlib
import logging
import math

import redis
import redis.asyncio as aioredis
from redis.backoff import ConstantBackoff
from redis.retry import Retry

from ax_g_ai.errors import PatientContextNotReadyError, PatientContextUnavailableError
from ax_g_ai.services.patient_context_store import PreparedRawEmrContext, context_from_json, context_to_json

logger = logging.getLogger("uvicorn.error")
_PREFIX = "axg:patient-context:"
STARTUP_RETRY_INTERVAL_SECONDS = 2.0


class RedisPatientContextStore:
    """Redis에 환자 문맥을 TTL과 함께 저장하는 ``ContextStore`` 구현.

    역할: 환자 ID를 해시한 키로 문맥 JSON을 쓰고·읽고·지운다. Redis 오류는 ``PatientContextUnavailableError``로,
    없거나 읽을 수 없는 문맥은 ``PatientContextNotReadyError``로 바꾼다.
    """

    backend = "redis"

    def __init__(self, client: aioredis.Redis, *, ttl_seconds: float, prefix: str = _PREFIX) -> None:
        self._client = client
        self._ttl_ms = max(1, int(ttl_seconds * 1000))
        self._prefix = prefix

    async def ping(self) -> bool:
        """Client 소켓 시간 제한 안에서 PING 한 번. 실패는 오류 클래스 이름만 기록한다(URL에 비밀번호가 들어 있다)."""
        try:
            return bool(await self._client.ping())
        except (redis.RedisError, OSError) as error:
            logger.warning("patient_context_store_ping_failed error_type=%s", type(error).__name__)
            return False

    @classmethod
    def connect(
        cls, url: str, *, ttl_seconds: float, timeout_seconds: float = 1.0, wait_seconds: float = 0,
    ) -> RedisPatientContextStore:
        """서버를 (시작 시점에 동기로) 확인하고, 연결 풀을 쓰는 비동기 Client로 저장소를 만든다.

        역할: 동기 Client로 PING 해 서버 연결과 인증을 확인한다. ``wait_seconds``(API와 함께 시작하는 서버용) 동안은
        연결 오류를 ``STARTUP_RETRY_INTERVAL_SECONDS`` 간격으로 재시도한다. 확인 뒤 30초 간격 상태 확인을 켠 비동기
        Client를 만든다.
        입력:
            url: Redis URL(비밀번호를 담을 수 있다).
            ttl_seconds: 문맥 보관 기간(초).
            timeout_seconds: 소켓 연결·읽기 시간 제한(초).
            wait_seconds: 시작 시 연결 재시도에 쓸 대략적인 총 시간(초). 0이면 재시도하지 않는다.
        출력: ``RedisPatientContextStore``.
        예외:
            redis.RedisError: 서버에 끝내 닿지 않거나 인증을 거부할 때.
        """
        retries = math.ceil(wait_seconds / STARTUP_RETRY_INTERVAL_SECONDS)
        # OSError도 넣는다: redis-py는 자기 오류만 재시도하는데, 거부되거나 이름을 풀 수 없는 연결은 포기하기
        # 전까지 OSError다(아직 시작 중인 서버).
        probe_retry = Retry(
            ConstantBackoff(STARTUP_RETRY_INTERVAL_SECONDS), retries,
            supported_errors=(redis.ConnectionError, redis.TimeoutError, OSError),
        )
        with redis.Redis.from_url(url, socket_timeout=timeout_seconds, socket_connect_timeout=timeout_seconds,
                                  retry=probe_retry) as probe:
            probe.ping()
        client = aioredis.Redis.from_url(
            url, health_check_interval=30, socket_timeout=timeout_seconds, socket_connect_timeout=timeout_seconds,
        )
        return cls(client, ttl_seconds=ttl_seconds)

    def _key(self, patient_id: str) -> str:
        return self._prefix + hashlib.sha256(patient_id.encode("utf-8")).hexdigest()

    async def put(self, patient_id: str, context: PreparedRawEmrContext) -> None:
        try:
            await self._client.set(self._key(patient_id), context_to_json(context), px=self._ttl_ms)
        except redis.RedisError as error:
            raise _unavailable(error) from error

    async def get(self, patient_id: str) -> PreparedRawEmrContext:
        """환자 문맥을 읽어 되살린다.

        역할: 해시 키로 값을 읽고 ``context_from_json``으로 복원한다. ``rule_input``은 저장하지 않으므로 비어 있다.
        입력:
            patient_id: 환자 ID.
        출력: 저장된 ``PreparedRawEmrContext``.
        예외:
            PatientContextUnavailableError: Redis 오류.
            PatientContextNotReadyError: 키가 없거나(만료 포함) 저장 형식을 읽을 수 없을 때.
        """
        try:
            raw = await self._client.get(self._key(patient_id))
        except redis.RedisError as error:
            raise _unavailable(error) from error
        if raw is None:
            raise PatientContextNotReadyError()
        try:
            return context_from_json(raw)
        except (ValueError, KeyError, TypeError) as error:
            # 다른 형식(예: 롤링 업데이트 중의 새 릴리스): 문맥을 다시 준비하도록 요청한다.
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
    # 오류 클래스 이름만 남긴다: 메시지에 서버 주소가 들어 있을 수 있다.
    logger.warning("patient_context_store_failed error_type=%s", type(error).__name__)
    return PatientContextUnavailableError("patient_context_store:unavailable")
