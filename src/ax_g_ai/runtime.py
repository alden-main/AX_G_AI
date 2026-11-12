"""환경 설정으로 Bridge API 런타임을 조립한다.

말뭉치(근거 검색·규칙)와 DUR은 하나의 스냅샷(``CorpusSnapshot``)으로 묶는다. PostgreSQL을 쓸 수
있으면 근거와 규칙을 같은 활성 릴리스에서 읽고 ``LiveCorpus``가 릴리스·DUR 데이터셋 변경을 함께
반영한다. DB를 쓸 수 없으면 근거와 규칙을 모두 파일 말뭉치에서 읽어 기동 시점으로 고정한다.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

from fastapi import FastAPI

from ax_g_ai.adapters.main_bridge import RawEmrPayloadMapper
from ax_g_ai.adapters.provider import build_provider_client
from ax_g_ai.api import create_app
from ax_g_ai.config import AISettings, ConfigurationError, RuntimeSettings
from ax_g_ai.services.clinical_chat import ClinicalChatService
from ax_g_ai.services.clinical_evidence import EvidenceRetriever, InMemoryEvidenceIndex
from ax_g_ai.services.clinical_recommendation import ClinicalRecommender, RuleCorpus
from ax_g_ai.services.corpus_snapshot import CorpusSnapshot, CorpusSource, StaticCorpus
from ax_g_ai.services.emr_mapping import SiteCodeMap
from ax_g_ai.services.patient_context_store import ContextStore, PatientContextStore
from ax_g_ai.services.reference_lookup import SqliteReferenceLookup

logger = logging.getLogger("uvicorn.error")
_T = TypeVar("_T")
_STARTUP_RETRY_INTERVAL_SECONDS = 2.0


def create_runtime_app() -> FastAPI:
    """필수 설정이 없거나 잘못되면 AI를 호출하지 않는 fail-closed 앱을 생성한다."""
    try:
        ai_settings = AISettings.from_environment()
        runtime_settings = RuntimeSettings.from_environment()
    except ConfigurationError as error:
        # The message names fields and reasons only, never configured values.
        logger.warning("runtime_configuration_invalid reason=%s", error)
        return create_app()
    database_path = runtime_settings.reference_data_db_path
    if database_path is not None and runtime_settings.attributed_sources_only:
        # medicine_dict/disease_dict: provenance and license unconfirmed, so it cannot be attributed.
        logger.warning("reference_data_disabled reason=attributed_sources_only")
        database_path = None
    lookup = None if database_path is None else SqliteReferenceLookup(database_path)
    corpus = _corpus_source(runtime_settings)
    return create_app(
        ClinicalChatService(
            build_provider_client(ai_settings), lookup, None, runtime_settings.evidence_max_chunks,
            allow_general_knowledge=not runtime_settings.attributed_sources_only,
        ),
        RawEmrPayloadMapper(lookup),
        _context_store(runtime_settings),
        corpus=corpus,
        on_shutdown=(corpus.close,),
    )


def _context_store(settings: RuntimeSettings) -> ContextStore:
    """Redis when configured and reachable (shared by workers), else this process's memory."""
    workers = os.getenv("WEB_CONCURRENCY", "1")
    if settings.patient_context_redis_url is not None:
        import redis

        from ax_g_ai.adapters.redis_context_store import RedisPatientContextStore

        url = settings.patient_context_redis_url.get_secret_value()
        try:
            store = _wait_for(
                lambda: RedisPatientContextStore.connect(
                    url, ttl_seconds=settings.patient_context_ttl_seconds,
                    timeout_seconds=settings.patient_context_store_timeout_seconds,
                ),
                (redis.RedisError,), settings.dependency_startup_wait_seconds, "redis",
            )
        except redis.RedisError as error:
            # Error class only; the URL holds the password.
            logger.warning("patient_context_store_unavailable error_type=%s fallback=memory", type(error).__name__)
        else:
            logger.info("patient_context_store backend=redis workers=%s", workers)
            return store
    if workers.strip() not in ("", "1"):
        logger.warning("patient_context_store backend=memory workers=%s: contexts are not shared between workers "
                       "(set AX_G_PATIENT_CONTEXT_REDIS_URL)", workers)
    return PatientContextStore(settings.patient_context_ttl_seconds, settings.patient_context_max_entries)


def _corpus_source(settings: RuntimeSettings) -> CorpusSource:
    """Evidence and rules of the active PostgreSQL release (kept current), else of the corpus files (fixed)."""
    if settings.corpus_database_url is not None:
        live = _live_corpus(settings)
        if live is not None:
            return live
    elif settings.dur_enabled:
        logger.warning("dur_unavailable reason=no_database_url")
    evidence: EvidenceRetriever | None = None
    rules: RuleCorpus | None = None
    if settings.evidence_corpus_dir is not None:
        evidence = InMemoryEvidenceIndex.load(
            settings.evidence_corpus_dir, include_unreviewed=settings.evidence_include_unreviewed,
            require_license=settings.attributed_sources_only,
        )
        logger.info("clinical_evidence_loaded backend=files chunks=%d unreviewed=%d", len(evidence), evidence.unreviewed_count)
        if settings.clinical_rules_enabled:
            rules = RuleCorpus.load(
                settings.evidence_corpus_dir, include_unreviewed=settings.evidence_include_unreviewed,
                require_license=settings.attributed_sources_only,
            )
    recommender = _recommender(settings, rules, None, backend="files")
    return StaticCorpus(CorpusSnapshot(evidence, recommender))


def _live_corpus(settings: RuntimeSettings) -> CorpusSource | None:
    # Imported lazily so deployments without the database never load the driver.
    import psycopg

    from ax_g_ai.corpus_store.embedding import TeiEmbeddingClient
    from ax_g_ai.corpus_store.live import LiveCorpus
    from ax_g_ai.corpus_store.retriever import CorpusUnavailableError, PostgresEvidenceRetriever, load_index, open_pool
    from ax_g_ai.corpus_store.rules import read_rows
    from ax_g_ai.dur_store.reader import read_tables
    from ax_g_ai.services.dur_check import DurChecker, DurUnavailableError

    embedder = None if settings.embedding_url is None else TeiEmbeddingClient(
        settings.embedding_url, timeout_seconds=settings.embedding_timeout_seconds,
    )
    include_unreviewed, require_license = settings.evidence_include_unreviewed, settings.attributed_sources_only

    def build(conn: psycopg.Connection, release_id: int, dur_dataset_id: int | None) -> CorpusSnapshot:
        index = load_index(conn, release_id, include_unreviewed=include_unreviewed, require_license=require_license)
        evidence = PostgresEvidenceRetriever(
            pool, release_id, index, embedder=embedder, model_id=settings.embedding_model,
            include_unreviewed=include_unreviewed, similarity_floor=settings.evidence_similarity_floor,
        )
        logger.info(
            "clinical_evidence_loaded backend=postgres release_id=%d chunks=%d unreviewed=%d semantic=%s",
            release_id, len(evidence), evidence.unreviewed_count, embedder is not None,
        )
        rules = None
        if settings.clinical_rules_enabled:
            rules = RuleCorpus.from_rows(read_rows(conn, release_id), include_unreviewed=include_unreviewed, require_license=require_license)
        dur = None
        if settings.dur_enabled:
            if dur_dataset_id is None:
                logger.warning("dur_unavailable reason=no active dur dataset")
            else:
                try:
                    dur = DurChecker(read_tables(conn, dur_dataset_id))
                except DurUnavailableError as error:
                    logger.warning("dur_unavailable reason=%s", error)
                else:
                    logger.info("dur_loaded backend=postgres dataset_id=%s products=%d source=%s",
                                dur.dataset_id, len(dur), dur.source.get("id"))
        recommender = _recommender(settings, rules, dur, backend=f"postgres release_id={release_id}")
        return CorpusSnapshot(evidence, recommender, release_id, dur_dataset_id if dur is not None else None)

    closers: list[Callable[[], None]] = [embedder.close] if embedder is not None else []
    url = settings.corpus_database_url.get_secret_value()  # type: ignore[union-attr]
    pool = None  # read by ``build``

    def open_live() -> LiveCorpus:
        nonlocal pool
        pool = open_pool(url)
        try:
            return LiveCorpus(pool, build, refresh_seconds=settings.corpus_refresh_seconds, watch_dur=settings.dur_enabled,
                              on_close=closers)
        except BaseException:
            pool.close()
            raise

    try:
        return _wait_for(open_live, (psycopg.Error,), settings.dependency_startup_wait_seconds, "postgres")
    except (psycopg.Error, CorpusUnavailableError) as error:
        _close_all(closers)
        logger.warning("clinical_corpus_database_unavailable error_type=%s fallback=files", type(error).__name__)
        return None


def _wait_for(connect: Callable[[], _T], errors: tuple[type[Exception], ...], wait_seconds: float, name: str) -> _T:
    """``connect()`` retried every few seconds for up to ``wait_seconds`` (a dependency starting with the API); re-raises the last error."""
    deadline = time.monotonic() + wait_seconds
    attempt = 0
    while True:
        attempt += 1
        try:
            return connect()
        except errors as error:
            if time.monotonic() + _STARTUP_RETRY_INTERVAL_SECONDS > deadline:
                raise
            # Error class only: messages may carry the connection URL.
            logger.warning("startup_dependency_waiting dependency=%s attempt=%d error_type=%s", name, attempt, type(error).__name__)
            time.sleep(_STARTUP_RETRY_INTERVAL_SECONDS)


def _recommender(settings: RuntimeSettings, rules: RuleCorpus | None, dur, *, backend: str) -> ClinicalRecommender | None:
    if rules is None:
        return None
    site_map = None
    if settings.site_code_map_path is not None:
        # A malformed site map is a deployment error: fail at start-up (a later release keeps the current snapshot).
        site_map = SiteCodeMap.load(Path(settings.site_code_map_path), rules.catalog)
    recommender = ClinicalRecommender(
        rules, site_map, max_evidence=settings.recommendation_max_evidence, dur=dur, doses=settings.clinical_doses_enabled,
    )
    logger.info(
        "clinical_rules_loaded backend=%s rules=%d unreviewed=%d skipped=%d site_map=%s doses=%s dose_rules=%d dur=%s",
        backend, len(recommender), recommender.unreviewed_count, rules.skipped, site_map is not None,
        settings.clinical_doses_enabled, recommender.dose_rule_count, dur is not None,
    )
    return recommender


def _close_all(closers: list[Callable[[], None]]) -> None:
    for close in closers:
        close()
