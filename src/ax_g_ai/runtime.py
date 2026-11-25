"""환경 설정으로 Bridge API 런타임을 조립한다.

설계상 위치: 조립(composition root) 계층. ``ax-g-ai`` 명령과 Uvicorn 팩토리(``ax_g_ai.runtime:create_runtime_app``)가
호출하며, 설정·어댑터·서비스를 묶어 ``api.create_app``에 넘긴다.

역할: 말뭉치(근거 검색·규칙)와 DUR은 하나의 스냅샷(``CorpusSnapshot``)으로 묶는다. PostgreSQL을 쓸 수
있으면 근거와 규칙을 같은 활성 릴리스에서 읽고 ``LiveCorpus``가 릴리스·DUR 데이터셋 변경을 함께
반영한다. DB를 쓸 수 없으면 근거와 규칙을 모두 파일 말뭉치에서 읽어 기동 시점으로 고정한다.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from pathlib import Path

from fastapi import FastAPI

from ax_g_ai.adapters.main_bridge import RawEmrPayloadMapper
from ax_g_ai.adapters.provider import build_provider_client
from ax_g_ai.adapters.sqlite_reference_lookup import SqliteReferenceLookup
from ax_g_ai.adapters.telemetry import instrumentation_from_environment
from ax_g_ai.api import create_app
from ax_g_ai.config import AISettings, ConfigurationError, RuntimeSettings
from ax_g_ai.services.clinical_chat import ClinicalChatService
from ax_g_ai.services.clinical_evidence import EvidenceRetriever, InMemoryEvidenceIndex
from ax_g_ai.services.clinical_recommendation import ClinicalRecommender, RuleCorpus
from ax_g_ai.services.clinical_recommendation import read_rows as read_corpus_rows
from ax_g_ai.services.corpus_snapshot import CorpusSnapshot, CorpusSource, StaticCorpus
from ax_g_ai.services.emr_mapping import SiteCodeMap
from ax_g_ai.services.patient_context_store import ContextStore, PatientContextStore

logger = logging.getLogger("uvicorn.error")


def create_runtime_app() -> FastAPI:
    """환경 설정으로 운영용 FastAPI 앱을 조립한다.

    역할: AI·런타임 설정을 읽고 참조 DB 조회기, 말뭉치 원천, 추적 계측, 환자 문맥 저장소를 만들어
    ``create_app``에 넘긴다. ``attributed_sources_only``이면 참조 DB를 쓰지 않는다. 필수 설정이 없거나 잘못되면
    AI를 호출하지 않는 fail-closed 앱(서비스 없음)을 만든다.
    입력: 없음(환경 변수를 읽는다).
    출력: FastAPI 앱. 종료 시 말뭉치 원천을 닫도록 등록되어 있다.
    """
    try:
        ai_settings = AISettings.from_environment()
        runtime_settings = RuntimeSettings.from_environment()
    except ConfigurationError as error:
        # 메시지에는 필드 이름과 사유만 있고 설정값은 없다.
        logger.warning("runtime_configuration_invalid reason=%s", error)
        return create_app()
    database_path = runtime_settings.reference_data_db_path
    if database_path is not None and runtime_settings.attributed_sources_only:
        # medicine_dict/disease_dict: 출처·라이선스가 확인되지 않아 출처를 밝힐 수 없다.
        logger.warning("reference_data_disabled reason=attributed_sources_only")
        database_path = None
    lookup = None if database_path is None else SqliteReferenceLookup(database_path)
    corpus = _corpus_source(runtime_settings)
    instrumentation = instrumentation_from_environment(include_content=ai_settings.trace_content)
    logger.info("ai_tracing enabled=%s content=%s", instrumentation is not None, instrumentation is not None and ai_settings.trace_content)
    return create_app(
        ClinicalChatService(
            build_provider_client(ai_settings, instrumentation=instrumentation), lookup, evidence_limit=runtime_settings.evidence_max_chunks,
            allow_general_knowledge=not runtime_settings.attributed_sources_only, prompt_labels=runtime_settings.prompt_data_labels,
            model_profile=runtime_settings.model_profile,
        ),
        RawEmrPayloadMapper(lookup),
        _context_store(runtime_settings),
        corpus=corpus,
        on_shutdown=(corpus.close,),
    )


def _context_store(settings: RuntimeSettings) -> ContextStore:
    """환자 문맥 저장소를 고른다.

    역할: Redis URL이 설정되어 있고 연결되면 워커들이 공유하는 Redis 저장소를, 아니면 이 프로세스의 메모리
    저장소를 쓴다. 워커가 여럿인데 메모리 저장소를 쓰게 되면 경고를 남긴다.
    입력:
        settings: 런타임 설정.
    출력: ``RedisPatientContextStore`` 또는 ``PatientContextStore``.
    """
    workers = os.getenv("WEB_CONCURRENCY", "1")
    if settings.patient_context_redis_url is not None:
        import redis

        from ax_g_ai.adapters.redis_context_store import RedisPatientContextStore

        url = settings.patient_context_redis_url.get_secret_value()
        try:
            store = RedisPatientContextStore.connect(
                url, ttl_seconds=settings.patient_context_ttl_seconds,
                timeout_seconds=settings.patient_context_store_timeout_seconds,
                wait_seconds=settings.dependency_startup_wait_seconds,
            )
        except redis.RedisError as error:
            # 오류 클래스만 남긴다. URL에는 비밀번호가 들어 있다.
            logger.warning("patient_context_store_unavailable error_type=%s fallback=memory", type(error).__name__)
        else:
            logger.info("patient_context_store backend=redis workers=%s", workers)
            return store
    if workers.strip() not in ("", "1"):
        logger.warning("patient_context_store backend=memory workers=%s: contexts are not shared between workers "
                       "(set AX_G_PATIENT_CONTEXT_REDIS_URL)", workers)
    return PatientContextStore(settings.patient_context_ttl_seconds, settings.patient_context_max_entries)


def _corpus_source(settings: RuntimeSettings) -> CorpusSource:
    """근거·규칙(·DUR)을 제공할 말뭉치 원천을 고른다.

    역할: DB URL이 있고 연결되면 활성 PostgreSQL 릴리스를 계속 반영하는 ``LiveCorpus``를, 아니면 파일 말뭉치를
    한 번 읽어 고정한 ``StaticCorpus``를 쓴다(DUR 없음). 파일 말뭉치 경로도 없으면 근거·규칙이 없는 스냅샷이다.
    입력:
        settings: 런타임 설정.
    출력: ``CorpusSource``.
    """
    if settings.corpus_database_url is not None:
        live = _live_corpus(settings)
        if live is not None:
            return live
    elif settings.dur_enabled:
        logger.warning("dur_unavailable reason=no_database_url")
    evidence: EvidenceRetriever | None = None
    rules: RuleCorpus | None = None
    if settings.evidence_corpus_dir is not None:
        # 말뭉치 파일을 한 번 읽어 근거 색인과 규칙에 함께 쓴다.
        rows = read_corpus_rows(settings.evidence_corpus_dir)
        gate = {"include_unreviewed": settings.evidence_include_unreviewed, "require_license": settings.attributed_sources_only}
        evidence = InMemoryEvidenceIndex.from_rows(rows, **gate)
        logger.info("clinical_evidence_loaded backend=files chunks=%d unreviewed=%d", len(evidence), evidence.unreviewed_count)
        if settings.clinical_rules_enabled:
            rules = RuleCorpus.from_rows(rows, **gate)
    recommender = _recommender(settings, rules, None, backend="files")
    return StaticCorpus(CorpusSnapshot(evidence, recommender))


def _live_corpus(settings: RuntimeSettings) -> CorpusSource | None:
    """PostgreSQL 활성 릴리스를 반영하는 ``LiveCorpus``를 만든다.

    역할: 연결 풀을 열고, 릴리스·DUR 데이터셋 ID로 근거 검색기(어휘 색인 + 선택적 임베딩)·규칙·DUR 점검기를 묶은
    ``CorpusSnapshot``을 만드는 ``build``를 ``LiveCorpus``에 넘긴다. DB를 쓸 수 없으면 열었던 자원을 닫고 ``None``을 돌려준다.
    입력:
        settings: 런타임 설정(``corpus_database_url``이 있어야 한다).
    출력: ``LiveCorpus`` 또는 ``None``(DB 연결 실패 → 호출자가 파일 말뭉치로 대체).
    """
    # DB를 쓰지 않는 배포에서 드라이버를 불러오지 않도록 지연 import한다.
    import psycopg

    from ax_g_ai.corpus_store.embedding import TeiEmbeddingClient
    from ax_g_ai.corpus_store.live import LiveCorpus
    from ax_g_ai.corpus_store.retriever import CorpusUnavailableError, PostgresEvidenceRetriever, open_pool
    from ax_g_ai.corpus_store.rules import read_rows
    from ax_g_ai.dur_store.reader import read_tables
    from ax_g_ai.services.dur_check import DurChecker, DurUnavailableError

    embedder = None if settings.embedding_url is None else TeiEmbeddingClient(
        settings.embedding_url, timeout_seconds=settings.embedding_timeout_seconds,
    )
    include_unreviewed, require_license = settings.evidence_include_unreviewed, settings.attributed_sources_only

    def build(conn: psycopg.Connection, release_id: int, dur_dataset_id: int | None) -> CorpusSnapshot:
        # 릴리스를 한 번 읽어 근거 색인과 규칙에 함께 쓴다(문단과 개념을 한 번만 읽는다).
        rows = read_rows(conn, release_id)
        index = InMemoryEvidenceIndex.from_rows(rows, include_unreviewed=include_unreviewed, require_license=require_license)
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
            rules = RuleCorpus.from_rows(rows, include_unreviewed=include_unreviewed, require_license=require_license)
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

    try:
        # 풀이 ``dependency_startup_wait_seconds``까지 스스로 다시 연결을 시도한다(API와 함께 시작하는 DB 대비).
        pool = open_pool(url, wait_seconds=settings.dependency_startup_wait_seconds)
        try:
            return LiveCorpus(pool, build, refresh_seconds=settings.corpus_refresh_seconds, watch_dur=settings.dur_enabled,
                              on_close=closers)
        except BaseException:
            pool.close()
            raise
    except (psycopg.Error, CorpusUnavailableError) as error:
        _close_all(closers)
        logger.warning("clinical_corpus_database_unavailable error_type=%s fallback=files", type(error).__name__)
        return None


def _recommender(settings: RuntimeSettings, rules: RuleCorpus | None, dur, *, backend: str) -> ClinicalRecommender | None:
    """규칙 말뭉치로 추천기를 만든다.

    역할: 병원 코드 매핑(설정 시)·DUR 점검기·용량 규칙 사용 여부를 붙여 ``ClinicalRecommender``를 만들고 적재 현황을 로그에 남긴다.
    입력:
        settings: 런타임 설정.
        rules: 규칙 말뭉치. ``None``이면 추천기를 만들지 않는다.
        dur: DUR 점검기(``DurChecker``) 또는 ``None``.
        backend: 로그에 남길 원천 설명(예: ``files``, ``postgres release_id=3``).
    출력: ``ClinicalRecommender`` 또는 ``None``.
    예외: 병원 코드 매핑 파일이 잘못되면 ``SiteCodeMap.load``의 예외가 그대로 전파된다.
    """
    if rules is None:
        return None
    site_map = None
    if settings.site_code_map_path is not None:
        # 잘못된 병원 코드 매핑은 배포 오류다. 기동 시 실패시킨다(이후 릴리스에서 실패하면 현재 스냅샷을 유지한다).
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
    """등록된 정리 함수를 차례로 호출한다."""
    for close in closers:
        close()
