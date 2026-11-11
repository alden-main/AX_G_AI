"""환경 설정으로 Bridge API 런타임을 조립한다."""

from __future__ import annotations

import logging

from fastapi import FastAPI

from ax_g_ai.adapters.main_bridge import RawEmrPayloadMapper
from ax_g_ai.adapters.provider import build_provider_client
from ax_g_ai.api import create_app
from ax_g_ai.config import AISettings, ConfigurationError, RuntimeSettings
from ax_g_ai.services.clinical_chat import ClinicalChatService
from ax_g_ai.services.clinical_evidence import EvidenceRetriever, InMemoryEvidenceIndex
from ax_g_ai.services.clinical_recommendation import ClinicalRecommender, RuleCorpus
from ax_g_ai.services.emr_mapping import SiteCodeMap
from ax_g_ai.services.patient_context_store import PatientContextStore
from ax_g_ai.services.reference_lookup import SqliteReferenceLookup

logger = logging.getLogger("uvicorn.error")


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
    evidence = _evidence_retriever(runtime_settings)
    return create_app(
        ClinicalChatService(
            build_provider_client(ai_settings), lookup, evidence, runtime_settings.evidence_max_chunks,
            allow_general_knowledge=not runtime_settings.attributed_sources_only,
        ),
        RawEmrPayloadMapper(lookup),
        PatientContextStore(
            runtime_settings.patient_context_ttl_seconds,
            runtime_settings.patient_context_max_entries,
        ),
        _clinical_recommender(runtime_settings),
    )


def _clinical_recommender(settings: RuntimeSettings) -> ClinicalRecommender | None:
    """Stage-2 rules from the active database release, else the corpus files, else none."""
    if not settings.clinical_rules_enabled:
        return None
    include_unreviewed = settings.evidence_include_unreviewed
    corpus: RuleCorpus | None = None
    backend = "files"
    if settings.corpus_database_url is not None:
        import psycopg

        from ax_g_ai.corpus_store.retriever import CorpusUnavailableError
        from ax_g_ai.corpus_store.rules import read_rows_from_database

        try:
            release_id, rows = read_rows_from_database(settings.corpus_database_url.get_secret_value())
        except (psycopg.Error, CorpusUnavailableError) as error:
            logger.warning("clinical_rules_database_unavailable error_type=%s fallback=files", type(error).__name__)
        else:
            corpus = RuleCorpus.from_rows(rows, include_unreviewed=include_unreviewed, require_license=settings.attributed_sources_only)
            backend = f"postgres release_id={release_id}"
    if corpus is None:
        if settings.evidence_corpus_dir is None:
            return None
        corpus = RuleCorpus.load(settings.evidence_corpus_dir, include_unreviewed=include_unreviewed, require_license=settings.attributed_sources_only)
    site_map = None
    if settings.site_code_map_path is not None:
        # A malformed site map is a deployment error: fail at start-up, not per patient.
        site_map = SiteCodeMap.load(settings.site_code_map_path, corpus.catalog)
    recommender = ClinicalRecommender(
        corpus, site_map, max_evidence=settings.recommendation_max_evidence, dur=_dur_checker(settings),
        doses=settings.clinical_doses_enabled,
    )
    logger.info(
        "clinical_rules_loaded backend=%s rules=%d unreviewed=%d skipped=%d site_map=%s doses=%s dose_rules=%d",
        backend, len(recommender), recommender.unreviewed_count, corpus.skipped, site_map is not None,
        settings.clinical_doses_enabled, recommender.dose_rule_count,
    )
    return recommender


def _dur_checker(settings: RuntimeSettings):
    """HIRA DUR lists from the active ``dur`` dataset in PostgreSQL, else none (logged)."""
    if not settings.dur_enabled:
        return None
    if settings.corpus_database_url is None:
        logger.warning("dur_unavailable reason=no_database_url")
        return None
    import psycopg

    from ax_g_ai.dur_store.reader import read_active_tables
    from ax_g_ai.services.dur_check import DurChecker, DurUnavailableError

    try:
        dur = DurChecker(read_active_tables(settings.corpus_database_url.get_secret_value()))
    except (psycopg.Error, DurUnavailableError) as error:
        # Error class (and our own reason text) only; the database URL is never logged.
        reason = error if isinstance(error, DurUnavailableError) else type(error).__name__
        logger.warning("dur_unavailable reason=%s", reason)
        return None
    logger.info("dur_loaded backend=postgres dataset_id=%s products=%d source=%s", dur.dataset_id, len(dur), dur.source.get("id"))
    return dur


def _evidence_retriever(settings: RuntimeSettings) -> EvidenceRetriever | None:
    """PostgreSQL hybrid retrieval when configured, else the file index, else none."""
    include_unreviewed = settings.evidence_include_unreviewed
    if settings.corpus_database_url is not None:
        # Imported lazily so deployments without the database never load the driver.
        import psycopg

        from ax_g_ai.corpus_store.embedding import TeiEmbeddingClient
        from ax_g_ai.corpus_store.retriever import CorpusUnavailableError, PostgresEvidenceRetriever

        embedder = None if settings.embedding_url is None else TeiEmbeddingClient(
            settings.embedding_url, timeout_seconds=settings.embedding_timeout_seconds
        )
        try:
            retriever = PostgresEvidenceRetriever(
                settings.corpus_database_url.get_secret_value(), embedder=embedder, model_id=settings.embedding_model,
                include_unreviewed=include_unreviewed, similarity_floor=settings.evidence_similarity_floor,
                require_license=settings.attributed_sources_only,
            )
        except (psycopg.Error, CorpusUnavailableError) as error:
            logger.warning("clinical_evidence_database_unavailable error_type=%s fallback=files", type(error).__name__)
        else:
            logger.info(
                "clinical_evidence_loaded backend=postgres release_id=%d chunks=%d unreviewed=%d semantic=%s",
                retriever.release_id, len(retriever), retriever.unreviewed_count, embedder is not None,
            )
            return retriever
    if settings.evidence_corpus_dir is None:
        return None
    index = InMemoryEvidenceIndex.load(
        settings.evidence_corpus_dir, include_unreviewed=include_unreviewed, require_license=settings.attributed_sources_only,
    )
    logger.info("clinical_evidence_loaded backend=files chunks=%d unreviewed=%d", len(index), index.unreviewed_count)
    return index
