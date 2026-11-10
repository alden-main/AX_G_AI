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
    lookup = None if database_path is None else SqliteReferenceLookup(database_path)
    evidence = _evidence_retriever(runtime_settings)
    return create_app(
        ClinicalChatService(build_provider_client(ai_settings), lookup, evidence, runtime_settings.evidence_max_chunks),
        RawEmrPayloadMapper(lookup),
        PatientContextStore(
            runtime_settings.patient_context_ttl_seconds,
            runtime_settings.patient_context_max_entries,
        ),
    )


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
    index = InMemoryEvidenceIndex.load(settings.evidence_corpus_dir, include_unreviewed=include_unreviewed)
    logger.info("clinical_evidence_loaded backend=files chunks=%d unreviewed=%d", len(index), index.unreviewed_count)
    return index
