"""Hybrid evidence retrieval from PostgreSQL: lexical (in-process BM25) + pgvector.

The active release's citable passages (``corpus.citable_evidence``, the single gate) are
loaded into an ``InMemoryEvidenceIndex`` for lexical and concept scoring, so lexical
behaviour matches the file-based index. Semantic candidates come from pgvector. A
passage is cited when its lexical score passes the lexical cutoff **or** its cosine
similarity passes ``similarity_floor``; accepted passages are ordered by reciprocal
rank fusion. If the embedding server or database fails at query time, the last loaded
lexical index still answers.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict
from typing import Any

import psycopg
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from ax_g_ai.corpus_store.embedding import EmbeddingError, TeiEmbeddingClient, vector_literal
from ax_g_ai.services.clinical_evidence import EvidenceChunk, InMemoryEvidenceIndex, concept_names, lexical_cutoff, prioritize

logger = logging.getLogger("uvicorn.error")
_RRF_K = 60


class CorpusUnavailableError(RuntimeError):
    """No active corpus release is available in the database."""


class PostgresEvidenceRetriever:
    def __init__(
        self, database_url: str, *, embedder: TeiEmbeddingClient | None, model_id: str, include_unreviewed: bool,
        similarity_floor: float = 0.6, candidates: int = 30, refresh_seconds: float = 60.0, require_license: bool = False,
        pool_size: int = 10,
    ) -> None:
        self._embedder = embedder
        self._model_id = model_id
        self._include_unreviewed = include_unreviewed
        self._require_license = require_license
        self._similarity_floor = similarity_floor
        self._candidates = candidates
        self._refresh_seconds = refresh_seconds
        self._lock = threading.Lock()
        # Pooled connections: a question no longer pays TCP + SCRAM authentication on every search.
        self._pool = ConnectionPool(
            database_url, min_size=1, max_size=pool_size, timeout=3, open=False,
            kwargs={"autocommit": True, "connect_timeout": 3, "row_factory": dict_row},
        )
        try:
            self._pool.open(wait=True, timeout=3)
            self._release_id, self._index = self._load()
        except BaseException:
            self._pool.close()  # stop the pool's reconnect workers before falling back to files
            raise
        self._checked_at = time.monotonic()

    def __len__(self) -> int:
        return len(self._index)

    @property
    def release_id(self) -> int:
        return self._release_id

    @property
    def unreviewed_count(self) -> int:
        return self._index.unreviewed_count

    def close(self) -> None:
        self._pool.close()
        if self._embedder is not None:
            self._embedder.close()

    def _load(self) -> tuple[int, InMemoryEvidenceIndex]:
        with self._pool.connection() as conn:
            row = conn.execute("SELECT release_id FROM corpus.corpus_release WHERE is_active").fetchone()
            if row is None:
                raise CorpusUnavailableError("no active corpus release")
            release_id = row["release_id"]
            chunks = conn.execute(
                "SELECT chunk_id, claim_id, chapter, title_path, body AS text, claim_status, ai_tier, verification, "
                "review_status, sources, conflicts, updated_at::text AS updated_at, doc_path "
                "FROM corpus.citable_evidence WHERE release_id = %s", (release_id,),
            ).fetchall()
            concepts = conn.execute(
                "SELECT c.concept_type, c.payload, coalesce(json_agg(json_build_object('claim_id', u.claim_id)) "
                "  FILTER (WHERE u.claim_id IS NOT NULL), '[]') AS used_by "
                "FROM corpus.concept c LEFT JOIN corpus.concept_usage u USING (release_id, concept_type, concept_id) "
                "WHERE c.release_id = %s GROUP BY c.concept_type, c.concept_id, c.payload", (release_id,),
            ).fetchall()
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in concepts:
            kind = "test" if row["concept_type"] == "test" else "class" if row["concept_type"] == "drug_class" else "drug"
            grouped[kind].append({**row["payload"], "used_by": row["used_by"]})
        names = concept_names(grouped["test"], grouped["drug"], grouped["class"])
        # is_citable() runs again in Python as a second gate behind the SQL view.
        index = InMemoryEvidenceIndex.from_records(
            chunks, names, include_unreviewed=self._include_unreviewed, require_license=self._require_license,
        )
        return release_id, index

    def _maybe_refresh(self) -> None:
        if time.monotonic() - self._checked_at < self._refresh_seconds:
            return
        with self._lock:
            if time.monotonic() - self._checked_at < self._refresh_seconds:
                return
            self._checked_at = time.monotonic()
            try:
                with self._pool.connection() as conn:
                    row = conn.execute("SELECT release_id FROM corpus.corpus_release WHERE is_active").fetchone()
                if row is not None and row["release_id"] != self._release_id:
                    self._release_id, self._index = self._load()
                    logger.info("clinical_evidence_release_loaded release_id=%d chunks=%d", self._release_id, len(self._index))
            except (psycopg.Error, CorpusUnavailableError) as error:
                logger.warning("clinical_evidence_refresh_failed error_type=%s", type(error).__name__)

    def semantic(self, question: str) -> list[tuple[str, float]]:
        """(chunk_id, cosine similarity), best first; empty when embeddings are unavailable."""
        if self._embedder is None:
            return []
        try:
            vector = vector_literal(self._embedder.embed([question])[0])
            with self._pool.connection() as conn:
                rows = conn.execute(
                    "SELECT c.chunk_id, 1 - (e.embedding <=> %(v)s::vector) AS similarity "
                    "FROM corpus.citable_evidence c JOIN corpus.evidence_embedding_1024 e "
                    "  ON e.model_id = %(model)s AND e.body_sha256 = c.body_sha256 "
                    "WHERE c.release_id = %(release)s AND (%(unreviewed)s OR c.review_status = 'approved') "
                    "ORDER BY e.embedding <=> %(v)s::vector LIMIT %(k)s",
                    {"v": vector, "model": self._model_id, "release": self._release_id,
                     "unreviewed": self._include_unreviewed, "k": self._candidates},
                ).fetchall()
        except (EmbeddingError, psycopg.Error) as error:
            logger.warning("clinical_evidence_semantic_failed error_type=%s", type(error).__name__)
            return []
        return [(row["chunk_id"], float(row["similarity"])) for row in rows]

    def search(self, question: str, limit: int) -> tuple[EvidenceChunk, ...]:
        if not question.strip() or limit <= 0:
            return ()
        self._maybe_refresh()
        index = self._index
        lexical = index.scored(question)
        cutoff = lexical_cutoff(lexical)
        semantic = self.semantic(question)
        accepted = {chunk.chunk_id for score, chunk in lexical if score >= cutoff}
        accepted |= {chunk_id for chunk_id, similarity in semantic if similarity >= self._similarity_floor}
        fused: dict[str, float] = defaultdict(float)
        for rank, (_score, chunk) in enumerate(lexical[: self._candidates], start=1):
            fused[chunk.chunk_id] += 1 / (_RRF_K + rank)
        for rank, (chunk_id, _similarity) in enumerate(semantic, start=1):
            fused[chunk_id] += 1 / (_RRF_K + rank)
        ordered = sorted((chunk_id for chunk_id in accepted if index.chunk(chunk_id) is not None),
                         key=lambda chunk_id: (-fused[chunk_id], chunk_id))
        return tuple(prioritize(index.chunk(chunk_id) for chunk_id in ordered)[:limit])  # type: ignore[misc]
