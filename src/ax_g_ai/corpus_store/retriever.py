"""Hybrid evidence retrieval from PostgreSQL: lexical (in-process BM25) + pgvector.

One ``PostgresEvidenceRetriever`` serves one corpus release: its citable passages
(``corpus.citable_evidence``, the single gate) are loaded into an ``InMemoryEvidenceIndex``
for lexical and concept scoring, so lexical behaviour matches the file-based index. Semantic
candidates come from pgvector. A passage is cited when its lexical score passes the lexical
cutoff **or** its cosine similarity passes ``similarity_floor``; accepted passages are ordered
by reciprocal rank fusion. If the embedding server or database fails at query time, the
loaded lexical index still answers.

A newer release is picked up by ``corpus_store.live.LiveCorpus``, which builds a new retriever
together with the rules of the same release and swaps both at once.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

import psycopg
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from ax_g_ai.corpus_store.embedding import EmbeddingError, TeiEmbeddingClient, vector_literal
from ax_g_ai.corpus_store.rules import read_rows
from ax_g_ai.services.clinical_evidence import EvidenceChunk, InMemoryEvidenceIndex, lexical_cutoff, prioritize

logger = logging.getLogger("uvicorn.error")
_RRF_K = 60


class CorpusUnavailableError(RuntimeError):
    """No active corpus release is available in the database."""


def open_pool(database_url: str, *, max_size: int = 10, wait_seconds: float = 0) -> ConnectionPool:
    """Connection pool for request-path queries; raises ``psycopg.Error`` (PoolTimeout) when the database is down.

    ``wait_seconds`` (a database starting with the API): the pool keeps reconnecting with backoff for up to that
    long before giving up, at least 3 s.
    """
    pool = ConnectionPool(
        database_url, min_size=1, max_size=max_size, timeout=3, open=False,
        kwargs={"autocommit": True, "connect_timeout": 3, "row_factory": dict_row},
    )
    try:
        pool.open(wait=True, timeout=max(3.0, wait_seconds))
    except BaseException:
        pool.close()  # stop the pool's reconnect workers before the caller falls back to files
        raise
    return pool


def active_release_id(conn: psycopg.Connection) -> int:
    row = conn.execute("SELECT release_id FROM corpus.corpus_release WHERE is_active").fetchone()
    if row is None:
        raise CorpusUnavailableError("no active corpus release")
    return _first(row)


def load_index(
    conn: psycopg.Connection, release_id: int, *, include_unreviewed: bool, require_license: bool = False,
) -> InMemoryEvidenceIndex:
    """The citable passages of ``release_id`` (it must be the active release: the gate view reads only that one).

    For a standalone retriever (``open``). The service reads ``read_rows`` once and builds the index and the rules from it.
    """
    # is_citable() runs again in Python as a second gate behind the SQL view.
    return InMemoryEvidenceIndex.from_rows(read_rows(conn, release_id), include_unreviewed=include_unreviewed, require_license=require_license)


class PostgresEvidenceRetriever:
    """Evidence search over one loaded release; the connection pool and embedder are shared, not owned."""

    def __init__(
        self, pool: ConnectionPool, release_id: int, index: InMemoryEvidenceIndex, *, embedder: TeiEmbeddingClient | None,
        model_id: str, include_unreviewed: bool, similarity_floor: float = 0.6, candidates: int = 30,
    ) -> None:
        self._pool = pool
        self._release_id = release_id
        self._index = index
        self._embedder = embedder
        self._model_id = model_id
        self._include_unreviewed = include_unreviewed
        self._similarity_floor = similarity_floor
        self._candidates = candidates

    @classmethod
    def open(
        cls, database_url: str, *, embedder: TeiEmbeddingClient | None, model_id: str, include_unreviewed: bool,
        require_license: bool = False, **options: Any,
    ) -> PostgresEvidenceRetriever:
        """A retriever for the active release with its own pool (tools and tests; the service uses ``LiveCorpus``)."""
        pool = open_pool(database_url)
        try:
            with pool.connection() as conn:
                release_id = active_release_id(conn)
                index = load_index(conn, release_id, include_unreviewed=include_unreviewed, require_license=require_license)
        except BaseException:
            pool.close()
            raise
        return cls(pool, release_id, index, embedder=embedder, model_id=model_id, include_unreviewed=include_unreviewed, **options)

    def close(self) -> None:
        """Close the pool (only for retrievers made by ``open``)."""
        self._pool.close()

    def __len__(self) -> int:
        return len(self._index)

    @property
    def release_id(self) -> int:
        return self._release_id

    @property
    def unreviewed_count(self) -> int:
        return self._index.unreviewed_count

    def semantic(self, question: str) -> list[tuple[str, float]]:
        """(chunk_id, cosine similarity), best first; empty when embeddings are unavailable."""
        if self._embedder is None:
            return []
        try:
            vector = vector_literal(self._embedder.embed([question])[0])
            with self._pool.connection() as conn:
                rows = conn.cursor(row_factory=dict_row).execute(
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


def _first(row: Any) -> Any:
    """First column of a row from a tuple or dict row factory."""
    return row[0] if isinstance(row, tuple) else next(iter(row.values()))
