"""Read one release's rule records, concepts and citable passages from PostgreSQL.

Returns rows shaped like ``text_data/corpus/*.jsonl`` so ``RuleCorpus.from_rows`` and
``InMemoryEvidenceIndex.from_rows`` build the rules and the evidence index from one read and apply
the same Python gate (``is_citable``) to database and file rows alike.
"""

from __future__ import annotations

from typing import Any

import psycopg
from psycopg.rows import dict_row

from ax_g_ai.corpus.files import CHUNKS_FILE, CONCEPT_FILES, RULE_FILES

_CONCEPT_FILE = {"test": "test_concepts.jsonl", "ingredient": "drug_concepts.jsonl", "product": "drug_concepts.jsonl",
                 "drug_class": "drug_classes.jsonl"}


def read_rows(conn: psycopg.Connection, release_id: int) -> dict[str, list[dict[str, Any]]]:
    """Rows by corpus file name for ``release_id`` (the active release: passages come from the gate view)."""
    rows: dict[str, list[dict[str, Any]]] = {name: [] for name in (*RULE_FILES, *CONCEPT_FILES, CHUNKS_FILE)}
    with conn.cursor(row_factory=dict_row) as cur:
        for row in cur.execute(
            "SELECT file, payload FROM corpus.corpus_record WHERE release_id = %s AND file = ANY(%s) ORDER BY record_id",
            (release_id, list(RULE_FILES)),
        ):
            rows[row["file"]].append(row["payload"])
        # The stored payload leaves out ``used_by`` (kept in concept_usage); the evidence index needs the claims using
        # each concept, so they are joined back in the shape of the jsonl rows.
        for row in cur.execute(
            "SELECT c.concept_type, c.payload, coalesce(json_agg(json_build_object('claim_id', u.claim_id)) "
            "  FILTER (WHERE u.claim_id IS NOT NULL), '[]') AS used_by "
            "FROM corpus.concept c LEFT JOIN corpus.concept_usage u USING (release_id, concept_type, concept_id) "
            "WHERE c.release_id = %s GROUP BY c.concept_type, c.concept_id, c.payload ORDER BY c.concept_id", (release_id,),
        ):
            name = _CONCEPT_FILE.get(row["concept_type"])
            if name is not None:
                rows[name].append({**row["payload"], "used_by": row["used_by"]})
        rows[CHUNKS_FILE] = list(cur.execute(
            "SELECT chunk_id, claim_id, chapter, title_path, body AS text, claim_status, ai_tier, verification, "
            "review_status, sources, conflicts, updated_at::text AS updated_at, doc_path "
            "FROM corpus.citable_evidence WHERE release_id = %s ORDER BY chunk_id", (release_id,),
        ))
    return rows


def read_rows_from_database(database_url: str) -> tuple[int, dict[str, list[dict[str, Any]]]]:
    """(release_id, rows) for the active release over a one-off connection (tools and tests)."""
    from ax_g_ai.corpus_store.retriever import active_release_id  # the retriever imports this module

    with psycopg.connect(database_url, autocommit=True, connect_timeout=3) as conn:
        release_id = active_release_id(conn)
        return release_id, read_rows(conn, release_id)
