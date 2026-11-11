"""Read the active release's rule records, concepts and citable passages from PostgreSQL.

Returns rows shaped like ``text_data/corpus/*.jsonl`` so ``RuleCorpus.from_rows`` applies
the same Python gate (``is_citable``) to database and file rows alike.
"""

from __future__ import annotations

from typing import Any

import psycopg
from psycopg.rows import dict_row

from ax_g_ai.corpus_store.retriever import CorpusUnavailableError
from ax_g_ai.corpus.files import CHUNKS_FILE, CONCEPT_FILES, RULE_FILES

_CONCEPT_FILE = {"test": "test_concepts.jsonl", "ingredient": "drug_concepts.jsonl", "product": "drug_concepts.jsonl",
                 "drug_class": "drug_classes.jsonl"}


def read_rows_from_database(database_url: str) -> tuple[int, dict[str, list[dict[str, Any]]]]:
    """(release_id, rows by corpus file name) for the active release."""
    rows: dict[str, list[dict[str, Any]]] = {name: [] for name in (*RULE_FILES, *CONCEPT_FILES, CHUNKS_FILE)}
    with psycopg.connect(database_url, autocommit=True, connect_timeout=3, row_factory=dict_row) as conn:
        active = conn.execute("SELECT release_id FROM corpus.corpus_release WHERE is_active").fetchone()
        if active is None:
            raise CorpusUnavailableError("no active corpus release")
        release_id = active["release_id"]
        for row in conn.execute(
            "SELECT file, payload FROM corpus.corpus_record WHERE release_id = %s AND file = ANY(%s) ORDER BY record_id",
            (release_id, list(RULE_FILES)),
        ):
            rows[row["file"]].append(row["payload"])
        for row in conn.execute(
            "SELECT concept_type, payload FROM corpus.concept WHERE release_id = %s ORDER BY concept_id", (release_id,),
        ):
            name = _CONCEPT_FILE.get(row["concept_type"])
            if name is not None:
                rows[name].append(row["payload"])
        rows[CHUNKS_FILE] = list(conn.execute(
            "SELECT chunk_id, claim_id, chapter, title_path, body AS text, claim_status, ai_tier, verification, "
            "review_status, sources, conflicts, updated_at::text AS updated_at, doc_path "
            "FROM corpus.citable_evidence WHERE release_id = %s ORDER BY chunk_id", (release_id,),
        ))
    return release_id, rows
