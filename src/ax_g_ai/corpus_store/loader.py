"""Load the generated corpus (``text_data/corpus/*.jsonl``) into PostgreSQL as a release.

Steps: ensure schema → insert a new inactive release (or reuse one with the same
manifest) → fill passage vectors (precomputed file ``corpus/embeddings/<model>.jsonl``
first, after a parity check against the embedding server; the server for the rest) →
activate atomically → prune old releases. Embedding failures do not block activation:
those passages are still found by lexical search, and the next load fills them.

CLI (compose one-shot service ``corpus-loader``)::

    ax-g-ai-load-corpus                    # env: AX_G_CORPUS_DATABASE_URL, AX_G_EMBEDDING_URL, ...
    ax-g-ai-load-corpus --activate 3       # roll back to release 3
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import psycopg
from psycopg.types.json import Jsonb

from ax_g_ai.corpus_store.embedding import EMBEDDING_DIMENSIONS, EmbeddingError, TeiEmbeddingClient, vector_literal
from ax_g_ai.corpus_store.precomputed import (
    PrecomputedEmbeddingError, cosine, embedding_input, embeddings_path, read_precomputed,
)
from ax_g_ai.config import ConfigurationError, LoaderSettings
from ax_g_ai.corpus.files import CHUNKS_FILE, CONCEPT_FILES, DEFAULT_EMBEDDING_MODEL, RECORD_FILES
from ax_g_ai.corpus.text import terms
from ax_g_ai.jsonl import parse_jsonl
from ax_g_ai.pg_release import ReleaseTable

logger = logging.getLogger("ax_g_ai.corpus_store")

REQUIRED_FILES = (CHUNKS_FILE, *RECORD_FILES, *CONCEPT_FILES)
KEEP_RELEASES = 5


class CorpusLoadError(RuntimeError):
    """The corpus files are missing or the loaded release failed validation."""


RELEASES = ReleaseTable("corpus", "corpus_release", "release_id", "ax_g_ai.corpus_store", KEEP_RELEASES, CorpusLoadError, "release")
ensure_schema = RELEASES.ensure_schema
activate = RELEASES.activate  # also the rollback: activate an older release


@dataclass
class LoadResult:
    release_id: int
    reused: bool
    counts: dict[str, int]
    embedded: int = 0          # computed by the embedding server
    from_file: int = 0         # taken from the precomputed file
    missing_embeddings: int = 0
    warnings: list[str] = field(default_factory=list)


def _sha(text: str | bytes) -> str:
    return hashlib.sha256(text.encode("utf-8") if isinstance(text, str) else text).hexdigest()


def read_corpus(corpus_dir: Path) -> tuple[dict[str, list[dict[str, Any]]], dict[str, dict[str, Any]], str]:
    missing = [name for name in REQUIRED_FILES if not (corpus_dir / name).is_file()]
    if missing:
        raise CorpusLoadError(f"corpus files missing: {', '.join(missing)}")
    rows: dict[str, list[dict[str, Any]]] = {}
    manifest: dict[str, dict[str, Any]] = {}
    for name in REQUIRED_FILES:
        raw = (corpus_dir / name).read_bytes()
        try:
            rows[name] = parse_jsonl(raw.decode("utf-8"))
        except ValueError as error:  # json.JSONDecodeError included
            raise CorpusLoadError(f"{name}: invalid JSON line ({error})") from error
        manifest[name] = {"count": len(rows[name]), "sha256": _sha(raw)}
    return rows, manifest, _sha(json.dumps(manifest, sort_keys=True))


def load_corpus(
    conn: psycopg.Connection, corpus_dir: Path, *, embedder: TeiEmbeddingClient | None = None,
    model_id: str = DEFAULT_EMBEDDING_MODEL, source_commit: str | None = None,
) -> LoadResult:
    rows, manifest, manifest_sha = read_corpus(corpus_dir)
    ensure_schema(conn)
    counts = {name: info["count"] for name, info in manifest.items()}
    existing = conn.execute(
        "SELECT release_id, is_active FROM corpus.corpus_release WHERE manifest_sha256 = %s ORDER BY release_id DESC LIMIT 1",
        (manifest_sha,),
    ).fetchone()
    if existing is not None:
        release_id, reused = existing[0], True
    else:
        with conn.transaction():
            release_id = conn.execute(
                "INSERT INTO corpus.corpus_release (built_at, source_commit, manifest, manifest_sha256) "
                "VALUES (now(), %s, %s, %s) RETURNING release_id",
                (source_commit, Jsonb(manifest), manifest_sha),
            ).fetchone()[0]
            _insert_release(conn, release_id, rows)
        reused = False
    _validate(conn, release_id, rows)
    result = LoadResult(release_id, reused, counts)
    _fill_embeddings(conn, release_id, corpus_dir, embedder, model_id, result)
    activate(conn, release_id)
    RELEASES.prune(conn)
    return result


def _insert_release(conn: psycopg.Connection, release_id: int, rows: dict[str, list[dict[str, Any]]]) -> None:
    chunks = rows[CHUNKS_FILE]
    records = [(name, record) for name in RECORD_FILES for record in rows[name]]
    sources: dict[str, dict[str, Any]] = {}
    for item in [*chunks, *(record for _name, record in records)]:
        for source in item.get("sources") or ():
            sources.setdefault(source["id"], source)
    with conn.cursor() as cur:
        cur.executemany(
            "INSERT INTO corpus.source (release_id, source_id, title, url, ai_use, access) VALUES (%s, %s, %s, %s, %s, %s)",
            [(release_id, sid, s.get("title") or sid, s.get("url"), s.get("ai_use", "U"), s.get("access")) for sid, s in sources.items()],
        )
        cur.executemany(
            "INSERT INTO corpus.evidence_chunk (release_id, chunk_id, claim_id, chapter, title_path, body, claim_type, claim_status, "
            "ai_tier, verification, review_status, sources, conflicts, updated_at, doc_path, body_sha256, search_tokens) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
            [(
                release_id, c["chunk_id"], c["claim_id"], str(c.get("chapter", "")), list(c.get("title_path") or ()), c.get("text", ""),
                c.get("claim_type"), c.get("claim_status", ""), c.get("ai_tier", ""), c.get("verification", ""), c.get("review_status", "draft"),
                Jsonb(c.get("sources") or []), Jsonb(c.get("conflicts") or []), c.get("updated_at"), c.get("doc_path"),
                _sha(embedding_input(c)), " ".join(terms(embedding_input(c))),
            ) for c in chunks],
        )
        cur.executemany(
            "INSERT INTO corpus.corpus_record (release_id, record_id, file, kind, claim_id, ai_tier, claim_status, verification, "
            "review_status, usable, payload) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
            [(
                release_id, r["id"], name, r.get("kind", name.removesuffix(".jsonl")), r.get("claim_id"), r.get("ai_tier", ""),
                r.get("claim_status"), r.get("verification"), r.get("review_status", "draft"), r.get("usable") is True, Jsonb(r),
            ) for name, r in records],
        )
        concepts, codes, usage = _concept_rows(release_id, rows)
        cur.executemany("INSERT INTO corpus.concept (release_id, concept_id, concept_type, name_ko, payload) VALUES (%s, %s, %s, %s, %s)", concepts)
        cur.executemany("INSERT INTO corpus.concept_code (release_id, code_system, code, concept_type, concept_id) VALUES (%s, %s, %s, %s, %s)", codes)
        cur.executemany(
            "INSERT INTO corpus.concept_usage (release_id, concept_type, concept_id, record_id, claim_id, roles, via_class) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s)", usage,
        )


def _concept_rows(release_id: int, rows: dict[str, list[dict[str, Any]]]) -> tuple[list[tuple], list[tuple], list[tuple]]:
    concepts, codes, usage = [], set(), []
    typed: Iterable[tuple[str, dict[str, Any]]] = [
        *(("test", row) for row in rows["test_concepts.jsonl"]),
        *((row.get("type", "ingredient"), row) for row in rows["drug_concepts.jsonl"]),
        *(("drug_class", row) for row in rows["drug_classes.jsonl"]),
    ]
    for concept_type, row in typed:
        payload = {key: value for key, value in row.items() if key not in ("used_by", "used_by_class")}
        name = row.get("name_ko") or (row.get("ko") or [None])[0]
        concepts.append((release_id, row["id"], concept_type, name, Jsonb(payload)))
        for code in row.get("loinc") or ():
            codes.add((release_id, "LOINC", code["code"], concept_type, row["id"]))
        for code in row.get("rxnorm") or ():
            codes.add((release_id, "RXNORM", str(code["rxcui"]), concept_type, row["id"]))
        for use in row.get("used_by") or ():
            usage.append((release_id, concept_type, row["id"], use["record_id"], use.get("claim_id"), list(use.get("roles") or ()), None))
        for use in row.get("used_by_class") or ():
            usage.append((release_id, concept_type, row["id"], use["record_id"], use.get("claim_id"), list(use.get("roles") or ()), use.get("via_class")))
    return concepts, sorted(codes), usage


def _validate(conn: psycopg.Connection, release_id: int, rows: dict[str, list[dict[str, Any]]]) -> None:
    expected = {
        "evidence_chunk": len(rows[CHUNKS_FILE]),
        "corpus_record": sum(len(rows[name]) for name in RECORD_FILES),
        "concept": sum(len(rows[name]) for name in CONCEPT_FILES),
    }
    for table, count in expected.items():
        actual = conn.execute(f"SELECT count(*) FROM corpus.{table} WHERE release_id = %s", (release_id,)).fetchone()[0]
        if actual != count:
            raise CorpusLoadError(f"release {release_id}: {table} has {actual} rows, expected {count}")


def _missing(conn: psycopg.Connection, release_id: int, model_id: str) -> list[tuple[str, str]]:
    return conn.execute(
        "SELECT DISTINCT c.body_sha256, array_to_string(c.title_path, ' > ') || E'\\n' || c.body "
        "FROM corpus.evidence_chunk c WHERE c.release_id = %s AND NOT EXISTS ("
        "  SELECT 1 FROM corpus.evidence_embedding_1024 e WHERE e.model_id = %s AND e.body_sha256 = c.body_sha256)",
        (release_id, model_id),
    ).fetchall()


def _missing_count(conn: psycopg.Connection, release_id: int, model_id: str) -> int:
    return len(_missing(conn, release_id, model_id))


PARITY_SAMPLE = 3
PARITY_MIN_COSINE = 0.99


def _store_vectors(conn: psycopg.Connection, model_id: str, items: list[tuple[str, list[float]]]) -> None:
    with conn.transaction():
        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO corpus.evidence_embedding_1024 (model_id, body_sha256, embedding) VALUES (%s, %s, %s::vector) "
                "ON CONFLICT (model_id, body_sha256) DO NOTHING",
                [(model_id, key, vector_literal(vector)) for key, vector in items],
            )


def _fill_embeddings(
    conn: psycopg.Connection, release_id: int, corpus_dir: Path, embedder: TeiEmbeddingClient | None,
    model_id: str, result: LoadResult,
) -> None:
    dimensions = embedder.dimensions if embedder is not None else EMBEDDING_DIMENSIONS
    with conn.transaction():
        conn.execute(
            "INSERT INTO corpus.embedding_model (model_id, dimensions) VALUES (%s, %s) ON CONFLICT (model_id) DO NOTHING",
            (model_id, dimensions),
        )
    todo = _missing(conn, release_id, model_id)
    path = embeddings_path(corpus_dir, model_id)
    if todo and path.is_file():
        try:
            header, file_vectors = read_precomputed(path, model_id=model_id, dimensions=dimensions)
        except PrecomputedEmbeddingError as error:
            result.warnings.append(f"precomputed embeddings ignored: {error}")
        else:
            usable = [(key, file_vectors[key]) for key, _text in todo if key in file_vectors]
            if usable and _parity_ok(embedder, todo, file_vectors, result):
                _store_vectors(conn, model_id, usable)
                result.from_file = len(usable)
                logger.info("precomputed embeddings used: %d (generator=%s)", len(usable), header.get("generator"))
        todo = _missing(conn, release_id, model_id)
    if embedder is not None:
        _embed_with_server(conn, todo, embedder, model_id, result)
    result.missing_embeddings = _missing_count(conn, release_id, model_id)


def _parity_ok(
    embedder: TeiEmbeddingClient | None, todo: list[tuple[str, str]], file_vectors: dict[str, list[float]], result: LoadResult,
) -> bool:
    """File vectors must match the serving model, or query and passage vectors would not be comparable."""
    if embedder is None:
        result.warnings.append("precomputed embeddings used without a parity check (no embedding server)")
        return True
    sample = [(key, text) for key, text in todo if key in file_vectors][:PARITY_SAMPLE]
    try:
        served = embedder.embed([text for _key, text in sample])
    except EmbeddingError as error:
        result.warnings.append(f"parity check skipped, embedding server unavailable: {error}")
        return True
    worst = min(cosine(file_vectors[key], vector) for (key, _text), vector in zip(sample, served))
    if worst < PARITY_MIN_COSINE:
        result.warnings.append(f"precomputed embeddings ignored: cosine {worst:.4f} vs embedding server < {PARITY_MIN_COSINE}")
        return False
    return True


def _embed_with_server(
    conn: psycopg.Connection, todo: list[tuple[str, str]], embedder: TeiEmbeddingClient, model_id: str, result: LoadResult,
) -> None:
    for start in range(0, len(todo), 32):
        batch = todo[start:start + 32]
        try:
            vectors = embedder.embed([text for _sha_key, text in batch])
        except EmbeddingError as error:
            result.warnings.append(f"embedding stopped: {error}")
            break
        _store_vectors(conn, model_id, [(key, vector) for (key, _text), vector in zip(batch, vectors)])
        result.embedded += len(batch)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Load the clinical corpus into PostgreSQL.")
    parser.add_argument("--corpus-dir", type=Path, help="corpus folder (default: AX_G_EVIDENCE_CORPUS_DIR or text_data/corpus)")
    parser.add_argument("--activate", type=int, metavar="RELEASE_ID", help="activate an existing release (rollback) and exit")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    try:
        settings = LoaderSettings.from_environment()
    except ConfigurationError as error:  # field names and reasons only, never values
        print(f"invalid configuration: {error}", file=sys.stderr)
        return 2
    embedder = None if settings.embedding_url is None else TeiEmbeddingClient(
        settings.embedding_url, timeout_seconds=settings.embedding_timeout_seconds,
    )
    try:
        with psycopg.connect(settings.corpus_database_url.get_secret_value(), autocommit=True, connect_timeout=10) as conn:
            if args.activate is not None:
                ensure_schema(conn)
                activate(conn, args.activate)
                print(f"activated release {args.activate}")
                return 0
            result = load_corpus(conn, args.corpus_dir or settings.evidence_corpus_dir, embedder=embedder,
                                 model_id=settings.embedding_model, source_commit=settings.corpus_source_commit)
    except (psycopg.Error, CorpusLoadError) as error:
        # Error class and message only; the database URL (with credentials) is never printed.
        print(f"corpus load failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    finally:
        if embedder is not None:
            embedder.close()
    print(f"release {result.release_id} active ({'reused' if result.reused else 'new'}): "
          f"chunks {result.counts[CHUNKS_FILE]}, from file {result.from_file}, embedded {result.embedded}, "
          f"missing embeddings {result.missing_embeddings}")
    for warning in result.warnings:
        print(f"warning: {warning}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
