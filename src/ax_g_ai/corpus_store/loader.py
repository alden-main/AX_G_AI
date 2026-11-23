"""생성된 말뭉치(``text_data/corpus/*.jsonl``)를 PostgreSQL에 릴리스로 적재하는 모듈과 CLI.

설계상 위치: 말뭉치 저장 계층의 쓰기 경로. compose 일회성 서비스 ``corpus-loader``(``ax-g-ai-load-corpus``)가 실행하며,
서비스 런타임은 이 결과를 ``retriever``·``rules``·``live``로 읽는다.

역할: 스키마 보장 → 새 비활성 릴리스 삽입(같은 manifest의 릴리스가 있으면 재사용) → 문단 벡터 채우기(먼저 사전 계산
파일 ``corpus/embeddings/<model>.jsonl``을 임베딩 서버와 일치 검사한 뒤 사용, 나머지는 서버로 계산) → 원자적 활성화 →
오래된 릴리스 정리. 임베딩 실패는 활성화를 막지 않는다. 그 문단은 어휘 검색으로 여전히 찾을 수 있고 다음 적재에서 채워진다.

CLI(compose 일회성 서비스 ``corpus-loader``)::

    ax-g-ai-load-corpus                    # 환경 변수: AX_G_CORPUS_DATABASE_URL, AX_G_EMBEDDING_URL, ...
    ax-g-ai-load-corpus --activate 3       # 릴리스 3으로 되돌리기
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import psycopg
from psycopg.types.json import Jsonb

from ax_g_ai.config import ConfigurationError, LoaderSettings
from ax_g_ai.corpus.files import CHUNKS_FILE, CONCEPT_FILES, DEFAULT_EMBEDDING_MODEL, RECORD_FILES
from ax_g_ai.corpus.text import terms
from ax_g_ai.corpus_store.embedding import EMBEDDING_DIMENSIONS, EmbeddingError, TeiEmbeddingClient, vector_literal
from ax_g_ai.corpus_store.precomputed import (
    PrecomputedEmbeddingError,
    cosine,
    embedding_input,
    embeddings_path,
    read_precomputed,
)
from ax_g_ai.jsonl import parse_jsonl
from ax_g_ai.pg_release import ReleaseTable, scalar

logger = logging.getLogger("ax_g_ai.corpus_store")

REQUIRED_FILES = (CHUNKS_FILE, *RECORD_FILES, *CONCEPT_FILES)
KEEP_RELEASES = 5


class CorpusLoadError(RuntimeError):
    """말뭉치 파일이 없거나 적재한 릴리스가 검증에 실패했을 때의 오류.

    역할: CLI가 DB 오류와 함께 잡아 종료 코드 1로 보고하는 적재 실패 예외. 롤백 대상 릴리스가 없을 때도
    ``ReleaseTable``이 이 예외를 쓴다.
    """


RELEASES = ReleaseTable("corpus", "corpus_release", "release_id", "ax_g_ai.corpus_store", KEEP_RELEASES, CorpusLoadError, "release")
ensure_schema = RELEASES.ensure_schema
activate = RELEASES.activate  # 롤백도 이것으로 한다: 이전 릴리스를 활성화


@dataclass
class LoadResult:
    """말뭉치 적재 한 번의 결과.

    역할: 릴리스 ID, 재사용 여부, 파일별 행 수, 임베딩 출처별 개수, 경고를 CLI 출력용으로 모은다.
    """

    release_id: int
    reused: bool
    counts: dict[str, int]
    embedded: int = 0          # 임베딩 서버가 계산한 수
    from_file: int = 0         # 사전 계산 파일에서 가져온 수
    missing_embeddings: int = 0
    warnings: list[str] = field(default_factory=list)


def _sha(text: str | bytes) -> str:
    """문자열(UTF-8)이나 바이트의 SHA-256 16진 문자열."""
    return hashlib.sha256(text.encode("utf-8") if isinstance(text, str) else text).hexdigest()


def read_corpus(corpus_dir: Path) -> tuple[dict[str, list[dict[str, Any]]], dict[str, dict[str, Any]], str]:
    """필수 말뭉치 파일을 모두 읽고 manifest를 만든다.

    역할: ``REQUIRED_FILES``가 모두 있는지 확인하고, 파일마다 JSONL을 파싱해 행 수와 원본 바이트 SHA-256을 기록한다.
    입력:
        corpus_dir: 말뭉치 폴더.
    출력: (파일별 행, 파일별 {count, sha256} manifest, manifest JSON의 SHA-256).
    예외: 파일이 없거나 JSON 줄이 잘못되면 ``CorpusLoadError``.
    """
    missing = [name for name in REQUIRED_FILES if not (corpus_dir / name).is_file()]
    if missing:
        raise CorpusLoadError(f"corpus files missing: {', '.join(missing)}")
    rows: dict[str, list[dict[str, Any]]] = {}
    manifest: dict[str, dict[str, Any]] = {}
    for name in REQUIRED_FILES:
        raw = (corpus_dir / name).read_bytes()
        try:
            rows[name] = parse_jsonl(raw.decode("utf-8"))
        except ValueError as error:  # json.JSONDecodeError 포함
            raise CorpusLoadError(f"{name}: invalid JSON line ({error})") from error
        manifest[name] = {"count": len(rows[name]), "sha256": _sha(raw)}
    return rows, manifest, _sha(json.dumps(manifest, sort_keys=True))


def load_corpus(
    conn: psycopg.Connection, corpus_dir: Path, *, embedder: TeiEmbeddingClient | None = None,
    model_id: str = DEFAULT_EMBEDDING_MODEL, source_commit: str | None = None,
) -> LoadResult:
    """말뭉치 폴더를 릴리스로 적재하고 활성화한다.

    역할: 같은 manifest SHA-256의 릴리스가 있으면 재사용하고, 없으면 한 트랜잭션에서 새 릴리스와 모든 행을 넣는다.
    이어 행 수를 검증하고 임베딩을 채운 뒤 활성화하고 오래된 릴리스를 정리한다.
    입력:
        conn: autocommit PostgreSQL 연결.
        corpus_dir: 말뭉치 폴더.
        embedder: 임베딩 서버 클라이언트. ``None``이면 사전 계산 파일만 쓴다(일치 검사 없음).
        model_id: 임베딩 모델 ID.
        source_commit: 릴리스에 기록할 원본 커밋(선택).
    출력: ``LoadResult``.
    예외: 파일 누락·검증 실패면 ``CorpusLoadError``, DB 오류면 ``psycopg.Error``.
    """
    rows, manifest, manifest_sha = read_corpus(corpus_dir)
    ensure_schema(conn)
    counts = {name: info["count"] for name, info in manifest.items()}
    existing = conn.execute(
        "SELECT release_id FROM corpus.corpus_release WHERE manifest_sha256 = %s ORDER BY release_id DESC LIMIT 1",
        (manifest_sha,),
    ).fetchone()
    if existing is not None:
        release_id, reused = existing[0], True
    else:
        with conn.transaction():
            release_id = scalar(conn.execute(
                "INSERT INTO corpus.corpus_release (built_at, source_commit, manifest, manifest_sha256) "
                "VALUES (now(), %s, %s, %s) RETURNING release_id",
                (source_commit, Jsonb(manifest), manifest_sha),
            ))
            _insert_release(conn, release_id, rows)
        reused = False
    _validate(conn, release_id, rows)
    result = LoadResult(release_id, reused, counts)
    _fill_embeddings(conn, release_id, corpus_dir, embedder, model_id, result)
    activate(conn, release_id)
    RELEASES.prune(conn)
    return result


def _insert_release(conn: psycopg.Connection, release_id: int, rows: dict[str, list[dict[str, Any]]]) -> None:
    """릴리스 하나의 출처·문단·규칙 레코드·개념 행을 DB에 넣는다.

    역할: 문단과 레코드에서 출처를 ID별로 모아(처음 나온 것 우선) ``source``에 넣고, 문단은 임베딩 입력 SHA-256과
    검색 토큰을 함께 ``evidence_chunk``에, 레코드는 ``corpus_record``에, 개념·코드·사용처는 ``_concept_rows``로 넣는다.
    입력:
        conn: PostgreSQL 연결(호출자가 트랜잭션을 연다).
        release_id: 새 릴리스 ID.
        rows: ``read_corpus``의 파일별 행.
    출력: 없음.
    """
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
    """개념 파일들을 ``concept``·``concept_code``·``concept_usage`` 삽입 행으로 바꾼다.

    역할: 검사 개념은 ``test``, 약 개념은 행의 ``type``(기본 ``ingredient``), 약물 분류는 ``drug_class``로 분류한다.
    payload에서는 ``used_by``·``used_by_class``를 빼고(사용처 표로 분리), LOINC·RxNorm 코드는 중복 없이 모은다.
    입력:
        release_id: 릴리스 ID.
        rows: 파일별 행.
    출력: (개념 행, 정렬된 코드 행, 사용처 행).
    """
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
    """적재된 문단·레코드·개념 행 수가 파일 행 수와 같은지 확인한다.

    역할: 재사용 릴리스를 포함해 활성화 전에 DB 내용이 파일과 맞는지 검증한다.
    입력:
        conn: PostgreSQL 연결.
        release_id: 검증할 릴리스 ID.
        rows: 파일별 행.
    출력: 없음.
    예외: 개수가 다르면 ``CorpusLoadError``.
    """
    expected = {
        "evidence_chunk": len(rows[CHUNKS_FILE]),
        "corpus_record": sum(len(rows[name]) for name in RECORD_FILES),
        "concept": sum(len(rows[name]) for name in CONCEPT_FILES),
    }
    for table, count in expected.items():
        actual = scalar(conn.execute(f"SELECT count(*) FROM corpus.{table} WHERE release_id = %s", (release_id,)))
        if actual != count:
            raise CorpusLoadError(f"release {release_id}: {table} has {actual} rows, expected {count}")


def _missing(conn: psycopg.Connection, release_id: int, model_id: str) -> list[tuple[str, str]]:
    """이 모델의 임베딩이 아직 없는 문단의 (본문 SHA-256, 임베딩 입력 텍스트) 목록(SHA-256 기준 중복 제거)."""
    return conn.execute(
        "SELECT DISTINCT c.body_sha256, array_to_string(c.title_path, ' > ') || E'\\n' || c.body "
        "FROM corpus.evidence_chunk c WHERE c.release_id = %s AND NOT EXISTS ("
        "  SELECT 1 FROM corpus.evidence_embedding_1024 e WHERE e.model_id = %s AND e.body_sha256 = c.body_sha256)",
        (release_id, model_id),
    ).fetchall()


def _missing_count(conn: psycopg.Connection, release_id: int, model_id: str) -> int:
    """임베딩이 없는 서로 다른 문단 본문 수."""
    return len(_missing(conn, release_id, model_id))


PARITY_SAMPLE = 3
PARITY_MIN_COSINE = 0.99


def _store_vectors(conn: psycopg.Connection, model_id: str, items: list[tuple[str, list[float]]]) -> None:
    """(본문 SHA-256, 벡터)들을 한 트랜잭션에서 저장한다. 이미 있는 키는 건너뛴다.

    역할: ``evidence_embedding_1024``에 ``ON CONFLICT DO NOTHING``으로 넣는다.
    입력:
        conn: PostgreSQL 연결.
        model_id: 임베딩 모델 ID.
        items: (본문 SHA-256, 벡터) 목록.
    출력: 없음.
    """
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
    """릴리스 문단 중 임베딩이 없는 것을 사전 계산 파일과 임베딩 서버로 채운다.

    역할: 모델을 등록하고, 누락 문단 중 파일에 있는 벡터를 일치 검사(``_parity_ok``) 통과 시 저장한다. 파일이 잘못됐으면
    경고만 남긴다. 남은 누락분은 서버가 있으면 서버로 계산하고, 마지막에 여전히 없는 개수를 기록한다.
    입력:
        conn: PostgreSQL 연결.
        release_id: 릴리스 ID.
        corpus_dir: 말뭉치 폴더(사전 계산 파일 위치).
        embedder: 임베딩 서버 클라이언트 또는 ``None``.
        model_id: 임베딩 모델 ID.
        result: 개수·경고를 갱신할 ``LoadResult``.
    출력: 없음(``result``를 갱신).
    """
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
    """사전 계산 벡터가 서빙 모델의 출력과 일치하는지 표본으로 확인한다.

    역할: 일치하지 않으면 질문 벡터와 문단 벡터를 비교할 수 없으므로, 파일에 있는 누락 문단 최대 ``PARITY_SAMPLE``개를
    서버로 임베딩해 최저 코사인이 ``PARITY_MIN_COSINE`` 이상인지 본다. 서버가 없거나 응답하지 않으면 경고를 남기고 통과시킨다.
    입력:
        embedder: 임베딩 서버 클라이언트 또는 ``None``.
        todo: 누락 문단의 (SHA-256, 텍스트) 목록.
        file_vectors: 파일의 {SHA-256: 벡터}.
        result: 경고를 추가할 ``LoadResult``.
    출력: 파일 벡터를 써도 되면 True, 불일치면 False.
    """
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
    """누락 문단을 32개씩 서버로 임베딩해 저장한다.

    역할: 임베딩 오류가 나면 경고를 남기고 멈춘다(이미 저장한 배치는 유지).
    입력:
        conn: PostgreSQL 연결.
        todo: (SHA-256, 텍스트) 목록.
        embedder: 임베딩 서버 클라이언트.
        model_id: 임베딩 모델 ID.
        result: ``embedded``·경고를 갱신할 ``LoadResult``.
    출력: 없음.
    """
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
    """``ax-g-ai-load-corpus`` CLI 진입점.

    역할: 환경 설정을 읽고, ``--activate``면 해당 릴리스만 활성화(롤백)하고, 아니면 말뭉치를 적재해 결과와 경고를 출력한다.
    오류 출력에는 오류 클래스와 메시지만 쓰며 DB URL은 출력하지 않는다.
    입력:
        argv: 명령행 인자(``None``이면 ``sys.argv``).
    출력: 종료 코드. 성공 0, 적재 실패 1, 설정 오류 2.
    """
    parser = argparse.ArgumentParser(description="Load the clinical corpus into PostgreSQL.")
    parser.add_argument("--corpus-dir", type=Path, help="corpus folder (default: AX_G_EVIDENCE_CORPUS_DIR or text_data/corpus)")
    parser.add_argument("--activate", type=int, metavar="RELEASE_ID", help="activate an existing release (rollback) and exit")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    try:
        settings = LoaderSettings.from_environment()
    except ConfigurationError as error:  # 필드 이름과 사유만 담고 값은 절대 담지 않는다
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
    except (psycopg.Error, CorpusLoadError, OSError) as error:
        # 오류 클래스와 메시지만 출력한다. (자격 증명이 담긴) DB URL은 절대 출력하지 않는다.
        # OSError: 말뭉치·임베딩 파일을 읽지 못한 경우(권한 등). DUR 적재기와 같이 종료 코드 1로 끝낸다.
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
