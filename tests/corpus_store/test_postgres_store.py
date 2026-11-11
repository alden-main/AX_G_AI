"""Corpus store tests.

The PostgreSQL tests need a disposable pgvector database (they create and drop the
``corpus`` schema)::

    docker run --rm -d --name axg-corpus-it -e POSTGRES_PASSWORD=itpass -p 127.0.0.1:55499:5432 pgvector/pgvector:0.8.7-pg16
    AX_G_TEST_DATABASE_URL=postgresql://postgres:itpass@127.0.0.1:55499/postgres uv run python -m unittest tests.corpus_store.test_postgres_store
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import math
import os
import shutil
import tempfile
import unittest
from pathlib import Path

import httpx

from ax_g_ai.corpus_store.embedding import EmbeddingError, TeiEmbeddingClient, vector_literal

TEST_DATABASE_URL = os.getenv("AX_G_TEST_DATABASE_URL")
SOURCE_A = {"id": "PAPER-A", "title": "A 지침", "locator": "S1", "url": "https://example.org/a", "ai_use": "A", "access": "primary_text"}


def tei(handler) -> TeiEmbeddingClient:
    return TeiEmbeddingClient("http://embedding", dimensions=4, transport=httpx.MockTransport(handler))


class TeiEmbeddingClientTest(unittest.TestCase):
    def test_batches_requests_and_validates_vectors(self) -> None:
        seen: list[dict] = []

        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            seen.append(body)
            return httpx.Response(200, json=[[0.5, 0.5, 0.5, 0.5]] * len(body["inputs"]))

        vectors = tei(handler).embed([f"문단 {i}" for i in range(20)])
        self.assertEqual(len(vectors), 20)
        self.assertEqual([len(b["inputs"]) for b in seen], [16, 4])
        self.assertTrue(all(b["normalize"] and b["truncate"] for b in seen))

    def test_wrong_shape_status_or_values_raise_embedding_error(self) -> None:
        for response in (
            httpx.Response(200, json=[[0.1, 0.2]]),
            httpx.Response(200, json={"error": "x"}),
            httpx.Response(503, json=[]),
            httpx.Response(200, content=b"[[0.1, 0.2, NaN, 0.3]]"),
        ):
            with self.subTest(response=response.status_code), self.assertRaises(EmbeddingError):
                tei(lambda request, r=response: r).embed(["질문"])

    def test_vector_literal(self) -> None:
        self.assertEqual(vector_literal([1.0, -0.25, 1e-9]), "[1,-0.25,1e-09]")


def fake_vector(text: str, dimensions: int = 1024) -> list[float]:
    """Deterministic unit vector; texts sharing a topic word get similar vectors."""
    vector = [0.0] * dimensions
    for word in ("담낭", "게실", "칼륨", "혈압"):
        if word in text:
            vector[int(hashlib.sha256(word.encode()).hexdigest(), 16) % dimensions] += 1.0
    vector[int(hashlib.sha256(text.encode()).hexdigest(), 16) % dimensions] += 0.3
    norm = math.sqrt(sum(x * x for x in vector))
    return [x / norm for x in vector]


class FakeEmbedder:
    dimensions = 1024

    def __init__(self, fail: bool = False) -> None:
        self.fail, self.calls = fail, 0

    def embed(self, texts):
        self.calls += 1
        if self.fail:
            raise EmbeddingError("down")
        return [fake_vector(text) for text in texts]


def chunk(claim_id: str, title: str, text: str, **overrides) -> dict:
    record = {
        "chunk_id": f"{claim_id}-chunk", "claim_id": claim_id, "chapter": claim_id.split("-")[0],
        "title_path": ["장", title], "claim_type": "treatment", "claim_status": "개정", "ai_tier": "A",
        "verification": "primary", "review_status": "draft", "usable": False, "source_ids": ["PAPER-A"],
        "sources": [SOURCE_A], "conflicts": [], "updated_at": "2026-11-10", "text": f"{title}\n{text}", "doc_path": "updated/x.md",
    }
    record.update(overrides)
    return record


FILLER = [chunk(f"900-C{i:02d}", f"주제 {i} 관리", f"무관한 일반 문장 번호 {i} 내용") for i in range(1, 60)]


def write_corpus(directory: Path, chunks: list[dict]) -> None:
    files = {name: [] for name in (
        "diagnosis.jsonl", "workup.jsonl", "treatment.jsonl", "dose_rules.jsonl", "lab_thresholds.jsonl", "formulas.jsonl",
        "monitoring.jsonl", "referral.jsonl", "test_concepts.jsonl", "drug_concepts.jsonl", "drug_classes.jsonl")}
    files["treatment.jsonl"] = [{"id": "369-C04-K1", "claim_id": "369-C04", "kind": "treatment", "ai_tier": "A", "claim_status": "개정",
                                 "verification": "primary", "review_status": "draft", "usable": False, "sources": [SOURCE_A]}]
    files["drug_concepts.jsonl"] = [{"id": "cefazolin", "type": "ingredient", "ko": ["세파졸린"], "rxnorm": [{"rxcui": "2180", "name": "cefazolin", "tty": "IN"}],
                                     "used_by": [{"record_id": "369-C04-K1", "claim_id": "369-C04", "roles": ["recommend"]}], "used_by_class": []}]
    files["test_concepts.jsonl"] = [{"id": "K", "section": "labs", "name_ko": "칼륨", "loinc": [{"code": "2823-3", "name": "Potassium"}], "aliases": ["Potassium"], "used_by": []}]
    files["chunks.jsonl"] = chunks
    for name, rows in files.items():
        (directory / name).write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


CHUNKS = [
    chunk("369-C03", "급성 담낭염 수술 시기", "복강경 담낭절제를 입원 7일 이내 조기에 한다."),
    chunk("369-C04", "급성 담낭염 항생제", "담낭절제 후 항생제를 일상적으로 쓰지 않는다."),
    chunk("369-C10", "급성 담낭염 비상업", "담낭염 B등급 근거 문장", sources=[SOURCE_A, {**SOURCE_A, "id": "PAPER-B", "ai_use": "B"}]),
    chunk("353-C02", "합병증 없는 게실염", "항생제 없이 외래 관리한다."),
    *FILLER,
]


class PrecomputedFileTest(unittest.TestCase):
    def setUp(self) -> None:
        self.dir = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.dir)
        write_corpus(self.dir, CHUNKS)

    def test_round_trip_and_coverage(self) -> None:
        from ax_g_ai.corpus_store.precomputed import (
            coverage, embedding_input, embedding_key, embeddings_path, read_precomputed, write_precomputed,
        )

        total, missing, error = coverage(self.dir)
        self.assertEqual((len(missing), error), (total, "file not found"))
        vectors = {embedding_key(c): fake_vector(embedding_input(c)) for c in CHUNKS[:-1]}
        write_precomputed(embeddings_path(self.dir), {"model_id": "kure-v1", "dimensions": 1024}, vectors)
        total, missing, error = coverage(self.dir)
        self.assertEqual((total, len(missing), error), (len(CHUNKS), 1, None))
        _header, loaded = read_precomputed(embeddings_path(self.dir), model_id="kure-v1", dimensions=1024)
        key = embedding_key(CHUNKS[0])
        self.assertAlmostEqual(sum(a * b for a, b in zip(loaded[key], vectors[key])), 1.0, places=5)

    def test_rejects_other_model_and_bad_rows(self) -> None:
        from ax_g_ai.corpus_store.precomputed import (
            PrecomputedEmbeddingError, embeddings_path, read_precomputed, write_precomputed,
        )

        path = embeddings_path(self.dir)
        write_precomputed(path, {"model_id": "bge-m3", "dimensions": 1024}, {})
        with self.assertRaises(PrecomputedEmbeddingError):
            read_precomputed(path, model_id="kure-v1", dimensions=1024)
        path.write_text('{"format": "ax-g-embeddings/1", "model_id": "kure-v1", "dimensions": 1024}\n{"sha256": "x", "vector": "AAAA"}\n')
        with self.assertRaises(PrecomputedEmbeddingError):
            read_precomputed(path, model_id="kure-v1", dimensions=1024)

    def test_check_cli_exit_codes(self) -> None:
        from ax_g_ai.corpus_store.precomputed import main

        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(main(["--check", str(self.dir)]), 1)


@unittest.skipUnless(TEST_DATABASE_URL, "AX_G_TEST_DATABASE_URL not set (needs a disposable pgvector database)")
class PostgresCorpusStoreTest(unittest.TestCase):
    def setUp(self) -> None:
        import psycopg

        self.psycopg = psycopg
        self.conn = psycopg.connect(TEST_DATABASE_URL, autocommit=True)
        self.conn.execute("DROP SCHEMA IF EXISTS corpus CASCADE")
        self.addCleanup(self.conn.close)
        self.addCleanup(lambda: self.conn.execute("DROP SCHEMA IF EXISTS corpus CASCADE"))
        self.dir = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.dir)
        write_corpus(self.dir, CHUNKS)

    def load(self, embedder=None):
        from ax_g_ai.corpus_store.loader import load_corpus

        return load_corpus(self.conn, self.dir, embedder=embedder, model_id="test-model")

    def retriever(self, embedder, **kwargs):
        from ax_g_ai.corpus_store.retriever import PostgresEvidenceRetriever

        return PostgresEvidenceRetriever(TEST_DATABASE_URL, embedder=embedder, model_id="test-model", include_unreviewed=True, **kwargs)

    def test_load_reuse_and_incremental_embedding(self) -> None:
        first = self.load(FakeEmbedder())
        self.assertFalse(first.reused)
        self.assertEqual((first.embedded, first.missing_embeddings), (len(CHUNKS), 0))
        again = self.load(FakeEmbedder())
        self.assertEqual((again.release_id, again.reused, again.embedded), (first.release_id, True, 0))
        changed = [*CHUNKS[:-1], chunk("900-C99", "새 주제", "새 문장")]
        write_corpus(self.dir, changed)
        second = self.load(FakeEmbedder())
        self.assertNotEqual(second.release_id, first.release_id)
        self.assertEqual(second.embedded, 1)  # only the new passage
        active = self.conn.execute("SELECT release_id FROM corpus.corpus_release WHERE is_active").fetchall()
        self.assertEqual(active, [(second.release_id,)])
        codes = self.conn.execute("SELECT code_system, code FROM corpus.concept_code WHERE release_id = %s ORDER BY 1", (second.release_id,)).fetchall()
        self.assertEqual(codes, [("LOINC", "2823-3"), ("RXNORM", "2180")])

    def write_file(self, chunks, transform=lambda v: v) -> None:
        from ax_g_ai.corpus_store.precomputed import embedding_input, embedding_key, embeddings_path, write_precomputed

        write_precomputed(embeddings_path(self.dir, "test-model"), {"model_id": "test-model", "dimensions": 1024},
                          {embedding_key(c): transform(fake_vector(embedding_input(c))) for c in chunks})

    def test_precomputed_file_fills_vectors_and_server_computes_only_the_rest(self) -> None:
        self.write_file(CHUNKS[:-2])
        server = FakeEmbedder()
        result = self.load(server)
        self.assertEqual((result.from_file, result.embedded, result.missing_embeddings), (len(CHUNKS) - 2, 2, 0))
        self.assertEqual(server.calls, 2)  # parity sample + the two missing passages
        self.assertEqual(self.retriever(FakeEmbedder()).search("급성 담낭염 수술 시기", 1)[0].chunk_id, "369-C03-chunk")

    def test_precomputed_file_from_another_model_output_is_ignored(self) -> None:
        self.write_file(CHUNKS, transform=lambda v: list(reversed(v)))  # same shape, different model output
        result = self.load(FakeEmbedder())
        self.assertEqual((result.from_file, result.embedded), (0, len(CHUNKS)))
        self.assertTrue(any("cosine" in warning for warning in result.warnings))

    def test_precomputed_file_without_server_is_used_with_a_warning(self) -> None:
        self.write_file(CHUNKS)
        result = self.load(None)
        self.assertEqual((result.from_file, result.missing_embeddings), (len(CHUNKS), 0))
        self.assertTrue(any("without a parity check" in warning for warning in result.warnings))

    def test_embedding_failure_still_activates_and_later_fills(self) -> None:
        result = self.load(FakeEmbedder(fail=True))
        self.assertEqual(result.missing_embeddings, len(CHUNKS))
        self.assertTrue(result.warnings)
        self.assertEqual(self.load(FakeEmbedder()).embedded, len(CHUNKS))

    def test_rollback_by_activating_previous_release(self) -> None:
        from ax_g_ai.corpus_store.loader import CorpusLoadError, activate

        first = self.load()
        write_corpus(self.dir, CHUNKS[:-1])
        self.load()
        activate(self.conn, first.release_id)
        self.assertEqual(self.conn.execute("SELECT release_id FROM corpus.corpus_release WHERE is_active").fetchone()[0], first.release_id)
        with self.assertRaises(CorpusLoadError):
            activate(self.conn, 9999)

    def test_gate_excludes_mixed_license_passages_and_unreviewed_switch(self) -> None:
        self.load(FakeEmbedder())
        retriever = self.retriever(FakeEmbedder())
        self.assertEqual(len(retriever), len(CHUNKS) - 1)
        self.assertNotIn("369-C10-chunk", [c.chunk_id for c in retriever.search("급성 담낭염 비상업 근거", 5)])
        from ax_g_ai.corpus_store.retriever import PostgresEvidenceRetriever

        approved_only = PostgresEvidenceRetriever(TEST_DATABASE_URL, embedder=None, model_id="test-model", include_unreviewed=False)
        self.assertEqual(len(approved_only), 0)

    def test_rules_read_from_the_active_release_match_the_file_gate(self) -> None:
        from ax_g_ai.corpus_store.rules import read_rows_from_database
        from ax_g_ai.services.clinical_recommendation import ClinicalRecommender, RuleCorpus

        workup = {"id": "369-C03-K1", "claim_id": "369-C03", "chapter": "369", "kind": "workup", "topic": "담낭염 평가",
                  "ai_tier": "A", "claim_status": "개정", "verification": "primary", "review_status": "draft", "usable": False,
                  "sources": [SOURCE_A], "trigger": {"dx": "K81"}, "tests": [{"lab": "K", "reason": "전해질"}]}
        (self.dir / "workup.jsonl").write_text(json.dumps(workup, ensure_ascii=False) + "\n", encoding="utf-8")
        result = self.load()

        release_id, rows = read_rows_from_database(TEST_DATABASE_URL)
        self.assertEqual(release_id, result.release_id)
        self.assertEqual([row["id"] for row in rows["workup.jsonl"]], ["369-C03-K1"])
        self.assertNotIn("369-C10-chunk", [row["chunk_id"] for row in rows["chunks.jsonl"]])  # SQL gate
        self.assertEqual(len(RuleCorpus.from_rows(rows, include_unreviewed=False)), 0)
        engine = ClinicalRecommender(RuleCorpus.from_rows(rows, include_unreviewed=True))
        assessment = engine.assess(engine.prepare({"diagnosisList": [{"diagnosisCode": "K81.0"}]}))
        self.assertEqual([(item["record_id"], item["status"]) for item in assessment.items], [("369-C03-K1", "met")])
        self.assertEqual([chunk.chunk_id for chunk in assessment.evidence], ["369-C03-chunk"])
        self.assertEqual(assessment.items[0]["tests"][0]["name"], "칼륨")

    def test_hybrid_search_semantic_floor_and_fallback(self) -> None:
        self.load(FakeEmbedder())
        hybrid = self.retriever(FakeEmbedder(), similarity_floor=0.6)
        self.assertEqual(hybrid.search("급성 담낭염 수술 시기", 1)[0].chunk_id, "369-C03-chunk")
        # Lexically unrelated wording, semantically close (fake vectors share the topic word).
        semantic_only = hybrid.search("쓸개 담낭 문제", 3)
        self.assertTrue(semantic_only and semantic_only[0].chunk_id.startswith("369-"))
        self.assertEqual(self.retriever(FakeEmbedder(), similarity_floor=0.99).search("쓸개 담낭 문제", 3), ())
        # Concept named in the question (cefazolin → 369-C04) still boosts in the DB-backed index.
        self.assertEqual(hybrid.search("급성 담낭염 담낭절제 세파졸린", 1)[0].chunk_id, "369-C04-chunk")
        down = self.retriever(FakeEmbedder(fail=True))
        with self.assertLogs("uvicorn.error", level="WARNING"):
            self.assertEqual(down.search("급성 담낭염 수술 시기", 1)[0].chunk_id, "369-C03-chunk")

    def test_no_active_release_raises(self) -> None:
        from ax_g_ai.corpus_store.loader import ensure_schema
        from ax_g_ai.corpus_store.retriever import CorpusUnavailableError

        ensure_schema(self.conn)
        with self.assertRaises(CorpusUnavailableError):
            self.retriever(None)


class RuntimeFallbackTest(unittest.TestCase):
    def test_unreachable_database_falls_back_to_file_index(self) -> None:
        from ax_g_ai.config import RuntimeSettings
        from ax_g_ai.runtime import _evidence_retriever
        from ax_g_ai.services.clinical_evidence import InMemoryEvidenceIndex

        with tempfile.TemporaryDirectory() as raw:
            write_corpus(Path(raw), CHUNKS)
            settings = RuntimeSettings(
                corpus_database_url="postgresql://nobody:x@127.0.0.1:1/none", evidence_corpus_dir=raw, evidence_include_unreviewed=True,
            )
            with self.assertLogs("uvicorn.error", level="INFO") as logs:
                retriever = _evidence_retriever(settings)
        self.assertIsInstance(retriever, InMemoryEvidenceIndex)
        self.assertTrue(any("fallback=files" in line for line in logs.output))
        self.assertFalse(any("nobody" in line for line in logs.output))  # credentials never logged


if __name__ == "__main__":
    unittest.main()
