from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from ax_g_ai.adapters.main_bridge import RawEmrPayloadMapper
from ax_g_ai.api import create_app
from ax_g_ai.services.clinical_chat import ClinicalChatService
from ax_g_ai.corpus.gate import is_citable
from ax_g_ai.services.clinical_evidence import InMemoryEvidenceIndex
from tests.services.test_clinical_chat import SNAPSHOT, RecordingProvider

REPO_CORPUS = Path(__file__).resolve().parents[2] / "text_data" / "corpus"
SOURCE_A = {"id": "PAPER-WSES-CHOLECYSTITIS-2020", "title": "WSES 2020", "locator": "Statements 3.1–3.5", "url": "https://example.org/wses", "ai_use": "A"}


def chunk(claim_id: str, title: str, text: str, **overrides: object) -> dict[str, object]:
    record = {
        "chunk_id": f"{claim_id}-chunk", "claim_id": claim_id, "chapter": claim_id.split("-")[0],
        "title_path": ["담낭 및 담도질환", title], "claim_status": "개정", "ai_tier": "A", "verification": "primary",
        "review_status": "draft", "sources": [SOURCE_A], "conflicts": ["조기 수술 시점: 원문 48–72시간 vs WSES 7일 이내"],
        "updated_at": "2026-11-10", "text": f"{title}\n{text}",
    }
    record.update(overrides)
    return record


CHUNKS = [
    chunk("369-C03", "급성 담낭염 수술 시기", "복강경 담낭절제가 1차 치료이고 입원 7일 이내 조기 수술을 한다."),
    chunk("369-C04", "급성 담낭염 항생제", "합병증 없는 담낭염에서 담낭절제 후 항생제를 일상적으로 쓰지 않는다.", review_status="approved"),
    chunk("369-C09", "근거 없음", "담낭 확인불가 문장", claim_status="확인불가"),
    chunk("369-C10", "급성 담낭염 비상업 출처", "담낭염 B등급 문장", sources=[{**SOURCE_A, "ai_use": "B"}]),
    chunk("369-C11", "급성 담낭염 2차 자료", "담낭염 2차 자료 문장", verification="secondary"),
    chunk("369-C12", "급성 담낭염 반려", "담낭염 반려 문장", review_status="rejected"),
]


# Unrelated chunks so term rarity (IDF) resembles the real ~280-chunk corpus the score cutoff is tuned for.
FILLER_TOPICS = ("갑상선 기능 저하", "골다공증 골절 예방", "편두통 예방 약물", "천식 흡입기 사용", "통풍 요산 저하", "빈혈 철분 보충",
                 "우울증 초기 약물", "금연 약물 선택", "고혈압 생활 습관", "만성 기침 평가", "피부 건선 국소 치료", "수면 무호흡 진단",
                 "녹내장 안압 측정", "결핵 잠복 감염 치료", "대상포진 예방 접종", "류마티스 관절염 메토트렉세이트", "파킨슨병 레보도파",
                 "치매 인지 선별", "뇌전증 발작 약물", "다발경화증 재발 치료")
FILLER = [
    {**chunk(f"900-C{number:02d}", topic, f"{topic} 관련 일반 문장 {number}"), "review_status": "approved", "chapter": "900", "conflicts": []}
    for number, topic in enumerate(FILLER_TOPICS * 8, start=1)
]


def write_corpus(directory: Path, chunks: list[dict[str, object]], concepts: list[dict[str, object]] | None = None) -> None:
    chunks = [*chunks, *FILLER]
    (directory / "chunks.jsonl").write_text("\n".join(json.dumps(c, ensure_ascii=False) for c in chunks) + "\n{broken\n", encoding="utf-8")
    if concepts is not None:
        (directory / "drug_concepts.jsonl").write_text("\n".join(json.dumps(c, ensure_ascii=False) for c in concepts) + "\n", encoding="utf-8")


class EvidenceGateTest(unittest.TestCase):
    def test_only_commercially_usable_primary_claims_are_citable(self) -> None:
        citable = [c["chunk_id"] for c in CHUNKS if is_citable(c, include_unreviewed=True)]
        self.assertEqual(citable, ["369-C03-chunk", "369-C04-chunk"])
        approved_only = [c["chunk_id"] for c in CHUNKS if is_citable(c, include_unreviewed=False)]
        self.assertEqual(approved_only, ["369-C04-chunk"])

    def test_load_respects_unreviewed_switch_and_skips_broken_lines(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            write_corpus(Path(raw), CHUNKS)
            self.assertEqual(len(InMemoryEvidenceIndex.load(Path(raw), include_unreviewed=False)), 1 + len(FILLER))
            index = InMemoryEvidenceIndex.load(Path(raw), include_unreviewed=True)
            self.assertEqual((len(index), index.unreviewed_count), (2 + len(FILLER), 1))


class EvidenceSearchTest(unittest.TestCase):
    def setUp(self) -> None:
        self.raw = tempfile.TemporaryDirectory()
        self.addCleanup(self.raw.cleanup)
        concepts = [{"id": "cefazolin", "type": "ingredient", "ko": ["세파졸린"], "used_by": [{"claim_id": "369-C04"}], "used_by_class": []}]
        write_corpus(Path(self.raw.name), CHUNKS, concepts)
        self.index = InMemoryEvidenceIndex.load(Path(self.raw.name), include_unreviewed=True)

    def test_relevant_question_finds_chunk_and_generic_question_finds_none(self) -> None:
        self.assertEqual([c.chunk_id for c in self.index.search("급성 담낭염 수술 시기는?", 1)], ["369-C03-chunk"])
        self.assertEqual(self.index.search("환자 혈압 추이 요약해줘", 3), ())
        self.assertEqual(self.index.search("", 3), ())

    def test_concept_named_in_question_boosts_claims_that_use_it(self) -> None:
        # Text alone ranks the surgery-timing chunk first; naming a drug the antibiotic claim uses reorders them.
        self.assertEqual(self.index.search("급성 담낭염 담낭절제", 1)[0].chunk_id, "369-C03-chunk")
        self.assertEqual(self.index.search("급성 담낭염 담낭절제 세파졸린", 1)[0].chunk_id, "369-C04-chunk")
        # A concept name alone is only a booster, never enough to cite.
        self.assertEqual(self.index.search("세파졸린", 3), ())

    def test_concept_names_match_whole_latin_words_and_compact_hangul(self) -> None:
        index = InMemoryEvidenceIndex([], {"메트포르민": frozenset({"C1"}), "metformin": frozenset({"C1", "C2"}),
                                           "vitamin d": frozenset({"C3"})})
        self.assertEqual(index.matched_claims("메트 포르민과 Metformin, vitamin D"), {"C1": 2, "C2": 1, "C3": 1})
        self.assertEqual(index.matched_claims("metformins vitamin dx"), {})

    def test_payload_carries_review_status_sources_and_conflicts(self) -> None:
        found = self.index.search("급성 담낭염 수술 시기", 1)[0].as_dict()
        self.assertEqual(found["review_status"], "draft")
        self.assertNotIn("review_label", found)
        self.assertEqual(found["sources"], [{k: SOURCE_A[k] for k in ("id", "title", "locator", "url")}])
        self.assertTrue(found["conflicts"])


class StaticRetriever:
    def __init__(self, chunks=(), fail: bool = False) -> None:
        self.chunks, self.fail, self.calls = tuple(chunks), fail, 0

    def search(self, question: str, limit: int):
        self.calls += 1
        if self.fail:
            raise RuntimeError("index unavailable")
        return self.chunks[:limit]


class EvidenceChatTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        raw = tempfile.TemporaryDirectory()
        self.addCleanup(raw.cleanup)
        write_corpus(Path(raw.name), CHUNKS)
        self.found = InMemoryEvidenceIndex.load(Path(raw.name), include_unreviewed=True).search("급성 담낭염 수술 시기", 2)

    async def test_prompt_numbers_evidence_and_states_citation_rules(self) -> None:
        provider = RecordingProvider()
        service = ClinicalChatService(provider, evidence=StaticRetriever(self.found))
        with self.assertLogs("uvicorn.error", level="INFO") as logs:
            await service.answer_prepared_snapshot(question="급성 담낭염 수술 시기", snapshot=SNAPSHOT)
        prompt = provider.requests[0].input
        self.assertIn("[최신화 근거 문단", prompt)
        self.assertIn("[근거 n: 출처명]", prompt)
        self.assertIn('"n":1', prompt)
        self.assertIn('"sources":[{"title":', prompt)  # source titles reach the model so it can attribute
        # No review-state warnings in the answer: content is attributed to its sources instead.
        self.assertNotIn("미검수", prompt)
        self.assertNotIn("review_label", prompt)
        self.assertLess(prompt.index("[비식별 EMR 요약"), prompt.index("[최신화 근거 문단"))
        self.assertTrue(any("clinical_evidence_cited chunk_ids=369-C03-chunk" in line and "unreviewed=1" in line for line in logs.output))
        self.assertFalse(any("급성 담낭염" in line for line in logs.output))  # question text is never logged

    async def test_retriever_failure_does_not_block_answer(self) -> None:
        provider = RecordingProvider()
        service = ClinicalChatService(provider, evidence=StaticRetriever(fail=True))
        with self.assertLogs("uvicorn.error", level="WARNING"):
            answer = await service.answer_prepared_snapshot(question="질문", snapshot=SNAPSHOT)
        self.assertEqual(answer, "요약 답변")
        self.assertNotIn("[최신화 근거 문단", provider.requests[0].input)

    async def test_precomputed_evidence_is_not_searched_again(self) -> None:
        retriever = StaticRetriever(self.found)
        service = ClinicalChatService(RecordingProvider(), evidence=retriever)
        await service.answer_prepared_snapshot(question="질문", snapshot=SNAPSHOT, evidence=())
        self.assertEqual(retriever.calls, 0)


class EvidenceApiTest(unittest.TestCase):
    def setUp(self) -> None:
        raw = tempfile.TemporaryDirectory()
        self.addCleanup(raw.cleanup)
        write_corpus(Path(raw.name), CHUNKS)
        found = InMemoryEvidenceIndex.load(Path(raw.name), include_unreviewed=True).search("급성 담낭염 수술 시기", 2)
        self.provider = RecordingProvider(answer="조기 복강경 담낭절제를 고려합니다 [근거 1: WSES 지침].")
        self.client = TestClient(create_app(ClinicalChatService(self.provider, evidence=StaticRetriever(found)), RawEmrPayloadMapper()))
        payload = {"patient": {"patientId": "pid-1"}, "bloodSugarList": [{"measuredAt": "2026-09-22 10:30:00", "glucoseValue": 120}]}
        self.provider_answer_for_summary()
        self.client.post("/internal/v1/patient-context", json={"patient_id": "pid-1", "emr_payload": payload})

    def provider_answer_for_summary(self) -> None:
        original = self.provider.respond
        summary = json.dumps({"message": "요약", "sections": [], "guidance": "확인"}, ensure_ascii=False)

        async def respond(request):
            answer = await original(request)
            if request.output_schema is not None:
                from ax_g_ai.services.ports import ProviderAnswer
                return ProviderAnswer(summary)
            return answer

        self.provider.respond = respond  # type: ignore[method-assign]

    def test_chat_response_and_stream_meta_list_the_cited_evidence(self) -> None:
        body = {"patient_id": "pid-1", "question": "급성 담낭염 수술 시기"}
        response = self.client.post("/internal/v1/clinical-chat", json=body)
        self.assertEqual(response.status_code, 200)
        evidence = response.json()["evidence"]
        self.assertEqual([(e["n"], e["chunk_id"], e["review_status"]) for e in evidence],
                         [(1, "369-C03-chunk", "draft"), (2, "369-C04-chunk", "approved")])
        self.assertFalse(any("review_label" in e for e in evidence))
        self.assertEqual(evidence[0]["sources"][0]["url"], "https://example.org/wses")
        stream = self.client.post("/internal/v1/clinical-chat/stream", json=body)
        meta = json.loads(stream.text.split("\n")[1].removeprefix("data: "))
        self.assertEqual(meta["evidence"][0]["chunk_id"], "369-C03-chunk")


@unittest.skipUnless((REPO_CORPUS / "chunks.jsonl").is_file(), "generated corpus not present")
class RepositoryEvidenceTest(unittest.TestCase):
    def test_repository_corpus_is_gated_and_searchable(self) -> None:
        self.assertEqual(len(InMemoryEvidenceIndex.load(REPO_CORPUS, include_unreviewed=False)),
                         sum(1 for line in (REPO_CORPUS / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
                             if is_citable(json.loads(line), include_unreviewed=False)))
        index = InMemoryEvidenceIndex.load(REPO_CORPUS, include_unreviewed=True)
        self.assertGreater(len(index), 200)
        self.assertEqual(index.search("칼륨이 5.3인데 피네레논 시작해도 되나요?", 3)[0].claim_id, "335-C11")
        self.assertEqual(index.search("SOAP 작성해줘", 3), ())


if __name__ == "__main__":
    unittest.main()
