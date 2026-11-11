from __future__ import annotations

import json
import unittest

from fastapi.testclient import TestClient

from ax_g_ai.adapters.main_bridge import RawEmrPayloadMapper
from ax_g_ai.adapters.openai import OpenAIRequest, ProviderAnswer
from ax_g_ai.api import create_app
from ax_g_ai.services.clinical_chat import ClinicalChatService
from ax_g_ai.services.clinical_evidence import InMemoryEvidenceIndex
from ax_g_ai.services.patient_context_store import PatientContextStore
from tests.services.test_clinical_recommendation import RULES, recommender, source


class ScriptedProvider:
    """Answers the findings-extraction call with JSON and everything else with text."""

    def __init__(self, extraction: str = '{"present":["hematochezia"],"absent":["prior_hbv_triple_panel"]}') -> None:
        self.extraction = extraction
        self.requests: list[OpenAIRequest] = []

    async def respond(self, request: OpenAIRequest) -> ProviderAnswer:
        self.requests.append(request)
        if request.output_schema_name == "clinical_findings":
            return ProviderAnswer(self.extraction)
        if request.output_schema is not None:
            return ProviderAnswer('{"message":"로드됨","sections":[],"guidance":"확인"}')
        return ProviderAnswer("의심 진단: 만성 콩팥병 가능성 [근거 1: VA/DoD CKD 2025]")

    async def stream(self, request: OpenAIRequest):
        self.requests.append(request)
        yield "추천 "
        yield "검사"


def payload() -> dict[str, object]:
    return {"patient_id": "pid-1", "emr_payload": {"data": source()}}


def client(provider: ScriptedProvider, *, rules: bool = True, general_knowledge: bool = False) -> TestClient:
    evidence = InMemoryEvidenceIndex.from_records(RULES["chunks.jsonl"], {}, include_unreviewed=True)
    service = ClinicalChatService(provider, evidence=evidence, allow_general_knowledge=general_knowledge)  # type: ignore[arg-type]
    return TestClient(create_app(service, RawEmrPayloadMapper(), PatientContextStore(), recommender() if rules else None))


class ClinicalRecommendationApiTest(unittest.TestCase):
    def test_recommendation_applies_note_findings_rules_without_review_warnings(self) -> None:
        provider = ScriptedProvider()
        api = client(provider)
        self.assertEqual(api.post("/internal/v1/patient-context", json=payload()).status_code, 200)

        response = api.post("/internal/v1/clinical-recommendations", json={
            "patient_id": "pid-1", "clinical_note": "3일 전부터 혈변, B형간염 검사 받은 적 없음", "findings": {"hematochezia": False},
        })

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["answer"], "의심 진단: 만성 콩팥병 가능성 [근거 1: VA/DoD CKD 2025]")
        self.assertNotIn("review_notice", body)  # no repeated warning: reference use is the service's stated purpose
        assessment = body["assessment"]
        # Explicit findings override the note extraction.
        self.assertEqual(assessment["findings_used"], {"hematochezia": False, "prior_hbv_triple_panel": False})
        ids = {item["record_id"]: item for item in assessment["items"]}
        self.assertEqual(ids["362-C01-K1"]["status"], "met")
        self.assertNotIn("352-C01-K1", ids)
        self.assertTrue(all(item["review_status"] == "draft" and "review_label" not in item for item in assessment["items"]))
        self.assertEqual(body["evidence"][0]["n"], 1)
        self.assertNotIn("review_label", body["evidence"][0])

        extraction, answer = provider.requests[-2], provider.requests[-1]
        self.assertIn("hematochezia", extraction.output_schema["properties"]["present"]["items"]["enum"])  # type: ignore[index]
        self.assertIn("[말뭉치 규칙 평가 결과", answer.input)
        # Attribution-only by default: no unsourced suggestions.
        self.assertNotIn("[AI 일반 지식·출처 없음]", answer.input)
        self.assertIn("출처를 밝힐 수 없는 일반 의학 지식으로", answer.input)
        self.assertEqual(body["evidence"][0]["sources"][0]["license"], "Public domain (U.S. Government work)")
        self.assertEqual(ids["362-C01-K1"]["sources"][0]["license"], "Public domain (U.S. Government work)")
        self.assertIn("혈변", answer.input)
        self.assertNotIn("pid-1", answer.input)

    def test_general_knowledge_suggestions_only_when_attribution_only_is_off(self) -> None:
        provider = ScriptedProvider()
        api = client(provider, general_knowledge=True)
        api.post("/internal/v1/patient-context", json=payload())
        api.post("/internal/v1/clinical-recommendations", json={"patient_id": "pid-1"})
        self.assertIn("[AI 일반 지식·출처 없음]", provider.requests[-1].input)

    def test_stream_sends_assessment_in_meta_then_text(self) -> None:
        api = client(ScriptedProvider())
        api.post("/internal/v1/patient-context", json=payload())

        response = api.post("/internal/v1/clinical-recommendations/stream", json={"patient_id": "pid-1"})

        self.assertEqual(response.status_code, 200)
        events = [block for block in response.text.split("\n\n") if block]
        meta = json.loads(events[0].split("data: ", 1)[1])
        self.assertEqual(events[0].split("\n")[0], "event: meta")
        self.assertIn("items", meta["assessment"])
        self.assertNotIn("review_notice", meta)
        self.assertTrue(events[-1].startswith("event: done"))

    def test_invalid_extraction_is_ignored_and_findings_endpoint_lists_vocabulary(self) -> None:
        api = client(ScriptedProvider(extraction="not json"))
        api.post("/internal/v1/patient-context", json=payload())

        with self.assertLogs("uvicorn.error", level="WARNING"):
            response = api.post("/internal/v1/clinical-recommendations", json={"patient_id": "pid-1", "clinical_note": "혈변"})
        self.assertEqual(response.json()["assessment"]["findings_used"], {})

        listed = api.get("/internal/v1/clinical-recommendations/findings").json()
        self.assertIn("hematochezia", listed["findings"])
        self.assertEqual(listed["rules"], listed["unreviewed_rules"])

    def test_without_rules_the_endpoint_fails_closed_and_chat_is_unchanged(self) -> None:
        provider = ScriptedProvider()
        api = client(provider, rules=False)
        api.post("/internal/v1/patient-context", json=payload())

        response = api.post("/internal/v1/clinical-recommendations", json={"patient_id": "pid-1"})
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["code"], "RECOMMENDATION_UNAVAILABLE")

        api.post("/internal/v1/clinical-chat", json={"patient_id": "pid-1", "question": "추천 검사 알려줘"})
        self.assertNotIn("[말뭉치 규칙 평가 결과", provider.requests[-1].input)

    def test_chat_adds_rule_results_only_for_diagnosis_or_test_questions(self) -> None:
        provider = ScriptedProvider()
        api = client(provider)
        api.post("/internal/v1/patient-context", json=payload())

        api.post("/internal/v1/clinical-chat", json={"patient_id": "pid-1", "question": "최근 혈당 추이는?"})
        self.assertNotIn("[말뭉치 규칙 평가 결과", provider.requests[-1].input)

        response = api.post("/internal/v1/clinical-chat", json={"patient_id": "pid-1", "question": "추가로 필요한 검사는?"})
        self.assertIn("[말뭉치 규칙 평가 결과", provider.requests[-1].input)
        evidence = response.json()["evidence"]
        self.assertLessEqual(len([item for item in evidence if item["chunk_id"] in {"335-C01-chunk", "335-C03-chunk", "404-C03-chunk"}]), 4)
        rule_section = provider.requests[-1].input.split("[말뭉치 규칙 평가 결과", 1)[1]
        assessment = json.loads(rule_section.split("\n")[2])
        refs = [ref for item in assessment["items"] for ref in item.get("evidence_refs", ())]
        self.assertTrue(refs and max(refs) <= len(evidence))
        self.assertNotIn("sources", rule_section.split("\n")[2])  # shown on screen, not repeated to the model


if __name__ == "__main__":
    unittest.main()


class DurRecommendationApiTest(unittest.TestCase):
    def test_recommendation_and_medication_chat_include_dur_alerts(self) -> None:
        import shutil

        from ax_g_ai.services.clinical_recommendation import ClinicalRecommender, RuleCorpus
        from tests.services.test_dur_check import checker_from_snapshot, fixture_snapshot

        snapshot_dir = fixture_snapshot()
        self.addCleanup(shutil.rmtree, snapshot_dir.parent)
        engine = ClinicalRecommender(RuleCorpus.from_rows(RULES, include_unreviewed=True), dur=checker_from_snapshot(snapshot_dir))
        provider = ScriptedProvider()
        evidence = InMemoryEvidenceIndex.from_records(RULES["chunks.jsonl"], {}, include_unreviewed=True)
        api = TestClient(create_app(ClinicalChatService(provider, evidence=evidence), RawEmrPayloadMapper(), PatientContextStore(), engine))  # type: ignore[arg-type]
        emr = source(medicineDtailList=[{"medCode": "111111111", "medName": "클래리정"}, {"medCode": "222222222", "medName": "심바정"}])
        api.post("/internal/v1/patient-context", json={"patient_id": "pid-1", "emr_payload": emr})

        body = api.post("/internal/v1/clinical-recommendations", json={"patient_id": "pid-1"}).json()
        alerts = body["assessment"]["medication_alerts"]
        self.assertEqual([a["type"] for a in alerts["alerts"]], ["병용금기"])
        self.assertEqual(alerts["source"]["license"], "KOGL Type 1 (공공누리 제1유형 — 출처표시)")
        self.assertIn("medication_alerts", provider.requests[-1].input)

        api.post("/internal/v1/clinical-chat", json={"patient_id": "pid-1", "question": "현재 복용 약 병용금기 있나요?"})
        self.assertIn("병용금기", provider.requests[-1].input)
        self.assertIn("근육병증", provider.requests[-1].input)


class DoseRecommendationApiTest(unittest.TestCase):
    def setUp(self) -> None:
        from tests.services.test_clinical_dosing import DOSE_CHUNKS, emr, engine

        self.provider = ScriptedProvider()
        evidence = InMemoryEvidenceIndex.from_records([*RULES["chunks.jsonl"], *DOSE_CHUNKS], {}, include_unreviewed=True)
        service = ClinicalChatService(self.provider, evidence=evidence)  # type: ignore[arg-type]
        self.api = TestClient(create_app(service, RawEmrPayloadMapper(), PatientContextStore(), engine()))
        self.api.post("/internal/v1/patient-context", json={"patient_id": "pid-1", "emr_payload": emr()})

    def test_recommendation_returns_dose_results_and_the_prompt_carries_dose_rules(self) -> None:
        body = self.api.post("/internal/v1/clinical-recommendations",
                             json={"patient_id": "pid-1", "dose_drugs": ["레보티록신"]}).json()
        doses = {item["record_id"]: item for item in body["assessment"]["dose_assessments"]}
        self.assertEqual(set(doses), {"DR-PREGABALIN", "DR-LEVOTHYROXINE"})
        self.assertEqual(doses["DR-PREGABALIN"]["prescriptions"][0]["check"]["status"], "exceeds")
        for key in ("dose_notice", "unreviewed"):
            self.assertNotIn(key, body["assessment"])
        self.assertNotIn("review_notice", body)
        self.assertNotIn("notice", doses["DR-PREGABALIN"])
        refs = [ref for item in doses.values() for ref in item["evidence_refs"]]
        self.assertTrue(refs and max(refs) <= len(body["evidence"]))
        prompt = self.provider.requests[-1].input
        self.assertIn("dose_assessments는 말뭉치 용량 규칙을", prompt)
        self.assertIn("`용량 참고`", prompt)
        self.assertIn("92.8", prompt)

    def test_findings_endpoint_lists_dose_drugs_and_bad_input_is_rejected(self) -> None:
        listing = self.api.get("/internal/v1/clinical-recommendations/findings").json()
        self.assertIn("pregabalin", listing["dose_drugs"])
        self.assertIn("basal_insulin", listing["dose_drugs"])
        too_many = self.api.post("/internal/v1/clinical-recommendations", json={"patient_id": "pid-1", "dose_drugs": ["x"] * 21})
        self.assertEqual(too_many.status_code, 400)

    def test_chat_dose_question_gets_dose_results(self) -> None:
        self.api.post("/internal/v1/clinical-chat", json={"patient_id": "pid-1", "question": "프레가발린 용량 괜찮나요?"})
        prompt = self.provider.requests[-1].input
        self.assertIn("dose_assessments는 말뭉치 용량 규칙을", prompt)
        self.assertIn("limit_daily_mg", prompt)
        self.api.post("/internal/v1/clinical-chat", json={"patient_id": "pid-1", "question": "최근 혈당 추이는?"})
        self.assertNotIn("dose_assessments", self.provider.requests[-1].input)
