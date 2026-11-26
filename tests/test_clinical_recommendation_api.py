from __future__ import annotations

import json
import unittest

from fastapi.testclient import TestClient
from pydantic import ValidationError

from ax_g_ai.adapters.main_bridge import RawEmrPayloadMapper
from ax_g_ai.api import create_app
from ax_g_ai.services.clinical_chat import ClinicalChatService
from ax_g_ai.services.clinical_evidence import InMemoryEvidenceIndex
from ax_g_ai.services.corpus_snapshot import CorpusSnapshot, StaticCorpus
from ax_g_ai.services.patient_context_store import PatientContextStore
from ax_g_ai.services.ports import OpenAIRequest, ProviderOutputError
from tests.services.test_clinical_recommendation import RULES, recommender, source
from tests.sse import post_stream


def prompt_rules(prompt: str) -> dict:
    """The rule-assessment JSON the model sees (Korean data names, see prompts.data_labels)."""
    section = prompt.split("[말뭉치 규칙 평가 결과", 1)[1]
    return json.loads(next(line for line in section.split("\n") if line.startswith("{")))


class ScriptedProvider:
    """Answers the findings-extraction call with JSON and everything else with text."""

    def __init__(self, extraction: str = '{"present":["hematochezia"],"absent":["prior_hbv_triple_panel"]}') -> None:
        self.extraction = extraction
        self.requests: list[OpenAIRequest] = []
        self.output_types: dict[str, type] = {}

    async def respond_structured(self, request: OpenAIRequest, output_type):
        self.requests.append(request)
        self.output_types[request.operation] = output_type
        text = self.extraction
        try:
            return output_type.model_validate_json(text)
        except ValidationError as error:
            raise ProviderOutputError("invalid structured answer") from error

    async def stream(self, request: OpenAIRequest):
        self.requests.append(request)
        yield "추천 "
        yield "검사"


def payload() -> dict[str, object]:
    return {"patient_id": "pid-1", "emr_payload": {"data": source()}}


def client(provider: ScriptedProvider, *, rules: bool = True, general_knowledge: bool = False) -> TestClient:
    evidence = InMemoryEvidenceIndex.from_records(RULES["chunks.jsonl"], {}, include_unreviewed=True)
    service = ClinicalChatService(provider, allow_general_knowledge=general_knowledge)  # type: ignore[arg-type]
    corpus = StaticCorpus(CorpusSnapshot(evidence, recommender() if rules else None))
    return TestClient(create_app(service, RawEmrPayloadMapper(), PatientContextStore(), corpus=corpus))


class ClinicalRecommendationApiTest(unittest.TestCase):
    def test_recommendation_applies_note_findings_rules_without_review_warnings(self) -> None:
        provider = ScriptedProvider()
        api = client(provider)
        self.assertEqual(post_stream(api,"/internal/v1/patient-context", json=payload()).status_code, 200)

        response = post_stream(api,"/internal/v1/clinical-recommendations", json={
            "patient_id": "pid-1", "clinical_note": "3일 전부터 혈변, B형간염 검사 받은 적 없음", "findings": {"hematochezia": False},
        })

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["answer"], "추천 검사")
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
        self.assertEqual(extraction.operation, "finding_extraction")
        schema = provider.output_types["finding_extraction"].model_json_schema()
        self.assertIn("hematochezia", schema["properties"]["present"]["items"]["enum"])
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
        post_stream(api,"/internal/v1/patient-context", json=payload())
        post_stream(api,"/internal/v1/clinical-recommendations", json={"patient_id": "pid-1"})
        self.assertIn("[AI 일반 지식·출처 없음]", provider.requests[-1].input)

    def test_stream_sends_assessment_in_meta_then_text(self) -> None:
        api = client(ScriptedProvider())
        post_stream(api,"/internal/v1/patient-context", json=payload())

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
        post_stream(api,"/internal/v1/patient-context", json=payload())

        with self.assertLogs("uvicorn.error", level="WARNING"):
            response = post_stream(api,"/internal/v1/clinical-recommendations", json={"patient_id": "pid-1", "clinical_note": "혈변"})
        self.assertEqual(response.json()["assessment"]["findings_used"], {})

        listed = api.get("/internal/v1/clinical-recommendations/findings").json()
        self.assertIn("hematochezia", listed["findings"])
        self.assertEqual(listed["rules"], listed["unreviewed_rules"])

    def test_without_rules_the_endpoint_fails_closed_and_chat_is_unchanged(self) -> None:
        provider = ScriptedProvider()
        api = client(provider, rules=False)
        post_stream(api,"/internal/v1/patient-context", json=payload())

        response = post_stream(api,"/internal/v1/clinical-recommendations", json={"patient_id": "pid-1"})
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["code"], "RECOMMENDATION_UNAVAILABLE")

        post_stream(api,"/internal/v1/clinical-chat", json={"patient_id": "pid-1", "question": "추천 검사 알려줘"})
        self.assertNotIn("[말뭉치 규칙 평가 결과", provider.requests[-1].input)

    def test_chat_adds_rule_results_only_for_diagnosis_or_test_questions(self) -> None:
        provider = ScriptedProvider()
        api = client(provider)
        post_stream(api,"/internal/v1/patient-context", json=payload())

        post_stream(api,"/internal/v1/clinical-chat", json={"patient_id": "pid-1", "question": "최근 혈당 추이는?"})
        self.assertNotIn("[말뭉치 규칙 평가 결과", provider.requests[-1].input)

        response = post_stream(api,"/internal/v1/clinical-chat", json={"patient_id": "pid-1", "question": "추가로 필요한 검사는?"})
        self.assertIn("[말뭉치 규칙 평가 결과", provider.requests[-1].input)
        evidence = response.json()["evidence"]
        self.assertLessEqual(len([item for item in evidence if item["chunk_id"] in {"335-C01-chunk", "335-C03-chunk", "404-C03-chunk"}]), 4)
        assessment = prompt_rules(provider.requests[-1].input)
        refs = [ref for item in assessment["규칙 평가 항목"] for ref in item.get("근거 번호", ())]
        self.assertTrue(refs and max(refs) <= len(evidence))
        self.assertNotIn('"출처":', json.dumps(assessment, ensure_ascii=False))  # shown on screen, not repeated to the model


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
        api = TestClient(create_app(ClinicalChatService(provider), RawEmrPayloadMapper(), PatientContextStore(),  # type: ignore[arg-type]
                                    corpus=StaticCorpus(CorpusSnapshot(evidence, engine))))
        emr = source(medicineDtailList=[{"medCode": "111111111", "medName": "클래리정"}, {"medCode": "222222222", "medName": "심바정"}])
        post_stream(api,"/internal/v1/patient-context", json={"patient_id": "pid-1", "emr_payload": emr})

        body = post_stream(api,"/internal/v1/clinical-recommendations", json={"patient_id": "pid-1"}).json()
        alerts = body["assessment"]["medication_alerts"]
        self.assertEqual([a["type"] for a in alerts["alerts"]], ["병용금기"])
        self.assertEqual(alerts["source"]["license"], "KOGL Type 1 (공공누리 제1유형 — 출처표시)")
        self.assertIn('"DUR 점검 결과"', provider.requests[-1].input)
        self.assertNotIn("medication_alerts", provider.requests[-1].input)
        # The model does not judge DUR in the text; the server appends the fixed block instead.
        self.assertIn("병용·연령·임부·노인 금기에 해당하는지 직접 판단하지 마세요", provider.requests[-1].input)
        self.assertIn("\n\n[처방 점검 — 심평원 DUR 목록]\n• 병용금기: 클래리정 + 심바정", body["answer"])

        # A question about the DUR check alone is answered by the server's block, without the model or evidence.
        sent = len(provider.requests)
        body = post_stream(api,"/internal/v1/clinical-chat", json={"patient_id": "pid-1", "question": "현재 복용 약 병용금기 있나요?"}).json()
        self.assertEqual(len(provider.requests), sent)
        self.assertTrue(body["answer"].startswith("처방 점검 결과는 아래와 같습니다.\n\n[처방 점검 — 심평원 DUR 목록]\n• 병용금기: 클래리정 + 심바정"))
        self.assertIsNone(body.get("evidence"))
        # Plain wording without DUR terms is the same question.
        body = post_stream(api,"/internal/v1/clinical-chat", json={"patient_id": "pid-1", "question": "이 약들 같이 먹어도 괜찮아?"}).json()
        self.assertEqual(len(provider.requests), sent)
        self.assertIn("근육병증", body["answer"])

        post_stream(api,"/internal/v1/clinical-chat", json={"patient_id": "pid-1", "question": "현재 복용 약 병용금기가 있으면 이유도 설명해 주세요."})
        self.assertIn("병용금기", provider.requests[-1].input)
        self.assertIn("근육병증", provider.requests[-1].input)
        # A medication question carries the DUR result but not the unrelated diagnosis/test rule items.
        self.assertEqual(list(prompt_rules(provider.requests[-1].input)), ["DUR 점검 결과"])
        self.assertNotIn("진단·검사·협진·추적 규칙을", provider.requests[-1].input)

        # A question with no rule inputs (SOAP) still carries the DUR alerts, but not the unrelated rule items,
        # and the answer always ends with the fixed DUR block (JSON and streaming).
        body = post_stream(api,"/internal/v1/clinical-chat", json={"patient_id": "pid-1", "question": "SOAP 형식으로 정리해 주세요."}).json()
        rules = prompt_rules(provider.requests[-1].input)
        self.assertEqual(list(rules), ["DUR 점검 결과"])
        self.assertEqual([alert["유형"] for alert in rules["DUR 점검 결과"]["경고"]], ["병용금기"])
        self.assertIn("\n\n[처방 점검 — 심평원 DUR 목록]\n• 병용금기: ", body["answer"])
        self.assertIn("근육병증", body["answer"].split("[처방 점검", 1)[1])
        with api.stream("POST", "/internal/v1/clinical-chat/stream", json={"patient_id": "pid-1", "question": "SOAP 형식으로 정리해 주세요."}) as response:
            stream = response.read().decode()
        self.assertIn("[처방 점검 — 심평원 DUR 목록]", stream)
        self.assertLess(stream.index("[처방 점검"), stream.index("event: done"))

        # A drug named in the question is checked as one to add.
        post_stream(api,"/internal/v1/patient-context", json={"patient_id": "pid-1", "emr_payload": source(
            medicineDtailList=[{"medCode": "222222222", "medName": "심바정"}])})
        post_stream(api,"/internal/v1/clinical-chat", json={"patient_id": "pid-1", "question": "clarithromycin 추가해도 될까요?"})
        prompt = provider.requests[-1].input
        rules = prompt_rules(prompt)
        pair = next(alert for alert in rules["DUR 점검 결과"]["경고"] if alert["유형"] == "병용금기")
        self.assertEqual([product.get("추가 예정 약") for product in pair["제품"]], [True, None])
        self.assertEqual(rules["DUR 점검 결과"]["추가 예정 약"], [{"이름": "clarithromycin", "성분명": "clarithromycin"}])

        # Pregnancy stated in the recommendation question turns 임부금기 from possible (woman aged 30) into met.
        post_stream(api,"/internal/v1/patient-context", json={"patient_id": "pid-1", "emr_payload": source(
            calculationContext={"age_years": 30, "sex": "female"}, medicineDtailList=[{"medCode": "222222222", "medName": "심바정"}])})
        body = post_stream(api,"/internal/v1/clinical-recommendations", json={"patient_id": "pid-1", "question": "처방 점검"}).json()
        self.assertEqual([(a["type"], a["status"]) for a in body["assessment"]["medication_alerts"]["alerts"]], [("임부금기", "possible")])
        body = post_stream(api,"/internal/v1/clinical-recommendations", json={"patient_id": "pid-1", "question": "임신 10주 환자 처방 점검"}).json()
        self.assertEqual(body["assessment"]["findings_used"], {"pregnancy": True})
        self.assertEqual([(a["type"], a["status"]) for a in body["assessment"]["medication_alerts"]["alerts"]], [("임부금기", "met")])

        # Memo and question disagree without a change cue: neither is applied, 임부금기 asks to confirm.
        request = {"patient_id": "pid-1", "clinical_note": "임신 10주", "question": "임신 아님"}
        body = post_stream(api,"/internal/v1/clinical-recommendations", json=request).json()
        self.assertNotIn("pregnancy", body["assessment"]["findings_used"])
        self.assertEqual(body["assessment"]["finding_conflicts"], {"pregnancy": {"clinical_note": True, "question": False}})
        alert = body["assessment"]["medication_alerts"]["alerts"][0]
        self.assertEqual((alert["type"], alert["status"]), ("임부금기", "possible"))
        self.assertIn("서로 다름", alert["missing"][0]["label"])
        self.assertIn('"소견 충돌":{"임신":{"진료 메모":true,"질문":false}}', provider.requests[-1].input)
        # Explicit findings win, and there is nothing left to confirm.
        body = post_stream(api,"/internal/v1/clinical-recommendations", json={**request, "findings": {"pregnancy": False}}).json()
        self.assertFalse(body["assessment"]["findings_used"]["pregnancy"])
        self.assertNotIn("finding_conflicts", body["assessment"])
        self.assertEqual(body["assessment"]["medication_alerts"]["alerts"], [])
        # A change of status stated in the question overrides the older memo.
        body = post_stream(api,"/internal/v1/clinical-recommendations", json={**request, "question": "출산 후 2주, 처방 점검"}).json()
        self.assertFalse(body["assessment"]["findings_used"]["pregnancy"])
        self.assertNotIn("finding_conflicts", body["assessment"])
        body = post_stream(api,"/internal/v1/clinical-recommendations", json={
            "patient_id": "pid-1", "clinical_note": "임신 아님", "question": "이번에 임신 확인됨, 처방 점검"}).json()
        self.assertTrue(body["assessment"]["findings_used"]["pregnancy"])
        self.assertEqual([(a["type"], a["status"]) for a in body["assessment"]["medication_alerts"]["alerts"]], [("임부금기", "met")])


class DoseRecommendationApiTest(unittest.TestCase):
    def setUp(self) -> None:
        from tests.services.test_clinical_dosing import DOSE_CHUNKS, emr, engine

        self.provider = ScriptedProvider()
        evidence = InMemoryEvidenceIndex.from_records([*RULES["chunks.jsonl"], *DOSE_CHUNKS], {}, include_unreviewed=True)
        service = ClinicalChatService(self.provider)  # type: ignore[arg-type]
        self.api = TestClient(create_app(service, RawEmrPayloadMapper(), PatientContextStore(), corpus=StaticCorpus(CorpusSnapshot(evidence, engine()))))
        post_stream(self.api,"/internal/v1/patient-context", json={"patient_id": "pid-1", "emr_payload": emr()})

    def test_recommendation_returns_dose_results_and_the_prompt_carries_dose_rules(self) -> None:
        body = post_stream(self.api,"/internal/v1/clinical-recommendations",
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
        self.assertIn("'용량 결과'는 말뭉치 용량 규칙을", prompt)
        self.assertIn("`용량 참고`", prompt)
        self.assertIn("92.8", prompt)

    def test_findings_endpoint_lists_dose_drugs_and_bad_input_is_rejected(self) -> None:
        listing = self.api.get("/internal/v1/clinical-recommendations/findings").json()
        self.assertIn("pregabalin", listing["dose_drugs"])
        self.assertIn("basal_insulin", listing["dose_drugs"])
        too_many = post_stream(self.api,"/internal/v1/clinical-recommendations", json={"patient_id": "pid-1", "dose_drugs": ["x"] * 21})
        self.assertEqual(too_many.status_code, 400)

    def test_chat_dose_question_gets_dose_results(self) -> None:
        post_stream(self.api,"/internal/v1/clinical-chat", json={"patient_id": "pid-1", "question": "프레가발린 용량 괜찮나요?"})
        prompt = self.provider.requests[-1].input
        self.assertIn("'용량 결과'는 말뭉치 용량 규칙을", prompt)
        self.assertIn('"기준 1일 최대(mg)":75.0', prompt)
        # A dose question carries only the dose (and DUR) results, without the rule items or their evidence numbers.
        self.assertNotIn("규칙 평가 항목", prompt_rules(prompt))
        self.assertNotIn("근거 번호", json.dumps(prompt_rules(prompt), ensure_ascii=False))
        post_stream(self.api,"/internal/v1/clinical-chat", json={"patient_id": "pid-1", "question": "최근 혈당 추이는?"})
        self.assertNotIn("용량 결과", self.provider.requests[-1].input)
