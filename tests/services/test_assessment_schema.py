from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from pydantic import ValidationError

from ax_g_ai.services.clinical_recommendation import ClinicalRecommender, RuleCorpus
from ax_g_ai.services.clinical_recommendation.schema import AssessmentPayload, LabInterpretation, RecommendationItem
from tests.test_clinical_recommendation_api import ScriptedProvider, client, payload

REPO_CORPUS = Path(__file__).resolve().parents[2] / "text_data" / "corpus"

ITEM = {
    "record_id": "W-1", "claim_id": "C-1", "kind": "workup", "status": "possible", "topic": "CKD", "chapter": "1",
    "matched": [], "missing": [{"kind": "value", "key": "eGFR", "label": "추정 사구체여과율"}], "review_status": "draft",
    "region": "KR", "region_label": "국내", "priority": "primary", "sources": [{"id": "S", "title": "T", "region": "KR"}],
    "evidence_refs": [1],
}


class SchemaTest(unittest.TestCase):
    def test_unknown_keys_and_values_fail_at_creation(self) -> None:
        with self.assertRaises(ValidationError):
            RecommendationItem.model_validate({**ITEM, "evidence_ref": [1]})  # misspelt key
        with self.assertRaises(ValidationError):
            RecommendationItem.model_validate({**ITEM, "status": "likely"})
        with self.assertRaises(ValidationError):
            RecommendationItem.model_validate({**ITEM, "missing": [{"kind": "value", "key": "eGFR"}]})  # label required

    def test_only_set_keys_in_their_order_and_explicit_nulls_stay(self) -> None:
        lab = {"record_id": "L", "claim_id": None, "lab": "Na", "name": "나트륨", "value": 131, "unit": None, "test_date": None,
               "band": "경도", "context": "저나트륨", "review_status": "draft", "region": None, "region_label": "미상",
               "priority": "primary", "evidence_refs": []}
        dumped = LabInterpretation.model_validate(lab).model_dump(mode="json")
        self.assertEqual(json.dumps(dumped, ensure_ascii=False), json.dumps(lab, ensure_ascii=False))  # 131 stays an int
        self.assertNotIn("superseded_by", dumped)
        reordered = dict(reversed(list(ITEM.items())))
        self.assertEqual(list(RecommendationItem.model_validate(reordered).model_dump()), list(reordered))


class ApiSchemaTest(unittest.TestCase):
    def test_http_and_sse_carry_the_same_assessment_json(self) -> None:
        api = client(ScriptedProvider())
        api.post("/internal/v1/patient-context", json=payload())
        body = {"patient_id": "pid-1", "clinical_note": "혈변"}
        http = api.post("/internal/v1/clinical-recommendations", json=body).json()["assessment"]
        meta_line = next(line for line in api.post("/internal/v1/clinical-recommendations/stream", json=body).text.splitlines()
                         if line.startswith("data:") and '"assessment"' in line)
        self.assertEqual(json.dumps(http, ensure_ascii=False), json.dumps(json.loads(meta_line[5:])["assessment"], ensure_ascii=False))

    def test_openapi_documents_the_assessment(self) -> None:
        schemas = client(ScriptedProvider()).get("/openapi.json").json()["components"]["schemas"]
        self.assertEqual(schemas["ClinicalRecommendationResponseBody"]["properties"]["assessment"]["$ref"],
                         "#/components/schemas/AssessmentPayload")
        self.assertEqual(schemas["RecommendationItem"]["properties"]["status"]["enum"], ["met", "possible"])


class SchemaDocumentTest(unittest.TestCase):
    def test_document_tables_match_the_schema(self) -> None:
        from ax_g_ai.services.clinical_recommendation import schema_doc

        document = schema_doc.DOC.read_text(encoding="utf-8")
        self.assertEqual(schema_doc.render(document), document,
                         "docs/assessment_schema.md is out of date: run python -m ax_g_ai.services.clinical_recommendation.schema_doc")


@unittest.skipUnless((REPO_CORPUS / "chunks.jsonl").is_file(), "generated corpus not present")
class RepositoryCorpusSchemaTest(unittest.TestCase):
    def test_every_rule_of_the_repository_corpus_produces_a_valid_payload(self) -> None:
        """Catches a corpus change that the payload schema does not describe (before it reaches the API)."""
        corpus = RuleCorpus.load(REPO_CORPUS, include_unreviewed=True)
        engine = ClinicalRecommender(corpus, doses=True, max_possible=10_000, max_dose_items=10_000)
        emr = {
            "calculationContext": {"age_years": 72, "sex": "female", "weight_kg": 58, "height_cm": 155},
            "diagnosisList": [{"diagnosisCode": code} for code in ("E11.9", "N18.3", "I10", "I50.9", "K74.6", "J44.9")],
            "medicineDtailList": [{"medCode": "X", "medName": "다이아벡스정500mg(메트포르민염산염)", "dosage": 500, "unit": "mg",
                                   "frequency": 3, "interval": "1일"}],
            "labResultList": [{"examCode": "2160-0", "value": 1.6, "unit": "mg/dL", "testDate": "2026-10-01"}],
        }
        drugs = sorted({target for record in corpus.rules["dose_rules.jsonl"]
                        if (target := record.get("ingredient") or record.get("ingredient_class"))})
        for variant in (emr, {**copy.deepcopy(emr), "calculationContext": {"age_years": 30, "sex": "female"}}):
            assessment = engine.assess(engine.prepare(variant), {key: True for key in engine.finding_vocabulary}, drugs)
            payload = assessment.payload
            self.assertIsInstance(payload, AssessmentPayload)
            self.assertGreater(len(payload.items) + len(payload.dose_assessments or ()), 50)


if __name__ == "__main__":
    unittest.main()
