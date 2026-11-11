from __future__ import annotations

import unittest
from pathlib import Path

from ax_g_ai.services.clinical_dosing import (
    DoseFormulaError, MissingInput, evaluate_formula, formula_variables, prescribed_daily_mg, range_label, select_band,
)
from ax_g_ai.services.clinical_recommendation import DOSE_NOTICE, ClinicalRecommender, RuleCorpus
from tests.services.test_clinical_recommendation import GATE, RULES, TESTS

REPO_CORPUS = Path(__file__).resolve().parents[2] / "text_data" / "corpus"

US = {"id": "OPENFDA-X", "ai_use": "A", "title": "openFDA label", "url": "https://example.org", "region": "US",
      "license": "Public domain (U.S. Government work)"}
KR = {"id": "KR-X", "ai_use": "A", "title": "국내 지침", "url": "https://example.kr", "region": "KR", "license": "KOGL Type 1"}


def dose(record_id: str, claim_id: str, **fields: object) -> dict[str, object]:
    return {**GATE, "sources": [US], "id": record_id, "claim_id": claim_id, "chapter": claim_id.split("-")[0],
            "kind": "dose_rule", "topic": record_id, **fields}


DOSE_RULES = [
    dose("DR-PREGABALIN", "419-C04", ingredient="pregabalin", indication="당뇨병성 말초신경병증 통증",
         renal_param="CrCl", renal_method="Cockcroft-Gault", standard_dose="50 mg 1일 3회 시작",
         bands=[{"range": {"min": 60}, "daily_mg": {"start": 150, "max": 300}},
                {"range": {"min": 30, "max": 59.999}, "daily_mg": {"start": 75, "max": 150}},
                {"range": {"min": 15, "max": 29.999}, "daily_mg": {"start": "25–50", "max": 75}, "alert": "caution"},
                {"range": {"max": 14.999}, "daily_mg": {"start": 25, "max": "25–50"}}],
         alert_note="미국 허가 기준 참고 범위"),
    dose("DR-LEVOTHYROXINE", "405-C01", ingredient="levothyroxine", indication="성인 갑상선기능저하증",
         calculation={"full_replacement_mcg_day": {"formula": "weight_kg × 1.6", "unit": "mcg/day"}},
         rules=[{"when": {"var": "age", "op": ">=", "value": 65}, "action": "낮게 시작", "alert": "caution"},
                {"when": {"finding": "central_hypothyroidism"}, "action": "유리 T4 기준"}],
         administration=["공복"]),
    dose("DR-ALTEPLASE", "446-C01", ingredient="alteplase",
         calculation={"total_mg": {"formula": "min(weight_kg × 0.9, 90)", "unit": "mg"},
                      "bolus_mg": {"formula": "total_mg × 0.1", "note": "1분간"}}),
    dose("DR-BASAL-INSULIN", "418-C14", ingredient_class="basal_insulin", sources=[KR],
         calculation={"start_dose_u_per_day": {"formula": "weight_kg × [0.1, 0.2]", "unit": "U/day"}}),
    dose("DR-REGIMEN-ONLY", "347-C09", regimens={"a": "x"}),  # no single drug: never anchored
]
DOSE_TESTS = [
    *TESTS,
    {"id": "CrCl", "section": "derived", "name_ko": "크레아티닌 청소율(Cockcroft-Gault)", "unit": "mL/min", "corpus_ids": ["CrCl"], "aliases": []},
    {"id": "eGFR", "section": "labs", "name_ko": "추정 사구체여과율", "unit": "mL/min/1.73m2", "corpus_ids": ["eGFR"], "aliases": []},
    {"id": "weight_kg", "section": "vitals", "name_ko": "체중", "unit": "kg", "corpus_ids": ["weight_kg"], "aliases": []},
]
DOSE_DRUGS = [
    *RULES["drug_concepts.jsonl"],
    {"id": "pregabalin", "type": "ingredient", "ko": ["프레가발린"], "rxnorm": [], "classes": [], "corpus_ids": ["pregabalin"]},
    {"id": "levothyroxine", "type": "ingredient", "ko": ["레보티록신"], "rxnorm": [], "classes": [], "corpus_ids": ["levothyroxine"]},
    {"id": "alteplase", "type": "ingredient", "ko": ["알테플라제"], "rxnorm": [], "classes": [], "corpus_ids": ["alteplase"]},
    {"id": "insulin_glargine", "type": "ingredient", "ko": ["인슐린글라진"], "rxnorm": [], "classes": ["basal_insulin"], "corpus_ids": ["insulin_glargine"]},
    {"id": "acetaminophen", "type": "ingredient", "ko": ["아세트아미노펜"], "rxnorm": [], "classes": [], "corpus_ids": ["acetaminophen"]},
]
DOSE_CLASSES = [*RULES["drug_classes.jsonl"], {"id": "basal_insulin", "exhaustive": True, "derivable": True, "corpus_ids": ["basal_insulin"]}]
DOSE_CHUNKS = [
    {**GATE, "chunk_id": f"{claim}-chunk", "claim_id": claim, "chapter": claim.split("-")[0], "title_path": [claim], "text": f"{claim} 용량 근거"}
    for claim in ("419-C04", "405-C01", "446-C01", "418-C14")
]
DOSE_ROWS = {**RULES, "dose_rules.jsonl": DOSE_RULES, "test_concepts.jsonl": DOSE_TESTS, "drug_concepts.jsonl": DOSE_DRUGS,
             "drug_classes.jsonl": DOSE_CLASSES, "chunks.jsonl": [*RULES["chunks.jsonl"], *DOSE_CHUNKS]}


def engine(**options) -> ClinicalRecommender:
    return ClinicalRecommender(RuleCorpus.from_rows(DOSE_ROWS, include_unreviewed=True), doses=True, **options)


def emr(*, age: float | None = 72, sex: str = "female", weight: float | None = 58, meds: list[dict] | None = None) -> dict:
    context = {"sex": sex, **({"age_years": age} if age is not None else {}), **({"weight_kg": weight} if weight is not None else {})}
    return {
        "patient": {"patientId": "pid-1"},
        "calculationContext": context,
        "diagnosisList": [{"diagnosisCode": "E11.9"}],
        "labResultList": [{"examCode": "2160-0", "itemName": "Creatinine", "value": "1.6", "unit": "mg/dL", "testDate": "2026-09-04"}],
        "medicineDtailList": meds if meds is not None else [
            {"medCode": "m1", "medName": "리리카캡슐(프레가발린)", "dosage": 150, "unit": "mg", "frequency": 2},
        ],
    }


def doses_by_id(assessment) -> dict[str, dict]:
    return {item["record_id"]: item for item in assessment.dose_assessments}


class FormulaTest(unittest.TestCase):
    def test_weight_formulas_caps_ranges_and_chained_outputs(self) -> None:
        self.assertEqual(evaluate_formula("min(weight_kg, 100) × 150", {"weight_kg": 70}), 10500)
        self.assertEqual(evaluate_formula("min(weight_kg, 100) × 150", {"weight_kg": 120}), 15000)
        self.assertEqual(evaluate_formula("min(weight_kg × 0.9, 90)", {"weight_kg": 110}), 90)
        self.assertEqual(evaluate_formula("weight_kg × [0.1, 0.2]", {"weight_kg": 70}), (7.0, 14.0))
        self.assertAlmostEqual(evaluate_formula("total_mg × 0.1", {"total_mg": 63.0}), 6.3)
        self.assertEqual(evaluate_formula("(4 - 2) / 4 + -1", {}), -0.5)
        self.assertEqual(formula_variables("min(weight_kg × 0.9, 90) + total_mg"), ["weight_kg", "total_mg"])

    def test_missing_inputs_and_unsupported_syntax_never_guess(self) -> None:
        with self.assertRaises(MissingInput) as caught:
            evaluate_formula("weight_kg × 1.6", {})
        self.assertEqual(caught.exception.name, "weight_kg")
        for expression in ("weight_kg ** 2", "__import__('os')", "exp(1)", "weight_kg × ", "1 / 0", "[1, [2, 3]]", "weight_kg 2"):
            with self.subTest(expression=expression), self.assertRaises(DoseFormulaError):
                evaluate_formula(expression, {"weight_kg": 70})

    def test_band_selection_and_korean_range_labels(self) -> None:
        bands = DOSE_RULES[0]["bands"]
        self.assertEqual(select_band(bands, 29.1)["daily_mg"]["max"], 75)
        self.assertEqual(select_band(bands, 60)["daily_mg"]["max"], 300)
        self.assertIsNone(select_band([{"range": {"min": 30}}], 10))
        self.assertEqual(range_label({"min": 30, "max": 49.999}, "mL/min"), "30 이상 50 미만 mL/min")
        self.assertEqual(range_label({"min": 60.001}), "60 초과")
        self.assertEqual(range_label({"min": 5, "max": 15}), "5 이상 15 이하")

    def test_daily_amount_only_from_explicit_mass_units_and_daily_schedules(self) -> None:
        self.assertEqual(prescribed_daily_mg({"dosage": 150, "unit": "mg", "frequency": 2}).mg_per_day, 300)
        self.assertEqual(prescribed_daily_mg({"dosage": "0.5", "unit": "g", "frequency": 3, "interval": "1일"}).mg_per_day, 1500)
        self.assertEqual(prescribed_daily_mg({"dosage": 100, "unit": "mcg", "frequency": 1}).mg_per_day, 0.1)
        for dose_fields in ({"dosage": 1, "unit": "정", "frequency": 2}, {"dosage": 15, "unit": "mg", "frequency": 1, "interval": "1주"},
                            {"dosage": 15, "unit": "mg"}, {"dosage": 0, "unit": "mg", "frequency": 1}, None):
            with self.subTest(dose=dose_fields):
                self.assertIsNone(prescribed_daily_mg(dose_fields))


class DoseAssessmentTest(unittest.TestCase):
    def test_prescribed_drug_gets_renal_band_and_daily_dose_check(self) -> None:
        r = engine()
        item = doses_by_id(r.assess(r.prepare(emr())))["DR-PREGABALIN"]
        self.assertEqual(item["anchor"], ["prescribed"])
        renal = item["renal"]
        self.assertEqual((renal["param"], renal["value"], renal["basis"]), ("CrCl", 29.1, "Cockcroft-Gault(실제 체중)"))
        self.assertEqual(renal["band"]["range"], "15 이상 30 미만 mL/min")
        check = item["prescriptions"][0]["check"]
        self.assertEqual((item["prescriptions"][0]["daily_mg"], check["limit_daily_mg"], check["status"]), (300, 75, "exceeds"))
        self.assertEqual(item["alert"], "high")
        self.assertEqual((item["review_label"], item["region"]), ("의료진 미검수", "US"))
        self.assertIn("국내 허가", item["notice"])
        self.assertEqual([entry["key"] for entry in item["reference"]], ["standard_dose"])
        self.assertTrue(item["evidence_refs"])

    def test_within_limit_and_missing_renal_inputs(self) -> None:
        r = engine()
        low = emr(meds=[{"medCode": "m1", "medName": "리리카캡슐(프레가발린)", "dosage": 25, "unit": "mg", "frequency": 2}])
        self.assertEqual(doses_by_id(r.assess(r.prepare(low)))["DR-PREGABALIN"]["prescriptions"][0]["check"]["status"], "within")
        no_weight = doses_by_id(r.assess(r.prepare(emr(weight=None))))["DR-PREGABALIN"]
        self.assertNotIn("band", no_weight["renal"])
        self.assertEqual([gap["key"] for gap in no_weight["missing"]], ["CrCl"])
        self.assertNotIn("check", no_weight["prescriptions"][0])  # no band, no comparison

    def test_requested_drugs_by_name_or_id_get_weight_calculations(self) -> None:
        r = engine()
        assessment = r.assess(r.prepare(emr()), dose_drugs=["레보티록신", "alteplase", "insulin_glargine", "아세트아미노펜", "모르는약"])
        items = doses_by_id(assessment)
        levo = items["DR-LEVOTHYROXINE"]
        self.assertEqual(levo["anchor"], ["requested"])
        self.assertEqual(levo["calculations"][0] | {}, {"name": "full_replacement_mcg_day", "formula": "weight_kg × 1.6", "unit": "mcg/day",
                                                      "inputs": {"weight_kg": 58.0}, "value": 92.8})
        self.assertEqual([(rule["status"], rule["action"]) for rule in levo["rules"]], [("met", "낮게 시작"), ("possible", "유리 T4 기준")])
        self.assertEqual(levo["missing"], [])  # rule-level unknowns stay on the rule
        alteplase = {calc["name"]: calc for calc in items["DR-ALTEPLASE"]["calculations"]}
        self.assertEqual((alteplase["total_mg"]["value"], alteplase["bolus_mg"]["value"], alteplase["bolus_mg"]["unit"]), (52.2, 5.2, "mg"))
        insulin = items["DR-BASAL-INSULIN"]
        self.assertEqual((insulin["drug"]["type"], insulin["calculations"][0]["range"]), ("class", [5.8, 11.6]))
        self.assertNotIn("notice", insulin)  # domestic source
        self.assertNotIn("DR-REGIMEN-ONLY", items)
        self.assertEqual({entry["input"]: entry["reason"] for entry in assessment.unmatched_dose_drugs},
                         {"아세트아미노펜": "말뭉치에 이 약의 용량 규칙 없음", "모르는약": "성분을 확인할 수 없음"})
        payload = assessment.as_dict()
        self.assertEqual(payload["dose_notice"], DOSE_NOTICE)
        self.assertTrue(payload["unreviewed"])

    def test_prescribed_class_member_anchors_class_rules(self) -> None:
        r = engine()
        meds = [{"medCode": "g1", "medName": "란투스주(인슐린글라진)", "dosage": 10, "unit": "U", "frequency": 1}]
        item = doses_by_id(r.assess(r.prepare(emr(meds=meds))))["DR-BASAL-INSULIN"]
        self.assertEqual((item["anchor"], item["prescriptions"][0]["med_code"]), (["prescribed"], "g1"))
        self.assertNotIn("daily_mg", item["prescriptions"][0])  # units are not a mass

    def test_children_and_unknown_age_get_no_adult_calculations(self) -> None:
        r = engine()
        child = doses_by_id(r.assess(r.prepare(emr(age=12)), dose_drugs=["levothyroxine"]))["DR-LEVOTHYROXINE"]
        self.assertNotIn("calculations", child)
        self.assertIn("18세 미만", child["not_applied"])
        unknown = doses_by_id(r.assess(r.prepare(emr(age=None)), dose_drugs=["levothyroxine"]))["DR-LEVOTHYROXINE"]
        self.assertNotIn("calculations", unknown)
        self.assertEqual([gap["key"] for gap in unknown["missing"]], ["age"])

    def test_weight_missing_is_reported_not_guessed(self) -> None:
        r = engine()
        item = doses_by_id(r.assess(r.prepare(emr(weight=None)), dose_drugs=["levothyroxine"]))["DR-LEVOTHYROXINE"]
        calc = item["calculations"][0]
        self.assertNotIn("value", calc)
        self.assertEqual(calc["missing"][0]["key"], "weight_kg")
        self.assertIn("weight_kg", [gap["key"] for gap in item["missing"]])

    def test_dose_evidence_survives_a_full_stage_two_evidence_cap(self) -> None:
        r = engine(max_evidence=1)
        assessment = r.assess(r.prepare(emr()))
        self.assertEqual(len([ref for item in assessment.items for ref in item["evidence_refs"]]) <= 1, True)
        self.assertTrue(doses_by_id(assessment)["DR-PREGABALIN"]["evidence_refs"])

    def test_dose_stage_off_by_default(self) -> None:
        plain = ClinicalRecommender(RuleCorpus.from_rows(DOSE_ROWS, include_unreviewed=True))
        assessment = plain.assess(plain.prepare(emr()), dose_drugs=["levothyroxine"])
        self.assertIsNone(assessment.dose_assessments)
        self.assertNotIn("dose_assessments", assessment.as_dict())
        reviewed_only = ClinicalRecommender(RuleCorpus.from_rows(DOSE_ROWS, include_unreviewed=False), doses=True)
        self.assertEqual(reviewed_only.dose_rule_count, 0)  # draft dose rules need the unreviewed switch


class RepositoryDoseTest(unittest.TestCase):
    def test_real_corpus_renal_bands_and_weight_formulas(self) -> None:
        corpus = RuleCorpus.load(REPO_CORPUS, include_unreviewed=True, require_license=True)
        r = ClinicalRecommender(corpus, doses=True)
        source = emr(meds=[
            {"medCode": "1", "medName": "리리카캡슐150밀리그램(프레가발린)", "dosage": 150, "unit": "mg", "frequency": 2},
            {"medCode": "2", "medName": "다이아벡스정500밀리그램(메트포르민염산염)", "dosage": 500, "unit": "mg", "frequency": 2},
        ])
        items = doses_by_id(r.assess(r.prepare(source), dose_drugs=["levothyroxine", "alteplase"]))
        pregabalin = items["DR-PREGABALIN-DPN"]
        self.assertEqual((pregabalin["renal"]["value"], pregabalin["prescriptions"][0]["check"]["status"]), (29.1, "exceeds"))
        metformin = items["DR-METFORMIN"]
        self.assertEqual((metformin["renal"]["param"], metformin["renal"]["value"], metformin["renal"]["band"]["alert"]), ("eGFR", 34.1, "caution"))
        self.assertEqual(items["DR-LEVOTHYROXINE"]["calculations"][0]["value"], 92.8)
        self.assertEqual({c["name"]: c["value"] for c in items["DR-ALTEPLASE-AIS"]["calculations"]},
                         {"total_mg": 52.2, "bolus_mg": 5.2, "infusion_mg": 47.0})
        self.assertTrue(all(item["evidence_refs"] for item in items.values()))

    def test_every_repository_dose_formula_parses(self) -> None:
        corpus = RuleCorpus.load(REPO_CORPUS, include_unreviewed=True)
        variables = {"weight_kg": 70.0}
        for record in corpus.rules["dose_rules.jsonl"]:
            for name, spec in (record.get("calculation") or {}).items():
                with self.subTest(record=record["id"], name=name):
                    variables[name] = evaluate_formula(spec["formula"], variables)


if __name__ == "__main__":
    unittest.main()
