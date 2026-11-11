from __future__ import annotations

import unittest
from pathlib import Path

from ax_g_ai.services.clinical_recommendation import ClinicalRecommender, RuleCorpus

REPO_CORPUS = Path(__file__).resolve().parents[2] / "text_data" / "corpus"

GATE = {"ai_tier": "A", "verification": "primary", "claim_status": "유지", "review_status": "draft",
        "sources": [{"id": "VADOD-CKD-2025", "ai_use": "A", "title": "VA/DoD CKD 2025", "url": "https://example.org",
                     "license": "Public domain (U.S. Government work)"}]}
TESTS = [
    {"id": "SCr", "section": "labs", "name_ko": "혈청 크레아티닌", "unit": "mg/dL", "range": [0.05, 30],
     "loinc": [{"code": "2160-0"}], "corpus_ids": ["SCr"], "aliases": []},
    {"id": "Na", "section": "labs", "name_ko": "나트륨", "unit": "mmol/L", "range": [80, 200],
     "loinc": [{"code": "2951-2"}], "corpus_ids": ["Na"], "aliases": []},
    {"id": "UACR", "section": "labs", "name_ko": "소변 알부민/크레아티닌 비", "unit": "mg/g", "corpus_ids": ["UACR"], "aliases": []},
    {"id": "Renal_ultrasound", "section": "exams", "kind": "imaging", "name_ko": "신장 초음파", "corpus_ids": ["Renal_ultrasound"], "aliases": []},
    {"id": "Colonoscopy", "section": "exams", "kind": "endoscopy", "name_ko": "대장내시경", "corpus_ids": ["Colonoscopy"], "aliases": []},
    {"id": "HBsAg", "section": "labs", "name_ko": "B형간염 표면항원", "corpus_ids": ["HBsAg"], "aliases": []},
    {"id": "K", "section": "labs", "name_ko": "칼륨", "unit": "mmol/L", "corpus_ids": ["K"], "aliases": []},
    {"id": "age", "section": "clinical", "name_ko": "나이", "corpus_ids": ["age"], "aliases": []},
]
DRUGS = [{"id": "lisinopril", "type": "ingredient", "ko": ["리시노프릴"], "rxnorm": [], "classes": ["RASi"], "corpus_ids": ["lisinopril"]}]
CLASSES = [{"id": "RASi", "exhaustive": True, "derivable": True, "corpus_ids": ["RASi"]}]


def rule(record_id: str, kind: str, **fields: object) -> dict[str, object]:
    return {**GATE, "id": record_id, "claim_id": record_id.rsplit("-", 1)[0], "chapter": "335", "kind": kind, "topic": record_id, **fields}


RULES = {
    "workup.jsonl": [
        rule("335-C03-K1", "workup", trigger={"any": [{"dx": "E11"}, {"dx": "I10"}]},
             tests=[{"lab": "UACR", "reason": "알부민뇨"}], strength="VA/DoD Rec 1 (Weak for)"),
        rule("334-C02-K1", "workup", trigger={"any": [{"finding": "new_decline_kidney_function"}, {"var": "SCr", "op": ">", "value": 3}]},
             tests=[{"exam": "Renal_ultrasound", "reason": "수신증 확인"}]),
        rule("352-C01-K1", "workup", trigger={"all": [{"var": "age", "op": ">=", "value": 50}, {"finding": "hematochezia"}]},
             tests=[{"exam": "Colonoscopy", "reason": "출혈 원인"}]),
        rule("362-C01-K1", "workup", trigger={"all": [{"var": "age", "op": ">=", "value": 18}, {"not": {"finding": "prior_hbv_triple_panel"}}]},
             tests=[{"lab": "HBsAg"}]),
        rule("404-C03-K2", "workup", trigger={"var": "Na", "op": "<", "value": 135}, tests=[{"lab": "Na"}]),
        rule("999-C01-K1", "workup", trigger={"dx": "E11"}, tests=[{"lab": "UACR"}],
             sources=[{"id": "KDIGO", "ai_use": "C"}]),  # non-commercial source: never loaded
        rule("998-C01-K1", "workup", trigger={"dx": "E11"}, tests=[{"lab": "UACR"}], claim_status="폐기"),
    ],
    "diagnosis.jsonl": [
        rule("335-C01-K1", "diagnosis", condition={"name": "만성 콩팥병", "kcd": ["N18"]},
             logic={"var": "SCr", "op": ">", "value": 2}),
        rule("335-C01-K9", "diagnosis", condition={"name": "텍스트 기준만", "kcd": ["N19"]}, criteria=["문장 기준"]),
    ],
    "referral.jsonl": [
        rule("335-C22-K1", "referral", to="신장내과", criteria=[{"var": "SCr", "op": ">", "value": 4}, {"dx": "Q61.2"}]),
    ],
    "monitoring.jsonl": [
        rule("335-C09-R1", "monitoring", after={"event": "start_or_uptitrate", "med_class": "RASi"}, check=["SCr", "K"],
             within_days=[14, 28], rules=[{"when": {"var": "SCr_rise_pct_4wk", "op": ">", "value": 30}, "action": "원인 평가"}]),
    ],
    "lab_thresholds.jsonl": [
        rule("404-C03-K1", "lab_threshold", lab="Na", unit="mmol/L", context="저나트륨혈증 중증도",
             bands=[{"label": "경도", "min": 130, "max": 134.999}, {"label": "중등도", "min": 125, "max": 129.999}]),
    ],
    "chunks.jsonl": [
        {**GATE, "chunk_id": "335-C03-chunk", "claim_id": "335-C03", "chapter": "335", "title_path": ["335장", "CKD 검사"], "text": "당뇨·고혈압 환자는 UACR과 eGFR로 CKD를 선별한다."},
        {**GATE, "chunk_id": "404-C03-chunk", "claim_id": "404-C03", "chapter": "404", "title_path": ["404장"], "text": "저나트륨혈증 평가."},
        {**GATE, "chunk_id": "335-C01-chunk", "claim_id": "335-C01", "chapter": "335", "title_path": ["335장"], "text": "CKD 정의."},
    ],
    "test_concepts.jsonl": TESTS, "drug_concepts.jsonl": DRUGS, "drug_classes.jsonl": CLASSES,
}


def recommender(*, include_unreviewed: bool = True, **options: int) -> ClinicalRecommender:
    return ClinicalRecommender(RuleCorpus.from_rows(RULES, include_unreviewed=include_unreviewed), **options)


def source(**extra: object) -> dict[str, object]:
    return {
        "patient": {"patientId": "pid-1"},
        "calculationContext": {"age_years": 67, "sex": "male"},
        "diagnosisList": [{"diagnosisCode": "E119", "kcdName": "2형 당뇨병"}],
        "labResultList": [
            {"examCode": "2160-0", "itemName": "Creatinine", "value": "2.4", "unit": "mg/dL", "testDate": "2026-09-04"},
            {"examCode": "2951-2", "itemName": "Na", "value": "131", "unit": "mmol/L", "testDate": "2026-09-04"},
        ],
        "medicineDtailList": [],
        **extra,
    }


def by_id(assessment) -> dict[str, dict]:
    return {item["record_id"]: item for item in assessment.items}


class RuleGateTest(unittest.TestCase):
    def test_draft_rules_need_the_unreviewed_switch_and_non_commercial_or_retired_never_load(self) -> None:
        reviewed_only = RuleCorpus.from_rows(RULES, include_unreviewed=False)
        with_drafts = RuleCorpus.from_rows(RULES, include_unreviewed=True)

        self.assertEqual(len(reviewed_only), 0)
        self.assertEqual(len(reviewed_only.chunks_by_claim), 0)
        ids = {record["id"] for records in with_drafts.rules.values() for record in records}
        self.assertNotIn("999-C01-K1", ids)
        self.assertNotIn("998-C01-K1", ids)
        self.assertEqual(with_drafts.skipped, 2)
        self.assertEqual(with_drafts.unreviewed_count, len(with_drafts))

    def test_attribution_only_drops_records_whose_source_has_no_license(self) -> None:
        unlicensed = {"sources": [{"id": "X", "ai_use": "A", "title": "제목만"}]}
        rows = {**RULES, "workup.jsonl": [*RULES["workup.jsonl"], rule("700-C01-K1", "workup", trigger={"dx": "E11"}, **unlicensed)]}
        self.assertIn("700-C01-K1", {r["id"] for r in RuleCorpus.from_rows(rows, include_unreviewed=True).rules["workup.jsonl"]})
        strict = RuleCorpus.from_rows(rows, include_unreviewed=True, require_license=True)
        self.assertNotIn("700-C01-K1", {r["id"] for r in strict.rules["workup.jsonl"]})
        self.assertEqual(len(strict), len(RuleCorpus.from_rows(RULES, include_unreviewed=True)))

    def test_repository_corpus_cites_only_sources_with_a_license_and_title(self) -> None:
        loose = RuleCorpus.load(REPO_CORPUS, include_unreviewed=True)
        strict = RuleCorpus.load(REPO_CORPUS, include_unreviewed=True, require_license=True)
        self.assertEqual(len(strict), len(loose))
        self.assertEqual(sum(map(len, strict.chunks_by_claim.values())), sum(map(len, loose.chunks_by_claim.values())))

    def test_repository_corpus_loads_rules_only_with_the_unreviewed_switch(self) -> None:
        self.assertEqual(len(RuleCorpus.load(REPO_CORPUS, include_unreviewed=False)), 0)
        drafts = RuleCorpus.load(REPO_CORPUS, include_unreviewed=True)
        self.assertGreater(len(drafts.rules["workup.jsonl"]), 50)
        # Dose rules pass the same gate but stay unused unless stage 3 is switched on.
        self.assertGreater(len(drafts.rules["dose_rules.jsonl"]), 50)
        off, on = ClinicalRecommender(drafts), ClinicalRecommender(drafts, doses=True)
        self.assertEqual((off.dose_rule_count, off.dose_drugs), (0, ()))
        self.assertIn("hla_b5801_positive", set(on.finding_vocabulary) - set(off.finding_vocabulary))
        self.assertIsNone(off.assess(off.prepare({})).dose_assessments)


class AssessmentTest(unittest.TestCase):
    def test_met_rules_carry_tests_matched_facts_sources_and_unreviewed_label(self) -> None:
        engine = recommender()
        items = by_id(engine.assess(engine.prepare(source())))

        ckd = items["335-C03-K1"]
        self.assertEqual(ckd["status"], "met")
        self.assertEqual(ckd["matched"], ["진단 기록 E11 (E11.9)"])
        self.assertEqual(ckd["tests"], [{"id": "UACR", "name": "소변 알부민/크레아티닌 비", "category": "검체검사(혈액·소변 등)", "reason": "알부민뇨"}])
        self.assertEqual(ckd["review_label"], "의료진 미검수")
        self.assertEqual(ckd["sources"], [{"id": "VADOD-CKD-2025", "title": "VA/DoD CKD 2025", "url": "https://example.org",
                                           "license": "Public domain (U.S. Government work)"}])
        self.assertEqual(ckd["strength"], "VA/DoD Rec 1 (Weak for)")

        self.assertEqual(items["404-C03-K2"]["matched"], ["나트륨 < 135 (환자 값 131)"])
        diagnosis = items["335-C01-K1"]
        self.assertEqual(diagnosis["diagnosis"], {"name": "만성 콩팥병", "kcd": ["N18"]})
        self.assertFalse(diagnosis["already_recorded"])
        self.assertNotIn("335-C01-K9", items)  # text-only criteria cannot be evaluated

    def test_possible_needs_a_patient_anchor_and_lists_missing_inputs(self) -> None:
        engine = recommender()
        items = by_id(engine.assess(engine.prepare(source())))

        # SCr 2.4 is not > 3, the finding is unknown: false OR unknown = unknown, but nothing anchors it.
        self.assertNotIn("334-C02-K1", items)
        # Age alone with an unknown symptom is too weak to report.
        self.assertNotIn("352-C01-K1", items)
        # Age alone is enough for screening whose only unknown is "already screened".
        hbv = items["362-C01-K1"]
        self.assertEqual(hbv["status"], "possible")
        self.assertEqual(hbv["missing"], [{"kind": "finding", "key": "prior_hbv_triple_panel", "label": "prior hbv triple panel"}])

    def test_clinician_findings_complete_rules_and_unknown_keys_are_reported(self) -> None:
        engine = recommender()
        assessment = engine.assess(engine.prepare(source()), {"hematochezia": True, "finding:prior_hbv_triple_panel": True, "made_up": True})

        items = by_id(assessment)
        colonoscopy = items["352-C01-K1"]
        self.assertEqual(colonoscopy["status"], "met")
        self.assertEqual(colonoscopy["tests"][0]["category"], "내시경검사")
        self.assertNotIn("362-C01-K1", items)  # already screened → false
        self.assertEqual(assessment.findings_used, {"hematochezia": True, "prior_hbv_triple_panel": True})
        self.assertEqual(assessment.unknown_findings, ("made_up",))

    def test_recorded_diagnosis_referral_and_monitoring_after_a_prescription(self) -> None:
        engine = recommender()
        emr = source(
            diagnosisList=[{"diagnosisCode": "N18.3"}, {"diagnosisCode": "Q61.2"}],
            medicineDtailList=[{"medCode": "X1", "medName": "제스트릴정(리시노프릴)"}],
        )
        items = by_id(engine.assess(engine.prepare(emr)))

        self.assertTrue(items["335-C01-K1"]["already_recorded"])
        self.assertEqual(items["335-C22-K1"]["referral_to"], "신장내과")
        self.assertEqual(items["335-C22-K1"]["status"], "met")
        monitoring = items["335-C09-R1"]
        self.assertEqual(monitoring["status"], "possible")
        self.assertEqual([test["id"] for test in monitoring["tests"]], ["SCr", "K"])
        self.assertEqual(monitoring["interval"], {"within_days": [14, 28]})
        self.assertEqual(monitoring["missing"][0]["key"], "start_or_uptitrate")

    def test_ranking_puts_met_before_possible_and_diagnosis_first(self) -> None:
        engine = recommender()
        items = engine.assess(engine.prepare(source())).items
        statuses = [item["status"] for item in items]
        self.assertEqual(statuses, sorted(statuses, key=lambda status: status != "met"))
        self.assertEqual(items[0]["kind"], "diagnosis")

    def test_evidence_is_numbered_once_per_passage_and_capped(self) -> None:
        engine = recommender(max_evidence=1)
        assessment = engine.assess(engine.prepare(source()))

        self.assertEqual([chunk.chunk_id for chunk in assessment.evidence], ["335-C01-chunk"])
        items = by_id(assessment)
        self.assertEqual(items["335-C01-K1"]["evidence_refs"], [1])
        self.assertEqual(items["335-C03-K1"]["evidence_refs"], [])
        self.assertEqual(assessment.lab_interpretations[0]["evidence_refs"], [])

    def test_lab_bands_and_derived_flags(self) -> None:
        engine = recommender()
        assessment = engine.assess(engine.prepare(source()))

        self.assertEqual(
            [(lab["lab"], lab["band"], lab["value"]) for lab in assessment.lab_interpretations], [("Na", "경도", 131.0)],
        )
        self.assertTrue(assessment.unreviewed)
        payload = assessment.as_dict()
        self.assertEqual(set(payload), {"items", "lab_interpretations", "findings_used", "unknown_findings", "mapping_issues", "unreviewed"})

    def test_empty_emr_gives_no_patient_specific_items(self) -> None:
        engine = recommender()
        assessment = engine.assess(engine.prepare({}))
        self.assertEqual(assessment.items, ())
        self.assertIn("hematochezia", engine.finding_vocabulary)
        self.assertNotIn("SCr", engine.finding_vocabulary)


if __name__ == "__main__":
    unittest.main()


class RepositorySupplementRulesTest(unittest.TestCase):
    """Supplement chapters (S01–S04): cardiology, respiratory, infection, cancer screening."""

    def engine(self) -> ClinicalRecommender:
        return ClinicalRecommender(RuleCorpus.load(REPO_CORPUS, include_unreviewed=True, require_license=True), max_possible=40)

    def test_blood_pressure_and_oxygen_from_the_emr_reach_cardio_and_copd_rules(self) -> None:
        engine = self.engine()
        emr = {
            "calculationContext": {"age_years": 70, "sex": "male"},
            "diagnosisList": [{"diagnosisCode": "J44.9"}],
            "bloodPressureList": [{"measuredAt": "2026-09-01 09:00:00", "sbp": 150, "dbp": 92},
                                  {"measuredAt": "2026-09-02 09:00:00", "sbp": None, "dbp": 88}],
            "oxygenSaturationList": [{"measuredAt": "2026-09-02 09:00:00", "spo2Value": 87}],
        }
        prepared = engine.prepare(emr)
        self.assertEqual(prepared.mapping.facts.values["SBP"], 150)
        self.assertEqual(prepared.mapping.facts.values["DBP"], 88)  # latest row; a null systolic is skipped
        self.assertFalse([issue for issue in prepared.mapping.issues if issue.code == "bloodPressureList"])
        items = {item["record_id"]: item for item in engine.assess(prepared).items}

        self.assertEqual(items["S01-C01-K1"]["diagnosis"]["kcd"], ["I10"])
        self.assertEqual(items["S01-C01-K2"]["status"], "met")
        self.assertEqual(items["S02-C02-K1"]["referral_to"], "가정 산소 평가(호흡기내과)")
        self.assertEqual(items["S01-C05-K1"]["tests"][0]["name"], "심전도")
        for item in items.values():
            self.assertTrue(all(source.get("license") for source in item["sources"]))

    def test_cancer_screening_follows_sex_age_and_clinician_findings(self) -> None:
        engine = self.engine()
        woman = engine.prepare({"calculationContext": {"age_years": 52, "sex": "female"}})
        items = {item["record_id"]: item for item in engine.assess(woman).items}
        self.assertEqual(items["S04-C01-K1"]["status"], "possible")  # unknown: already screened?
        self.assertEqual(items["S04-C03-K1"]["tests"][0]["category"], "영상검사")
        self.assertIn("S04-C04-K2", items)
        self.assertNotIn("S04-C02-K1", items)  # lung screening needs a smoking history

        smoker = engine.assess(woman, {"smoking_20_pack_years_current_or_quit_within_15y": True,
                                       "unfit_for_lung_surgery": False, "colorectal_screening_up_to_date": True})
        items = {item["record_id"]: item for item in smoker.items}
        self.assertEqual(items["S04-C02-K1"]["status"], "met")
        self.assertEqual(items["S04-C02-K1"]["tests"][0]["id"], "LDCT_chest")
        self.assertNotIn("S04-C01-K1", items)

        man = {item["record_id"] for item in engine.assess(engine.prepare({"calculationContext": {"age_years": 52, "sex": "male"}})).items}
        self.assertNotIn("S04-C03-K1", man)
        self.assertNotIn("S04-C04-K2", man)

    def test_infection_rules_use_pregnancy_and_tb_risk_findings(self) -> None:
        engine = self.engine()
        prepared = engine.prepare({"calculationContext": {"age_years": 30, "sex": "female"}})
        items = {item["record_id"]: item for item in engine.assess(prepared, {"pregnancy": True, "tb_contact": True}).items}
        self.assertEqual([test["id"] for test in items["S03-C03-K1"]["tests"]], ["Syphilis_serology", "Anti_HIV", "HBsAg", "Anti_HCV"])
        self.assertEqual(items["S03-C04-K1"]["tests"][0]["id"], "Latent_TB_screen")


class RegionPriorityTest(unittest.TestCase):
    """Domestic (KR) standards come first; superseded foreign ones are reference only."""

    def engine(self) -> ClinicalRecommender:
        return ClinicalRecommender(RuleCorpus.load(REPO_CORPUS, include_unreviewed=True, require_license=True), max_possible=40)

    def test_domestic_cancer_screening_supersedes_us_rules(self) -> None:
        engine = self.engine()
        woman = engine.prepare({"calculationContext": {"age_years": 56, "sex": "female"}})
        assessment = engine.assess(woman, {"smoking_30_pack_years_current": True, "smoking_20_pack_years_current_or_quit_within_15y": True,
                                           "unfit_for_lung_surgery": False})
        items = {item["record_id"]: item for item in assessment.items}

        for kr, us in (("S05-C03-K1", "S04-C01-K1"), ("S05-C04-K1", "S04-C03-K1"), ("S05-C05-K1", "S04-C04-K2"), ("S05-C06-K1", "S04-C02-K1")):
            self.assertEqual((items[kr]["priority"], items[kr]["region"]), ("primary", "KR"))
            self.assertEqual((items[us]["priority"], items[us]["region"]), ("reference", "US"))
            self.assertEqual(items[us]["superseded_by"], [kr.rsplit("-", 1)[0]])
        self.assertEqual(items["S05-C01-K1"]["tests"][0]["id"], "Upper_endoscopy")  # gastric cancer: domestic only
        priorities = [item["priority"] for item in assessment.items]
        self.assertEqual(priorities, sorted(priorities, key=lambda value: value != "primary"))
        self.assertEqual(items["S05-C03-K1"]["sources"][0]["license"], "Not protected (Korean Copyright Act Art. 7 — statutes·public notices)")

    def test_domestic_hypertension_threshold_and_health_screening_bands(self) -> None:
        engine = self.engine()
        prepared = engine.prepare({"calculationContext": {"age_years": 50, "sex": "male", "weight_kg": 80, "height_cm": 170},
                                   "diagnosisList": [],
                                   "bloodPressureList": [{"measuredAt": "2026-11-01 09:00:00", "sbp": 135, "dbp": 85}]})
        assessment = engine.assess(prepared)
        items = {item["record_id"]: item for item in assessment.items}
        self.assertNotIn("S05-C07-K1", items)          # 135/85 is not hypertension by the domestic 140/90 threshold
        self.assertEqual(items["S05-C07-K3"]["status"], "met")  # prehypertension → rule out masked hypertension
        self.assertEqual(items["S01-C01-K1"]["priority"], "reference")  # VA/DoD >130 still shown, as reference
        bands = {(lab["record_id"], lab["band"], lab["priority"]) for lab in assessment.lab_interpretations}
        self.assertIn(("S05-C09-K1", "정상B(경계)", "primary"), bands)
        self.assertIn(("S05-C09-K3", "정상B(경계)", "primary"), bands)       # BMI 27.7
        self.assertIn(("416-C01-K1", "비만 1단계", "reference"), bands)      # WHO Asian BMI, superseded
        first_reference = next(i for i, lab in enumerate(assessment.lab_interpretations) if lab["priority"] == "reference")
        self.assertTrue(all(lab["priority"] == "reference" for lab in assessment.lab_interpretations[first_reference:]))

    def test_evidence_search_puts_domestic_passages_first(self) -> None:
        from ax_g_ai.services.clinical_evidence import InMemoryEvidenceIndex

        index = InMemoryEvidenceIndex.load(REPO_CORPUS, include_unreviewed=True, require_license=True)
        found = index.search("대장암 검진 분변잠혈 대장내시경 연령", 5)
        ids = [chunk.claim_id for chunk in found]
        self.assertIn("S05-C03", ids)
        if "S04-C01" in ids:
            self.assertLess(ids.index("S05-C03"), ids.index("S04-C01"))
            self.assertEqual(index.chunk("S04-C01-chunk").as_dict()["priority"], "reference")
        self.assertEqual(index.chunk("S05-C03-chunk").as_dict()["region_label"], "국내")


class DomesticScreeningScheduleTest(unittest.TestCase):
    def engine(self) -> ClinicalRecommender:
        return ClinicalRecommender(RuleCorpus.load(REPO_CORPUS, include_unreviewed=True, require_license=True), max_possible=60)

    def items(self, age: int, sex: str, findings=None) -> dict[str, dict]:
        engine = self.engine()
        prepared = engine.prepare({"calculationContext": {"age_years": age, "sex": sex}, "diagnosisList": []})
        return {item["record_id"]: item for item in engine.assess(prepared, findings or {}).items}

    def test_national_checkup_items_by_age_and_sex(self) -> None:
        woman54 = self.items(54, "female")
        self.assertEqual(woman54["S05-C10-K4"]["tests"][0]["id"], "DXA")
        self.assertIn("S05-C10-K1", woman54)  # lipid panel, woman ≥40
        self.assertNotIn("S05-C10-K4", self.items(54, "male"))
        self.assertNotIn("S05-C10-K1", self.items(30, "female"))
        self.assertIn("S05-C10-K1", self.items(30, "male"))  # man ≥24
        man40 = self.items(40, "male")
        self.assertEqual([t["id"] for t in man40["S05-C10-K2"]["tests"]], ["HBsAg", "Anti_HBs"])
        self.assertEqual(man40["362-C01-K1"]["priority"], "reference")  # CDC adult HBV screening yields to domestic
        self.assertIn("S05-C10-K3", self.items(56, "female"))  # HCV antibody at 56

    def test_national_immunization_and_tb_workplace(self) -> None:
        elder = self.items(70, "male")
        self.assertEqual(elder["S05-C11-K1"]["tests"][0]["category"], "예방접종")
        self.assertEqual(elder["S05-C11-K2"]["tests"][0]["id"], "PPSV23_vaccination")
        self.assertIn("S05-C10-K6", elder)  # KDSQ-C ≥66
        nurse = self.items(30, "female", {"healthcare_facility_worker": True, "pregnancy": True})
        self.assertEqual(nurse["S05-C12-K1"]["status"], "met")
        self.assertEqual(nurse["S05-C11-K1"]["status"], "possible")  # pregnant: influenza unless vaccinated this season

    def test_screening_score_referrals(self) -> None:
        engine = self.engine()
        prepared = engine.prepare({"calculationContext": {"age_years": 45, "sex": "female"}, "diagnosisList": []})
        items = {item["record_id"]: item for item in engine.assess(prepared, {"phq9_item9_positive": True}).items}
        self.assertEqual(items["S05-C10-K11"]["referral_to"], "정신건강의학과 등 전문의(정신건강복지센터 상담 안내)")
