from __future__ import annotations

import unittest
from pathlib import Path

from ax_g_ai.services.clinical_corpus import PatientFacts, evaluate
from ax_g_ai.services.emr_mapping import ConceptCatalog, ConceptMapError, SiteCodeMap, map_emr, normalize_kcd

REPO_CORPUS = Path(__file__).resolve().parents[2] / "text_data" / "corpus"

TESTS = [
    {"id": "SCr", "section": "labs", "unit": "mg/dL", "convert": {"umol/L": 0.01131}, "range": [0.05, 30],
     "loinc": [{"code": "2160-0", "name": "Creatinine"}], "corpus_ids": ["SCr"], "aliases": ["Creatinine", "Cr"]},
    {"id": "eGFR", "section": "labs", "unit": "mL/min/1.73m2", "range": [1, 200], "loinc": [], "corpus_ids": ["eGFR"], "aliases": []},
    {"id": "CrCl", "section": "derived", "unit": "mL/min", "loinc": [], "corpus_ids": ["CrCl"], "aliases": []},
    {"id": "HbA1c", "section": "labs", "unit": "%", "convert": {"mmol/mol": {"factor": 0.09148, "offset": 2.152}},
     "range": [3, 20], "loinc": [], "corpus_ids": ["HbA1c"], "aliases": ["HbA1c"]},
    {"id": "Glucose", "section": "labs", "unit": "mg/dL", "range": [10, 2000], "loinc": [], "corpus_ids": ["Glucose"], "aliases": []},
    {"id": "Triglycerides", "section": "labs", "unit": "mg/dL", "loinc": [], "corpus_ids": ["TG", "Triglycerides"], "aliases": ["TG"]},
    {"id": "weight_kg", "section": "vitals", "unit": "kg", "corpus_ids": ["weight_kg"], "aliases": []},
    {"id": "CBC", "section": "panels", "corpus_ids": ["CBC"], "aliases": ["CBC"], "members": []},
]
DRUGS = [
    {"id": "metformin", "type": "ingredient", "ko": ["메트포르민"], "rxnorm": [{"name": "metformin"}], "classes": [], "corpus_ids": ["metformin"]},
    {"id": "sitagliptin", "type": "ingredient", "ko": ["시타글립틴"], "rxnorm": [], "classes": ["DPP4i"], "corpus_ids": ["sitagliptin"]},
    {"id": "atorvastatin", "type": "ingredient", "ko": ["아토르바스타틴"], "rxnorm": [], "classes": ["statin", "lipid_lowering"], "corpus_ids": ["atorvastatin"]},
    {"id": "prednisolone", "type": "ingredient", "ko": ["프레드니솔론"], "rxnorm": [], "classes": ["systemic_glucocorticoid"], "corpus_ids": ["prednisolone"]},
    {"id": "methylprednisolone", "type": "ingredient", "ko": ["메틸프레드니솔론"], "rxnorm": [], "classes": ["systemic_glucocorticoid"], "corpus_ids": ["methylprednisolone"]},
    {"id": "ibuprofen", "type": "ingredient", "ko": ["이부프로펜"], "rxnorm": [], "classes": ["NSAID", "oral_NSAID", "topical_NSAID"], "corpus_ids": ["ibuprofen"]},
    {"id": "lithium", "type": "ingredient", "ko": ["탄산리튬", "리튬"], "rxnorm": [], "classes": [], "corpus_ids": ["lithium"]},
    {"id": "buprenorphine", "type": "ingredient", "ko": ["부프레노르핀"], "rxnorm": [], "classes": ["opioid", "long_term_opioid"], "corpus_ids": ["buprenorphine"]},
    {"id": "naloxone", "type": "ingredient", "ko": ["날록손"], "rxnorm": [], "classes": [], "corpus_ids": ["naloxone"]},
    {"id": "buprenorphine_naloxone", "type": "product", "components": ["buprenorphine", "naloxone"], "classes": ["opioid"], "corpus_ids": ["buprenorphine_naloxone"]},
]
CLASSES = [
    {"id": "DPP4i", "exhaustive": True, "derivable": True, "corpus_ids": ["DPP4i"]},
    {"id": "statin", "exhaustive": True, "derivable": True, "corpus_ids": ["statin"]},
    {"id": "lipid_lowering", "exhaustive": False, "derivable": True, "corpus_ids": ["lipid_lowering"]},
    {"id": "systemic_glucocorticoid", "exhaustive": True, "derivable": True, "systemic_only": True, "corpus_ids": ["systemic_glucocorticoid"]},
    {"id": "NSAID", "exhaustive": False, "derivable": True, "systemic_only": True, "corpus_ids": ["NSAID", "nsaid"]},
    {"id": "oral_NSAID", "exhaustive": False, "derivable": True, "derive_from": "NSAID", "routes": ["oral"], "corpus_ids": ["oral_NSAID"]},
    {"id": "topical_NSAID", "exhaustive": False, "derivable": True, "derive_from": "NSAID", "routes": ["topical"], "corpus_ids": ["topical_NSAID"]},
    {"id": "opioid", "exhaustive": False, "derivable": True, "corpus_ids": ["opioid"]},
    {"id": "long_term_opioid", "exhaustive": False, "derivable": False, "corpus_ids": ["long_term_opioid"]},
]


def catalog() -> ConceptCatalog:
    return ConceptCatalog.from_rows(TESTS, DRUGS, CLASSES)


class LabMappingTest(unittest.TestCase):
    def test_site_map_unit_conversion_feeds_values_under_every_corpus_id(self) -> None:
        cat = catalog()
        site = SiteCodeMap.from_dict({"labs": [
            {"code": "L001", "concept": "SCr", "unit": "µmol/L"},
            {"code": "L002", "concept": "Triglycerides", "unit": "mg/dL"},
        ]}, cat)
        mapped = map_emr({"labResultList": [
            {"examCode": "L001", "itemName": "Creatinine", "value": "97", "testDate": "2026-09-04 09:00:00"},
            {"examCode": "L002", "itemName": "TG", "value": 180, "testDate": "2026-09-04 09:00:00"},
        ]}, cat, site)
        self.assertAlmostEqual(mapped.facts.values["SCr"], 1.0971, places=3)
        self.assertEqual(mapped.facts.values["TG"], 180)
        self.assertEqual(mapped.facts.values["Triglycerides"], 180)

    def test_name_only_match_is_a_suggestion_not_a_value(self) -> None:
        mapped = map_emr({"labResultList": [{"examCode": "X9", "itemName": "Creatinine", "value": "1.2", "testDate": "2026-09-04"}]}, catalog())
        self.assertNotIn("SCr", mapped.facts.values)
        self.assertEqual([(i.kind, i.suggestion) for i in mapped.issues], [("lab_unmapped", "SCr")])

    def test_loinc_code_without_unit_is_not_used(self) -> None:
        mapped = map_emr({"labResultList": [{"examCode": "2160-0", "value": "1.2", "testDate": "2026-09-04"}]}, catalog())
        self.assertNotIn("SCr", mapped.facts.values)
        self.assertEqual(mapped.issues[0].kind, "lab_unit_unknown")
        with_unit = map_emr({"labResultList": [{"examCode": "2160-0", "value": "1.2", "unit": "mg/dL", "testDate": "2026-09-04"}]}, catalog())
        self.assertEqual(with_unit.facts.values["SCr"], 1.2)

    def test_rejects_unconvertible_units_out_of_range_and_non_numeric_values(self) -> None:
        cat = catalog()
        site = SiteCodeMap.from_dict({"labs": [{"code": "A1", "concept": "HbA1c", "unit": "%"}, {"code": "A2", "concept": "SCr", "unit": "g/L"}]}, cat)
        mapped = map_emr({
            "labResultList": [
                {"examCode": "A1", "value": "<4.0", "testDate": "2026-09-01"},
                {"examCode": "A2", "value": "1.0", "testDate": "2026-09-01"},
            ],
            "glycatedHemoglobinList": [{"testDate": "2026-09-02", "hba1cValue": 100}],
        }, cat, site)
        self.assertEqual(mapped.facts.values, {})
        self.assertEqual(sorted(i.kind for i in mapped.issues), ["lab_non_numeric", "lab_out_of_range", "lab_unit_unsupported"])

    def test_ifcc_hba1c_converts_with_offset_and_latest_result_wins(self) -> None:
        cat = catalog()
        site = SiteCodeMap.from_dict({"labs": [{"code": "A1C", "concept": "HbA1c", "unit": "mmol/mol"}]}, cat)
        mapped = map_emr({"labResultList": [
            {"examCode": "A1C", "value": "53", "testDate": "2026-09-05"},
            {"examCode": "A1C", "value": "80", "testDate": "2026-08-01"},
        ]}, cat, site)
        self.assertAlmostEqual(mapped.facts.values["HbA1c"], 7.0, places=1)

    def test_panel_codes_are_not_single_values(self) -> None:
        cat = catalog()
        site = SiteCodeMap.from_dict({"labs": [{"code": "CBC1", "concept": "CBC", "unit": "-"}]}, cat)
        mapped = map_emr({"labResultList": [{"examCode": "CBC1", "value": "1", "testDate": "2026-09-01"}]}, cat, site)
        self.assertEqual(mapped.issues[0].kind, "lab_not_single")

    def test_derives_egfr_and_crcl_only_with_complete_inputs(self) -> None:
        cat = catalog()
        site = SiteCodeMap.from_dict({"labs": [{"code": "L001", "concept": "SCr", "unit": "mg/dL"}]}, cat)
        payload = {"labResultList": [{"examCode": "L001", "value": "1.0", "testDate": "2026-09-01"}]}
        self.assertNotIn("eGFR", map_emr(payload, cat, site).facts.values)
        mapped = map_emr({**payload, "calculationContext": {"age_years": 60, "sex": "male", "weight_kg": 72}}, cat, site)
        self.assertAlmostEqual(mapped.facts.values["eGFR"], 86.2, delta=0.1)  # 142 × 1.111^-1.2 × 0.9938^60
        self.assertAlmostEqual(mapped.facts.values["CrCl"], 80.0, places=1)

    def test_unknown_site_concept_is_rejected(self) -> None:
        with self.assertRaises(ConceptMapError):
            SiteCodeMap.from_dict({"labs": [{"code": "Z", "concept": "NotAConcept"}]}, catalog())


class MedicationMappingTest(unittest.TestCase):
    def test_korean_product_names_map_by_parenthesised_ingredients(self) -> None:
        mapped = map_emr({"medicineDtailList": [
            {"medCode": "1", "medName": "자누메트정50/500mg(시타글립틴인산염수화물,메트포르민염산염)"},
            {"medCode": "2", "medName": "리튬정300mg(탄산리튬)"},
        ]}, catalog())
        facts = mapped.facts
        self.assertTrue(facts.medications_complete)
        self.assertEqual(facts.medications, {"sitagliptin", "metformin", "lithium"})
        self.assertTrue(evaluate({"med_class": "DPP4i"}, facts))
        self.assertFalse(evaluate({"med_class": "statin"}, facts))
        self.assertFalse(evaluate({"med": "atorvastatin"}, facts))

    def test_prefix_matching_does_not_confuse_similar_names(self) -> None:
        mapped = map_emr({"medicineDtailList": [{"medCode": "1", "medName": "메치론정4mg(메틸프레드니솔론)"}]}, catalog())
        self.assertEqual(mapped.facts.medications, {"methylprednisolone"})

    def test_unmapped_prescription_makes_absence_unknown(self) -> None:
        mapped = map_emr({"medicineDtailList": [
            {"medCode": "1", "medName": "메트포르민정500mg(메트포르민염산염)"},
            {"medCode": "med-022", "medName": "약이름2"},
        ]}, catalog())
        self.assertFalse(mapped.facts.medications_complete)
        self.assertTrue(evaluate({"med": "metformin"}, mapped.facts))
        self.assertIsNone(evaluate({"med": "atorvastatin"}, mapped.facts))
        self.assertIsNone(evaluate({"med_class": "statin"}, mapped.facts))
        self.assertEqual([i.kind for i in mapped.issues], ["med_unmapped"])

    def test_site_map_non_catalog_ingredient_keeps_exhaustive_classes_decidable(self) -> None:
        cat = catalog()
        site = SiteCodeMap.from_dict({"meds": [{"code": "AML", "ingredients": ["amlodipine"]}]}, cat)
        mapped = map_emr({"medicineDtailList": [{"medCode": "AML", "medName": "노바스크정5mg"}]}, cat, site)
        self.assertTrue(mapped.facts.medications_complete)
        self.assertFalse(evaluate({"med_class": "statin"}, mapped.facts))       # exhaustive
        self.assertIsNone(evaluate({"med_class": "lipid_lowering"}, mapped.facts))  # non-exhaustive

    def test_topical_forms_leave_systemic_classes_and_route_classes_need_route(self) -> None:
        mapped = map_emr({"medicineDtailList": [
            {"medCode": "1", "medName": "소론도크림(프레드니솔론)"},
            {"medCode": "2", "medName": "부루펜정200mg(이부프로펜)"},
        ]}, catalog())
        facts = mapped.facts
        self.assertFalse(evaluate({"med_class": "systemic_glucocorticoid"}, facts))
        self.assertTrue(evaluate({"med_class": "NSAID"}, facts))
        self.assertTrue(evaluate({"med_class": "oral_NSAID"}, facts))
        unknown_route = map_emr({"medicineDtailList": [{"medCode": "3", "medName": "이부프로펜(이부프로펜)"}]}, catalog())
        self.assertIsNone(evaluate({"med_class": "oral_NSAID"}, unknown_route.facts))

    def test_products_and_non_derivable_classes(self) -> None:
        mapped = map_emr({"medicineDtailList": [{"medCode": "1", "medName": "서복손설하정(부프레노르핀염산염,날록손염산염이수화물)"}]}, catalog())
        facts = mapped.facts
        self.assertIn("buprenorphine_naloxone", facts.medications)
        self.assertTrue(evaluate({"med_class": "opioid"}, facts))
        self.assertIsNone(evaluate({"med_class": "long_term_opioid"}, facts))

    def test_missing_prescription_list_is_unknown(self) -> None:
        facts = map_emr({}, catalog()).facts
        self.assertIsNone(facts.medications)
        self.assertIsNone(evaluate({"med": "metformin"}, facts))


class DiagnosisAndEvaluatorTest(unittest.TestCase):
    def test_kcd_codes_are_normalised_to_dotted_form(self) -> None:
        self.assertEqual(normalize_kcd("k810"), "K81.0")
        self.assertEqual(normalize_kcd("N18.31"), "N18.31")
        facts = map_emr({"diagnosisList": [{"diagnosisCode": "K810"}]}, catalog()).facts
        self.assertTrue(evaluate({"dx": "K81.0"}, facts))

    def test_default_patient_facts_keep_previous_semantics(self) -> None:
        facts = PatientFacts(values={}, medications=frozenset({"metformin"}), medication_classes=frozenset())
        self.assertFalse(evaluate({"med": "insulin_glargine"}, facts))
        self.assertFalse(evaluate({"med_class": "statin"}, facts))


@unittest.skipUnless((REPO_CORPUS / "test_concepts.jsonl").is_file(), "generated concept catalog not present")
class RepositoryCatalogTest(unittest.TestCase):
    def test_generated_catalog_maps_sample_style_payload(self) -> None:
        cat = ConceptCatalog.load(REPO_CORPUS)
        self.assertIn("SCr", cat.tests)
        self.assertEqual(cat.test_by_loinc.get("2160-0"), "SCr")
        mapped = map_emr({"medicineDtailList": [{"medCode": "x", "medName": "리피토정10mg(아토르바스타틴칼슘삼수화물)"}]}, cat)
        self.assertEqual(mapped.facts.medications, {"atorvastatin"})
        self.assertTrue(evaluate({"med_class": "statin"}, mapped.facts))


if __name__ == "__main__":
    unittest.main()
