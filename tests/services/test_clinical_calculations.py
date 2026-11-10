from __future__ import annotations

import unittest

from ax_g_ai.services.clinical_calculations import calculate_review_aids, ckd_epi_2021_egfr, gfr_category


class ClinicalCalculationTest(unittest.TestCase):
    def test_calculates_bmi_bsa_and_crcl_from_complete_context(self) -> None:
        results, needed = calculate_review_aids({
            "age_years": 72,
            "sex": "male",
            "weight_kg": 68,
            "height_cm": 170,
            "serum_creatinine_mg_dl": 1.1,
        })

        self.assertEqual(needed, ())
        self.assertEqual([(item.name, item.value, item.unit) for item in results], [
            ("BMI", 23.5, "kg/m²"),
            ("체표면적", 1.79, "m²"),
            ("추정 CrCl", 58.4, "mL/min"),
            ("추정 GFR", 71.3, "mL/min/1.73m²"),
        ])
        self.assertEqual(results[-1].as_dict()["category"], "G2")

    def test_never_infers_missing_calculation_inputs(self) -> None:
        results, needed = calculate_review_aids({"weight_kg": 68, "height_cm": 170})

        self.assertEqual([item.name for item in results], ["BMI", "체표면적"])
        self.assertTrue(any("Cockcroft–Gault" in item for item in needed))
        self.assertTrue(any("CKD-EPI 2021" in item for item in needed))

    def test_ckd_epi_2021_matches_reference_values_and_categories(self) -> None:
        # Expected values computed independently from the published equation.
        cases = [
            (50, "female", 0.6, 109.3, "G1"),
            (40, "male", 0.9, 110.7, "G1"),
            (65, "female", 2.4, 21.9, "G4"),
            (80, "male", 6.0, 8.9, "G5"),
        ]
        for age, sex, creatinine, expected, category in cases:
            with self.subTest(age=age, sex=sex, creatinine=creatinine):
                self.assertEqual(round(ckd_epi_2021_egfr(age, sex, creatinine), 1), expected)
                self.assertEqual(gfr_category(expected), category)

    def test_gfr_category_boundaries_follow_kdigo(self) -> None:
        self.assertEqual([gfr_category(value) for value in (90, 89.9, 60, 59.9, 45, 44.9, 30, 29.9, 15, 14.9)],
                         ["G1", "G2", "G2", "G3a", "G3a", "G3b", "G3b", "G4", "G4", "G5"])

    def test_egfr_needs_no_weight_and_ignores_unsupported_sex(self) -> None:
        results, _ = calculate_review_aids({"age_years": 72, "sex": "male", "serum_creatinine_mg_dl": 1.1})
        self.assertEqual([item.name for item in results], ["추정 GFR"])

        results, needed = calculate_review_aids({"age_years": 72, "sex": "other", "serum_creatinine_mg_dl": 1.1})
        self.assertEqual(results, ())
        self.assertTrue(any("CKD-EPI 2021" in item and "성별" in item for item in needed))


if __name__ == "__main__":
    unittest.main()
