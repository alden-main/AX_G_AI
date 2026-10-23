from __future__ import annotations

import unittest

from ax_g_ai.services.clinical_calculations import calculate_review_aids


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
        ])

    def test_never_infers_missing_calculation_inputs(self) -> None:
        results, needed = calculate_review_aids({"weight_kg": 68, "height_cm": 170})

        self.assertEqual([item.name for item in results], ["BMI", "체표면적"])
        self.assertTrue(any("Cockcroft–Gault" in item for item in needed))


if __name__ == "__main__":
    unittest.main()
