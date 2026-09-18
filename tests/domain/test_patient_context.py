from __future__ import annotations

import unittest

from ax_g_ai.domain.patient_context import (
    CodeMapping,
    ContextBuildError,
    PatientContextBuilder,
    QualityStatus,
)


def payload() -> dict:
    return {
        "actor": {
            "user_id": "clinician-1",
            "organization_id": "hospital-a",
            "role": "physician",
            "purpose": "treatment",
        },
        "patient": {"patient_id": "patient-1"},
        "encounter": {"encounter_id": "encounter-1", "type": "outpatient", "status": "open"},
        "range": {
            "from": "2026-09-01T00:00:00+09:00",
            "to": "2026-09-17T00:00:00+09:00",
            "timezone": "Asia/Seoul",
        },
        "observations": [
            {
                "source_record_id": "observation-1",
                "code_system": "hospital-lab-v1",
                "code": "GLU",
                "value": 100,
                "unit": "mg/dL",
                "observed_at": "2026-09-10T09:00:00+09:00",
                "source": "laboratory",
                "status": "final",
                "freshness": "current",
            }
        ],
        "diagnoses": [],
        "prescriptions": [],
        "lab_results": [],
        "missing_data": [],
        "delayed_data": [],
    }


class PatientContextBuilderTest(unittest.TestCase):
    def setUp(self) -> None:
        self.builder = PatientContextBuilder(
            [CodeMapping("hospital-lab-v1", "GLU", "LOINC", "2345-7")]
        )

    def test_preserves_mapped_source_observation_for_clinical_use(self) -> None:
        context = self.builder.build(payload())

        observation = context.observations[0]
        self.assertEqual(observation.source_record_id, "observation-1")
        self.assertEqual(observation.normalized_code_system, "LOINC")
        self.assertEqual(observation.normalized_code, "2345-7")
        self.assertEqual(observation.quality, QualityStatus.USABLE)
        self.assertEqual(context.clinically_usable_observations, (observation,))

    def test_preserves_unmapped_code_but_excludes_it_from_clinical_use(self) -> None:
        source = payload()
        source["observations"][0]["code"] = "UNKNOWN"

        context = self.builder.build(source)

        self.assertEqual(context.observations[0].quality, QualityStatus.UNMAPPED_CODE)
        self.assertEqual(context.data_quality.unmapped, ("observation-1",))
        self.assertEqual(context.clinically_usable_observations, ())

    def test_missing_unit_is_preserved_as_unsafe_not_converted_to_a_value(self) -> None:
        source = payload()
        del source["observations"][0]["unit"]

        context = self.builder.build(source)

        self.assertIsNone(context.observations[0].unit)
        self.assertEqual(
            context.observations[0].quality, QualityStatus.MISSING_REQUIRED_FIELD
        )
        self.assertEqual(context.clinically_usable_observations, ())

    def test_rejects_context_without_a_timezone_qualified_range(self) -> None:
        source = payload()
        source["range"]["from"] = "2026-09-01T00:00:00"

        with self.assertRaisesRegex(ContextBuildError, "timezone"):
            self.builder.build(source)


if __name__ == "__main__":
    unittest.main()
