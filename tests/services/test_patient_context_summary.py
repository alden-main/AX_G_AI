from __future__ import annotations

import json
import unittest

from ax_g_ai.domain.patient_context import CodeMapping, PatientContextBuilder
from ax_g_ai.services.knowledge import KnowledgeDocument
from ax_g_ai.services.patient_context_summary import (
    PatientContextSummaryBuilder,
    compose_provider_question,
)
from datetime import date
from tests.domain.test_patient_context import payload


class PatientContextSummaryBuilderTest(unittest.TestCase):
    def setUp(self) -> None:
        self.context = PatientContextBuilder((
            CodeMapping("hospital-lab-v1", "GLU", "LOINC", "2345-7"),
        )).build(payload())

    def test_builds_a_minimal_summary_without_patient_or_source_identifiers(self) -> None:
        rendered = PatientContextSummaryBuilder().build(self.context).as_provider_json()
        summary = json.loads(rendered)

        self.assertEqual(summary["observations"], [{
            "code": "2345-7", "value": 100, "unit": "mg/dL",
            "observed_at": "2026-09-10T09:00:00+09:00", "status": "final", "freshness": "current",
        }])
        self.assertNotIn("patient-1", rendered)
        self.assertNotIn("clinician-1", rendered)
        self.assertNotIn("observation-1", rendered)

    def test_prompt_keeps_question_summary_and_evidence_metadata_in_separate_sections(self) -> None:
        prompt = compose_provider_question(
            "최근 혈당을 요약해 주세요.",
            PatientContextSummaryBuilder().build(self.context),
            (KnowledgeDocument("guide-1", "지침", "2026.1", date(2026, 1, 1), "3장", True, date(2026, 1, 1), None),),
        )

        self.assertIn("[의료진 질문]", prompt)
        self.assertIn("최근 혈당을 요약해 주세요.", prompt)
        self.assertIn('"document_id":"guide-1"', prompt)
        self.assertIn('"code":"2345-7"', prompt)
        self.assertNotIn("patient-1", prompt)


if __name__ == "__main__":
    unittest.main()
