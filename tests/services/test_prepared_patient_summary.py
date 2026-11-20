from __future__ import annotations

import unittest

from pydantic import ValidationError

from ax_g_ai.services.prepared_patient_summary import PreparedPatientSummary


class PreparedPatientSummaryTest(unittest.TestCase):
    def test_parses_the_message_sections_and_guidance_for_the_chat_card(self) -> None:
        summary = PreparedPatientSummary.model_validate_json(
            '{"message":"EMR 데이터가 로드되었습니다.","sections":['
            '{"label":"최근 검사","content":"HbA1c 6.8이 기록되어 있습니다."}'
            '],"guidance":"측정 시점을 함께 확인하세요."}'
        )

        self.assertEqual(summary.as_dict(), {
            "message": "EMR 데이터가 로드되었습니다.",
            "sections": [{"label": "최근 검사", "content": "HbA1c 6.8이 기록되어 있습니다."}],
            "guidance": "측정 시점을 함께 확인하세요.",
        })

    def test_rejects_freeform_or_incomplete_provider_output(self) -> None:
        with self.assertRaises(ValidationError):
            PreparedPatientSummary.model_validate_json("환자 상태를 확인하세요.")
        with self.assertRaises(ValidationError):
            PreparedPatientSummary.model_validate_json(
                '{"message":"제목","sections":[],"guidance":""}'
            )


    def test_output_schema_is_compatible_with_strict_structured_outputs(self) -> None:
        schema = PreparedPatientSummary.model_json_schema()
        objects = [schema, *schema["$defs"].values()]

        for schema in objects:
            self.assertIs(schema["additionalProperties"], False)
            self.assertEqual(set(schema["required"]), set(schema["properties"]))
        self.assertNotIn("minLength", str(PreparedPatientSummary.model_json_schema()))


if __name__ == "__main__":
    unittest.main()
