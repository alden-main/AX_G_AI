from __future__ import annotations

import unittest

from ax_g_ai.adapters.openai import OpenAIProviderError
from ax_g_ai.services.prepared_patient_summary import PreparedPatientSummary


class PreparedPatientSummaryTest(unittest.TestCase):
    def test_parses_the_message_sections_and_guidance_for_the_chat_card(self) -> None:
        summary = PreparedPatientSummary.from_provider_text(
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
        with self.assertRaises(OpenAIProviderError):
            PreparedPatientSummary.from_provider_text("환자 상태를 확인하세요.")
        with self.assertRaises(OpenAIProviderError):
            PreparedPatientSummary.from_provider_text(
                '{"message":"제목","sections":[],"guidance":""}'
            )


if __name__ == "__main__":
    unittest.main()
