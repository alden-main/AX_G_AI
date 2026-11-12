from __future__ import annotations

import unittest

from ax_g_ai.services.question_facts import has_medication_context, question_findings, stated_pregnancy


class StatedPregnancyTest(unittest.TestCase):
    def test_only_explicit_statements_count(self) -> None:
        for text in ("환자가 임신 12주인데 처방 점검해줘", "임신 중인 환자", "임산부 처방 검토", "임부인데 써도 되나", "patient is pregnant"):
            self.assertIs(stated_pregnancy(text), True, text)
        for text in ("임신 아님", "임신은 아닌데 병용금기 확인", "임신하지 않았다고 함", "임신 가능성 없음", "비임신 여성", "임신 중이 아님",
                     "not pregnant"):
            self.assertIs(stated_pregnancy(text), False, text)
        # A question about the fact, or the DUR list name, is not a statement.
        for text in ("임신 여부 확인해줘", "임부금기 약 있어?", "혈압 추이 요약", "", None):
            self.assertIsNone(stated_pregnancy(text), text)

    def test_findings_take_the_first_text_that_states_one(self) -> None:
        self.assertEqual(question_findings("처방 점검", "임신 8주"), {"pregnancy": True})
        self.assertEqual(question_findings("임신 아님", "임신 8주"), {"pregnancy": False})
        self.assertEqual(question_findings("요약해줘"), {})

    def test_medication_context(self) -> None:
        self.assertTrue(has_medication_context("와파린 추가해도 돼?"))
        self.assertTrue(has_medication_context("이 약들 같이 먹어도 괜찮아?"))
        self.assertFalse(has_medication_context("포도당 수치 어때?"))


if __name__ == "__main__":
    unittest.main()
