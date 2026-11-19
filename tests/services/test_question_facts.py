from __future__ import annotations

import unittest

from ax_g_ai.services.question_facts import (
    has_medication_context,
    merge_question_findings,
    pregnancy_statement,
    question_findings,
    stated_pregnancy,
)


class StatedPregnancyTest(unittest.TestCase):
    def test_only_explicit_statements_count(self) -> None:
        for text in ("환자가 임신 12주인데 처방 점검해줘", "임신 중인 환자", "임산부 처방 검토", "임부인데 써도 되나", "patient is pregnant"):
            self.assertIs(stated_pregnancy(text), True, text)
        for text in ("임신 아님", "임신은 아닌데 병용금기 확인", "임신하지 않았다고 함", "임신 가능성 없음", "비임신 여성", "임신 중이 아님",
                     "not pregnant"):
            self.assertIs(stated_pregnancy(text), False, text)
        # Polite and casual endings of the same denial.
        for text in ("임신은 아닙니다", "임신이 아닙니다", "임신 아닙니다", "임신 중은 아닙니다", "임신한 상태가 아닙니다",
                     "임신 상태는 아니에요", "임신 아냐", "임신은 아녜요"):
            self.assertIs(stated_pregnancy(text), False, text)
        self.assertEqual(merge_question_findings({"pregnancy": True}, "임신은 아닙니다. 처방 점검해 주세요"),
                         ({}, {"pregnancy": {"clinical_note": True, "question": False}}))
        # A question about the fact, or the DUR list name, is not a statement.
        for text in ("임신 여부 확인해줘", "임부금기 약 있어?", "혈압 추이 요약", "", None):
            self.assertIsNone(stated_pregnancy(text), text)

    def test_findings_take_the_first_text_that_states_one(self) -> None:
        self.assertEqual(question_findings("처방 점검", "임신 8주"), {"pregnancy": True})
        self.assertEqual(question_findings("임신 아님", "임신 8주"), {"pregnancy": False})
        self.assertEqual(question_findings("요약해줘"), {})

    def test_status_changes_are_marked(self) -> None:
        for text in ("출산 후 2주인데 이 약 써도 돼?", "어제 분만했어요", "제왕절개 후 처방 점검", "산후 우울로 내원", "유산된 뒤 처방",
                     "임신 중절 수술 후", "postpartum patient", "she gave birth last week"):
            self.assertEqual(pregnancy_statement(text), (False, True), text)
        for text in ("새로 임신 확인됨", "최근 임신 판정 받았어요", "출산 후 다시 임신 8주", "patient is now pregnant"):
            self.assertEqual(pregnancy_statement(text), (True, True), text)
        # Still pregnant or merely planned, or a question about it: not a change.
        self.assertEqual(pregnancy_statement("임신 30주, 출산 예정일 확인"), (True, False))
        self.assertEqual(pregnancy_statement("절박유산 진단, 임신 9주"), (True, False))
        self.assertEqual(pregnancy_statement("최근 임신 여부 확인해줘"), (None, False))
        self.assertEqual(pregnancy_statement("새로 임신하지 않았음"), (False, False))
        self.assertEqual(pregnancy_statement("BMI 계산후 알려줘"), (None, False))

    def test_merge_memo_and_question(self) -> None:
        # Memo silent: the question fills it in.
        self.assertEqual(merge_question_findings({}, "임신 8주인데"), ({"pregnancy": True}, {}))
        # Same value, or question silent: the memo stands.
        self.assertEqual(merge_question_findings({"pregnancy": True}, "임신 중"), ({"pregnancy": True}, {}))
        self.assertEqual(merge_question_findings({"pregnancy": True, "fever": True}, "처방 점검"), ({"pregnancy": True, "fever": True}, {}))
        # A status change in the question overrides the older memo, either way.
        self.assertEqual(merge_question_findings({"pregnancy": True}, "출산 후 처방 점검"), ({"pregnancy": False}, {}))
        self.assertEqual(merge_question_findings({"pregnancy": False}, "이번에 임신 확인됨"), ({"pregnancy": True}, {}))
        # Disagreement without a cue: neither is applied, the conflict is reported.
        findings, conflicts = merge_question_findings({"pregnancy": True, "fever": True}, "임신 아님")
        self.assertEqual(findings, {"fever": True})
        self.assertEqual(conflicts, {"pregnancy": {"clinical_note": True, "question": False}})

    def test_medication_context(self) -> None:
        self.assertTrue(has_medication_context("와파린 추가해도 돼?"))
        self.assertTrue(has_medication_context("이 약들 같이 먹어도 괜찮아?"))
        self.assertFalse(has_medication_context("포도당 수치 어때?"))


if __name__ == "__main__":
    unittest.main()
