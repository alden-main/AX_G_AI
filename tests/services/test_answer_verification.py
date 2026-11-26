from __future__ import annotations

import unittest

from ax_g_ai.services.answer_verification import OMISSION_NOTE, SentenceSplitter, build_facts, check_sentence
from ax_g_ai.services.clinical_chat import ClinicalChatService
from tests.services.test_answer_review import SequenceProvider

SNAPSHOT = {
    "blood_sugar": [
        {"measured_at": "2026-06-13T16:58:00+09:00", "value": 200, "unit": "mg/dL", "timing_type": "식후 2시간"},
        {"measured_at": "2026-06-23T09:32:00+09:00", "value": 164, "unit": "mg/dL"},
    ],
    "blood_pressure": [{"measured_at": "2026-06-20T17:12:00+09:00", "sbp": 138, "dbp": 82, "unit": "mmHg"}],
    "lab_results": [{"test_date": "2026-06-23T00:00:00+09:00", "value": "0.92", "item_name": "크레아티닌", "unit": "mg/dL"}],
    "prescriptions": [
        {"med_name": "심바타틴정20밀리그램(심바스타틴)", "prescriptionDate": "2026-06-25"},
        {"med_name": "클래리온정(클래리트로마이신)"},
    ],
}
FACTS = build_facts(SNAPSHOT)


class CheckSentenceTest(unittest.TestCase):
    def test_record_values_and_dates_pass(self) -> None:
        for sentence in (
            "크레아티닌은 2026-06-23 0.92 mg/dL입니다.",
            "혈압은 2026년 6월 20일 138/82 mmHg입니다.",
            "식후 2시간 혈당은 200 mg/dL입니다.",
            "심바스타틴은 2026-06-25에 처방되었습니다.",
            "클래리트로마이신은 처방일 기록 없음입니다.",
            "혈당 164 mg/dL(측정 시점 기록 없음)입니다.",
            "CrCl <30 mL/min이면 감량 기준입니다.",  # thresholds are not patient values
            "정상 상한(100 mg/dL)을 넘습니다.",  # reference values
            "이전보다 0.07 mg/dL 감소했습니다.",  # a computed difference
            "2025년 12월 8.8%였습니다.",  # a month without a day is not a date
            "변경을 검토할 수 있습니다.",
        ):
            self.assertIsNone(check_sentence(sentence, FACTS), sentence)

    def test_each_violation_is_found(self) -> None:
        cases = {
            "2026-07-08에 측정했습니다.": "unsupported_date",
            "CrCl 29.1 mL/min으로 감량이 필요합니다.": "unsupported_value",  # the leaked example value
            "Cockcroft–Gault로 재계산하면 약 103 mL/min 수준입니다.": "unsupported_value",
            "공복 혈당 164 mg/dL입니다.": "untimed_glucose_timing",
            "클래리온정 500mg(2026-06-25 시작)": "borrowed_prescription_date",
            "심바스타틴을 중단하세요.": "prescription_directive",
        }
        for sentence, expected in cases.items():
            self.assertEqual(check_sentence(sentence, FACTS), expected, sentence)

    def test_the_other_drugs_date_in_the_same_line_is_not_borrowing(self) -> None:
        line = "심바스타틴 20mg(28일분, 2026-06-25 처방)과 클래리트로마이신 500mg(처방일 미기재)"
        self.assertIsNone(check_sentence(line, FACTS))

    def test_assessment_and_evidence_values_count_as_sources(self) -> None:
        facts = build_facts(SNAPSHOT, {"medication_alerts": {"alerts": [{"notice": {"date": "2004-01-16"}}]}}, [{"text": "eGFR 30–44 mL/min/1.73m²"}])
        self.assertIsNone(check_sentence("고시(2004-01-16)에 따른 금기입니다.", facts))
        self.assertIsNone(check_sentence("eGFR 30–44 mL/min/1.73m² 구간입니다.", facts))


class SentenceSplitterTest(unittest.TestCase):
    def test_sentences_end_at_line_breaks_and_sentence_marks_but_not_decimals(self) -> None:
        splitter = SentenceSplitter()
        out = splitter.feed("값은 0.")
        out += splitter.feed("92 mg/dL입니다. 다음")
        out += splitter.feed(" 줄\n| 표 | 행 |\n")
        self.assertEqual(out, ["값은 0.92 mg/dL입니다. ", "다음 줄\n", "| 표 | 행 |\n"])
        self.assertEqual(splitter.feed("2026. 6. 23. 마지막"), [])
        self.assertEqual(splitter.flush(), ["2026. 6. 23. 마지막"])


class VerifiedStreamTest(unittest.IsolatedAsyncioTestCase):
    async def test_a_failing_first_sentence_is_requested_again(self) -> None:
        provider = SequenceProvider(("공복 혈당 164 mg/dL입니다. ", "나머지."), ("혈당 164 mg/dL(측정 시점 기록 없음)입니다.",))
        with self.assertLogs("uvicorn.error", level="INFO") as logs:
            deltas = [d async for d in ClinicalChatService(provider).stream_prepared_snapshot(question="혈당은?", snapshot=SNAPSHOT)]
        self.assertEqual("".join(deltas), "혈당 164 mg/dL(측정 시점 기록 없음)입니다.")
        self.assertEqual(len(provider.requests), 2)
        self.assertEqual(provider.closed, 2)  # the rejected stream was closed
        self.assertTrue(any("reason=unverified_opening attempt=1" in line for line in logs.output))
        self.assertTrue(any("answer_verified operation=prepared_snapshot_chat_stream sentences=1 dropped=1 regenerated=True "
                            "problems=untimed_glucose_timing" in line for line in logs.output))
        self.assertFalse(any("164" in line for line in logs.output))  # no answer text in the logs

    async def test_a_leading_blank_line_does_not_count_as_the_first_sentence(self) -> None:
        # Recommendation answers began with a blank line, which used to disable the first-sentence retry.
        provider = SequenceProvider(("\n공복 혈당 164 mg/dL입니다. ",), ("혈당 164 mg/dL(측정 시점 기록 없음)입니다.",))
        with self.assertLogs("uvicorn.error", level="INFO"):
            deltas = [d async for d in ClinicalChatService(provider).stream_prepared_snapshot(question="혈당은?", snapshot=SNAPSHOT)]
        self.assertEqual(len(provider.requests), 2)
        self.assertEqual("".join(deltas).strip(), "혈당 164 mg/dL(측정 시점 기록 없음)입니다.")

    async def test_a_failing_later_sentence_is_dropped_and_noted(self) -> None:
        provider = SequenceProvider(("크레아티닌은 0.92 mg/dL입니다. ", "CrCl 29.1 mL/min입니다. ", "추적이 필요합니다."))
        with self.assertLogs("uvicorn.error", level="INFO"):
            deltas = [d async for d in ClinicalChatService(provider).stream_prepared_snapshot(question="신기능은?", snapshot=SNAPSHOT)]
        self.assertEqual("".join(deltas), "크레아티닌은 0.92 mg/dL입니다. 추적이 필요합니다.\n\n" + OMISSION_NOTE)
        self.assertEqual(len(provider.requests), 1)

    async def test_nothing_verifiable_falls_back_to_the_notice(self) -> None:
        provider = SequenceProvider(("2026-07-08 측정.",), ("2026-07-09 측정.",))
        with self.assertLogs("uvicorn.error", level="INFO"):
            answer = "".join([d async for d in ClinicalChatService(provider).stream_summary_once(snapshot=SNAPSHOT)])
        # The model text falls back; the code-made record block still follows (R-F29).
        self.assertTrue(answer.startswith("AI 답변을 만들지 못했습니다. 질문을 조금 바꾸거나 다시 시도해 주세요.\n\n[기록 — 진단·처방]"))


if __name__ == "__main__":
    unittest.main()
