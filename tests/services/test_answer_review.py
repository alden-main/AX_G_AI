from __future__ import annotations

import unittest

from ax_g_ai.errors import ClinicalChatProviderUnavailableError
from ax_g_ai.services.answer_review import DUR_ONLY_ANSWER, Opening, check_opening, dur_notice, fallback_answer, record_notice
from ax_g_ai.services.clinical_chat import ClinicalChatService
from ax_g_ai.services.ports import OpenAIRequest, ProviderAnswer
from ax_g_ai.services.question_facts import asks_only_for_dur
from tests.sse import collect

QUESTION = "크레아티닌의 최근 값과 이전 값을 날짜·단위와 함께 비교해 주세요."
SNAPSHOT = {"labs": [{"name": "크레아티닌", "value": 0.92, "unit": "mg/dL"}]}


class SequenceProvider:
    """Returns scripted answers (or delta lists for streams) in order, one per call."""

    def __init__(self, *answers: str | tuple[str, ...]) -> None:
        self.answers = list(answers)
        self.requests: list[OpenAIRequest] = []
        self.closed = 0

    async def respond(self, request: OpenAIRequest) -> ProviderAnswer:
        self.requests.append(request)
        answer = self.answers.pop(0)
        return ProviderAnswer(answer if isinstance(answer, str) else "".join(answer))

    async def stream(self, request: OpenAIRequest):
        self.requests.append(request)
        answer = self.answers.pop(0)
        try:
            for delta in (answer,) if isinstance(answer, str) else answer:
                yield delta
        finally:
            self.closed += 1


class OpeningTest(unittest.TestCase):
    def test_echoed_question_is_stripped_ignoring_whitespace(self) -> None:
        answer = "크레아티닌의 최근 값과 이전 값을\n날짜·단위와 함께 비교해 주세요.    최근 값은 0.92 mg/dL입니다."
        self.assertEqual(check_opening(answer, QUESTION, final=True), Opening("ok", "최근 값은 0.92 mg/dL입니다."))

    def test_unusable_answers_are_classified(self) -> None:
        self.assertEqual(check_opening(QUESTION, QUESTION, final=True).problem, "echo_only")
        self.assertEqual(check_opening('```json\n{"medication_alerts": {}}\n```', QUESTION).problem, "raw_json")
        self.assertEqual(check_opening("{\"a\": 1}", QUESTION).problem, "raw_json")

    def test_ordinary_korean_clinical_text_passes(self) -> None:
        for answer in ("HbA1c 6.3%, eGFR 73.5 mL/min/1.73m² (CKD-EPI 2021)입니다.", "크레아티닌은 0.92 mg/dL로 이전보다 낮습니다."):
            self.assertEqual(check_opening(answer, QUESTION, final=True), Opening("ok", answer))

    def test_opening_waits_only_while_it_can_still_be_an_echo(self) -> None:
        self.assertEqual(check_opening("크레아티닌의 최근", QUESTION).state, "wait")
        self.assertEqual(check_opening("크레아티닌은", QUESTION).state, "ok")
        self.assertEqual(check_opening("최근", QUESTION).text, "최근")
        self.assertEqual(check_opening(QUESTION + " ", QUESTION).state, "wait")
        self.assertEqual(check_opening(QUESTION + " ", QUESTION, final=True).problem, "echo_only")
        self.assertEqual(check_opening(" {", QUESTION).problem, "raw_json")


class DurNoticeTest(unittest.TestCase):
    def test_alerts_become_fixed_lines_with_source(self) -> None:
        assessment = {"medication_alerts": {"alerts": [
            {"type": "병용금기", "status": "met", "detail": "근병증, 횡문근융해의 위험증가",
             "products": [{"med_name": "클래리온정(클래리트로마이신)"}, {"med_name": None, "ingredient_name": "simvastatin", "proposed": True}]},
            {"type": "임부금기", "status": "possible", "grade": "1등급", "detail": "임부에 대한 안전성 미확립.",
             "products": [{"med_name": "리스테롤정10밀리그램"}], "condition_note": "비고"},
        ]}}

        self.assertEqual(dur_notice(assessment), (
            "\n\n[처방 점검 — 심평원 DUR 목록]\n"
            "• 병용금기: 클래리온정(클래리트로마이신) + simvastatin (추가 시) — 근병증, 횡문근융해의 위험증가\n"
            "• 임부금기(1등급, 임신 여부 확인 필요): 리스테롤정10밀리그램 — 임부에 대한 안전성 미확립. (비고: 비고)"
        ))
        for empty in (None, {}, {"medication_alerts": None}):
            self.assertIsNone(dur_notice(empty))
        # Nothing was checked for a patient without prescriptions: no "no alerts" line (it would mislead).
        self.assertIsNone(dur_notice({"medication_alerts": {"alerts": [], "unmapped": []}}, prescribed=False))
        self.assertIn("• 현재 처방: DUR 목록상 해당 경고 없음", dur_notice({"medication_alerts": {"alerts": []}}))
        proposed_only = dur_notice({"medication_alerts": {"alerts": [], "proposed": [{"name": "aspirin"}]}}, prescribed=False)
        self.assertEqual(proposed_only, "\n\n[처방 점검 — 심평원 DUR 목록]\n• 추가 예정 약 aspirin: 현재 DUR 목록상 해당 경고 없음")

    def test_no_alert_unmapped_and_proposed_drugs_are_stated(self) -> None:
        checked = {"alerts": [], "unmapped": [{"med_code": "LT4-LOCAL-01", "med_name": "레보티록신나트륨정 0.1mg"}],
                   "proposed": [{"name": "clarithromycin", "ingredient_name": "clarithromycin"}]}

        self.assertEqual(dur_notice({"medication_alerts": checked}), (
            "\n\n[처방 점검 — 심평원 DUR 목록]\n"
            "• 현재 처방: DUR 목록상 해당 경고 없음\n"
            "• 추가 예정 약 clarithromycin: 현재 DUR 목록상 해당 경고 없음\n"
            "• DUR 점검 제외(목록에 없는 제품코드): 레보티록신나트륨정 0.1mg"
        ))
        flagged = {"alerts": [{"type": "병용금기", "status": "met", "products": [{"med_name": "clarithromycin", "proposed": True}]}],
                   "proposed": [{"name": "clarithromycin"}]}
        self.assertNotIn("추가 예정 약", dur_notice({"medication_alerts": flagged}))


class RecordNoticeTest(unittest.IsolatedAsyncioTestCase):
    SNAPSHOT = {
        "diagnoses": [{"code": "E119", "name_ko": "2형 당뇨병"}, {"code": "E039", "name_ko": "갑상선기능저하증"}],
        "prescriptions": [
            {"med_name": "시타그립정100밀리그램", "dosage": 100, "unit": "mg", "frequency": 1, "prescriptionDate": "2026-02-12"},
            {"med_name": "레보티록신나트륨정 0.1mg", "dosage": 0.1, "unit": "mg", "frequency": 1},
        ],
    }

    def test_recorded_diagnoses_and_prescriptions_are_listed_by_code(self) -> None:
        self.assertEqual(record_notice(self.SNAPSHOT), (
            "\n\n[기록 — 진단·처방]\n• 진단: 2형 당뇨병(E119), 갑상선기능저하증(E039)\n"
            "• 처방: 시타그립정100밀리그램 100mg 1일 1회(처방일 2026-02-12); 레보티록신나트륨정 0.1mg 0.1mg 1일 1회(처방일 기록 없음)"
        ))

    def test_recorded_none_says_so_and_missing_lists_are_left_out(self) -> None:
        self.assertEqual(record_notice({"recorded_none": ["처방"], "diagnoses": [{"code": "I10"}]}),
                         "\n\n[기록 — 진단·처방]\n• 진단: I10\n• 처방: 기록 없음")
        self.assertIsNone(record_notice({"blood_sugar": []}))

    async def test_the_summary_ends_with_the_record_block_outside_verification(self) -> None:
        provider = SequenceProvider(("당뇨 조절 중인 환자입니다.",))
        deltas = [d async for d in ClinicalChatService(provider).stream_summary_once(snapshot=self.SNAPSHOT)]
        self.assertEqual(deltas[-1], record_notice(self.SNAPSHOT))
        self.assertIn("하나씩 나열하지 말고", provider.requests[0].input)


class CheckedProviderCallTest(unittest.IsolatedAsyncioTestCase):
    async def test_unusable_answer_is_requested_again_once(self) -> None:
        provider = SequenceProvider(QUESTION, "최근 값은 0.92 mg/dL입니다.")

        with self.assertLogs("uvicorn.error", level="WARNING") as logs:
            answer = await collect(ClinicalChatService(provider).stream_prepared_snapshot(question=QUESTION, snapshot=SNAPSHOT))

        self.assertEqual(answer, "최근 값은 0.92 mg/dL입니다.")
        self.assertEqual(len(provider.requests), 2)
        self.assertIn("provider_answer_rejected operation=prepared_snapshot_chat_stream reason=echo_only attempt=1", logs.output[0])
        self.assertNotIn("크레아티닌", logs.output[0])

    async def test_two_unusable_answers_fall_back_to_a_fixed_notice(self) -> None:
        with self.assertLogs("uvicorn.error", level="WARNING") as logs:
            answer = await collect(ClinicalChatService(SequenceProvider(QUESTION, "{}")).stream_prepared_snapshot(question=QUESTION, snapshot=SNAPSHOT))
        self.assertEqual(answer, fallback_answer(has_notice=False))
        self.assertIn("provider_answer_fallback operation=prepared_snapshot_chat_stream reason=raw_json", logs.output[-1])

        assessment = {"medication_alerts": {"alerts": [{"type": "병용금기", "products": [{"med_name": "A"}, {"med_name": "B"}]}]}}
        with self.assertLogs("uvicorn.error", level="WARNING"):
            answer = await collect(ClinicalChatService(SequenceProvider("{}", "{}")).stream_recommendation(
                snapshot=SNAPSHOT, assessment=assessment, evidence=()))
        self.assertEqual(answer, fallback_answer(has_notice=True) + "\n\n[처방 점검 — 심평원 DUR 목록]\n• 병용금기: A + B")

    async def test_chat_answer_ends_with_the_dur_notice(self) -> None:
        assessment = {"medication_alerts": {"alerts": [{"type": "병용금기", "status": "met", "detail": "위험", "products": [{"med_name": "A"}, {"med_name": "B"}]}]}}
        service = ClinicalChatService(SequenceProvider(("답", "변")))

        deltas = [delta async for delta in service.stream_prepared_snapshot(question=QUESTION, snapshot=SNAPSHOT, assessment=assessment)]

        self.assertEqual(deltas, ["답변", "\n\n[처방 점검 — 심평원 DUR 목록]\n• 병용금기: A + B — 위험"])  # text goes out by sentence

    async def test_stream_strips_an_echo_and_retries_a_json_dump_before_sending_anything(self) -> None:
        provider = SequenceProvider(("크레아티닌의 최근 값과 ", "이전 값을 날짜·단위와 함께 비교해 주세요. ", "최근 ", "0.92"))
        deltas = [delta async for delta in ClinicalChatService(provider).stream_prepared_snapshot(question=QUESTION, snapshot=SNAPSHOT)]
        self.assertEqual(deltas, ["최근 0.92"])

        provider = SequenceProvider(("```json", '{"a": 1}'), ("최근 값", "입니다"))
        with self.assertLogs("uvicorn.error", level="WARNING") as logs:
            deltas = [delta async for delta in ClinicalChatService(provider).stream_prepared_snapshot(question=QUESTION, snapshot=SNAPSHOT)]
        self.assertEqual(deltas, ["최근 값입니다"])
        self.assertEqual(provider.closed, 2)  # the rejected stream was closed
        self.assertIn("reason=raw_json attempt=1", logs.output[0])

        provider = SequenceProvider((QUESTION,), (QUESTION,))
        with self.assertLogs("uvicorn.error", level="WARNING"):
            deltas = [delta async for delta in ClinicalChatService(provider).stream_prepared_snapshot(question=QUESTION, snapshot=SNAPSHOT)]
        self.assertEqual("".join(deltas), fallback_answer(has_notice=False))

    async def test_provider_call_failure_still_fails(self) -> None:
        class Broken:
            async def stream(self, request: OpenAIRequest):
                raise TimeoutError
                yield ""

        with self.assertLogs("uvicorn.error", level="WARNING"), self.assertRaises(ClinicalChatProviderUnavailableError):
            await collect(ClinicalChatService(Broken()).stream_prepared_snapshot(question=QUESTION, snapshot=SNAPSHOT))  # type: ignore[arg-type]


class DurOnlyQuestionTest(unittest.IsolatedAsyncioTestCase):
    ASSESSMENT = {"medication_alerts": {"alerts": [{"type": "병용금기", "detail": "위험", "products": [{"med_name": "A"}, {"med_name": "B"}]}]}}
    PRESCRIBED = {"prescriptions": [{"med_name": "A"}, {"med_name": "B"}]}

    def test_only_questions_about_the_dur_check_alone_qualify(self) -> None:
        for question in ("현재 처방된 약끼리 함께 써도 되는지 DUR 기준으로 확인해 주세요.", "병용금기 있나요?", "약물 상호작용 확인해 줘"):
            self.assertTrue(asks_only_for_dur(question), question)
        for question in ("병용금기인 이유가 뭔가요?", "병용금기면 대체 약은?", "DUR과 용량을 같이 봐 주세요", "SOAP로 정리해 주세요",
                         "임신 중인데 같이 먹어도 되나요?", "크레아티닌 수치 비교해 주세요"):
            self.assertFalse(asks_only_for_dur(question), question)

    async def test_dur_only_question_is_answered_without_the_model(self) -> None:
        provider = SequenceProvider()
        service = ClinicalChatService(provider)
        question = "현재 처방된 약끼리 함께 써도 되는지 DUR 기준으로 확인해 주세요."
        expected = DUR_ONLY_ANSWER + "\n\n[처방 점검 — 심평원 DUR 목록]\n• 병용금기: A + B — 위험"

        deltas = [delta async for delta in service.stream_prepared_snapshot(question=question, snapshot=self.PRESCRIBED, assessment=self.ASSESSMENT)]

        self.assertEqual(deltas, [expected])
        self.assertEqual(provider.requests, [])

    async def test_capable_profile_sends_the_block_first_then_a_short_commentary(self) -> None:
        # The block is sent at once (no thinking wait); the commentary follows from a separate, short request.
        provider = SequenceProvider(("근병증 위험이 있어 ", "근육통·CK를 확인할 수 있습니다."))
        service = ClinicalChatService(provider, model_profile="capable")
        deltas = [delta async for delta in service.stream_prepared_snapshot(
            question="병용금기 있나요?", snapshot=self.PRESCRIBED, assessment=self.ASSESSMENT)]

        block = DUR_ONLY_ANSWER + "\n\n[처방 점검 — 심평원 DUR 목록]\n• 병용금기: A + B — 위험"
        self.assertEqual(deltas, [block, "\n\n[처방 점검 해설]\n", "근병증 위험이 있어 근육통·CK를 확인할 수 있습니다."])
        self.assertEqual(provider.requests[0].operation, "dur_commentary_stream")
        self.assertIn("다섯 문장 이내", provider.requests[0].input)
        self.assertIn("임상적 의미", provider.requests[0].input)  # the capable DUR rule

    async def test_a_failed_commentary_leaves_the_block_as_the_answer(self) -> None:
        provider = SequenceProvider("{}", "{}")  # unusable twice: no fallback notice after the block
        service = ClinicalChatService(provider, model_profile="capable")
        with self.assertLogs("uvicorn.error", level="WARNING"):
            deltas = [delta async for delta in service.stream_prepared_snapshot(
                question="병용금기 있나요?", snapshot=self.PRESCRIBED, assessment=self.ASSESSMENT)]
        self.assertEqual(deltas, [DUR_ONLY_ANSWER + "\n\n[처방 점검 — 심평원 DUR 목록]\n• 병용금기: A + B — 위험"])

    async def test_without_a_dur_result_the_model_answers(self) -> None:
        provider = SequenceProvider("처방 기록이 없습니다.")
        answer = await collect(ClinicalChatService(provider).stream_prepared_snapshot(question="병용금기 있나요?", snapshot={}))
        self.assertEqual(answer, "처방 기록이 없습니다.")
        self.assertEqual(len(provider.requests), 1)


if __name__ == "__main__":
    unittest.main()
