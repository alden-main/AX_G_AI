from __future__ import annotations

import dataclasses
import os
import re
import unittest
from unittest.mock import patch

from ax_g_ai.config import ConfigurationError, RuntimeSettings
from ax_g_ai.prompts.clinical import (
    compose_prepared_snapshot_question,
    compose_recommendation_input,
    mark_undated_prescriptions,
    mark_untimed_glucose,
    prompt_texts,
)
from ax_g_ai.prompts.data_labels import KEY_NAMES, RULE_KEYS, VALUE_NAMES, korean_view, prompt_data, render_rule
from ax_g_ai.services.clinical_chat import ClinicalChatService
from ax_g_ai.services.ports import OpenAIRequest
from tests.sse import collect

SNAPSHOT = {"recorded_none": ["처방"], "lab_results": [{"test_date": "2026-06-23", "value": "0.92", "item_name": "크레아티닌", "unit": "mg/dL"}]}
ASSESSMENT = {
    "items": [{"kind": "workup", "status": "possible", "matched": ["나이 >= 18"], "missing": [{"kind": "finding", "key": "x_y", "label": "x y"}]}],
    "medication_alerts": {"alerts": [{"type": "병용금기", "status": "met", "products": [{"med_name": "A"}]}]},
    "dose_assessments": [{"kind": "dose", "anchor": ["prescribed"], "alert": "high", "rules": [{"status": "met"}]}],
}


class RuleKeysTest(unittest.TestCase):
    def test_every_rule_name_is_the_korean_name_of_its_english_key(self) -> None:
        for name, key in RULE_KEYS.items():
            values = {korean for names in VALUE_NAMES.values() for english, korean in names.items() if english == key}
            self.assertTrue(KEY_NAMES.get(key) == name or name in values, f"{name} -> {key}")

    def test_both_label_styles_render_every_marker(self) -> None:
        for labels in ("korean", "english"):
            for field in dataclasses.fields(prompt_texts(labels)):
                self.assertNotIn("⟪", getattr(prompt_texts(labels), field.name), (labels, field.name))
        with self.assertRaises(KeyError):
            render_rule("⟪없는 이름⟫", "korean")

    def test_rule_texts_carry_no_realistic_example_values(self) -> None:
        # A small model copied the example "CrCl 29.1 mL/min" into an answer about a patient without creatinine.
        texts = prompt_texts("korean")
        rules = " ".join(getattr(texts, field.name) for field in dataclasses.fields(texts))
        for example in ("29.1", "92.8", "58 kg", "2026-05-12", "VA/DoD", "프레가발린", "어지럼증", "임신 아님"):
            self.assertNotIn(example, rules)
        self.assertIsNone(re.search(r"\d+\.\d+\s*(?:mL/min|mg|mcg|kg)", rules))


class LabelStyleTest(unittest.TestCase):
    def test_korean_style_names_data_and_rules_in_korean(self) -> None:
        prompt = compose_recommendation_input(SNAPSHOT, ASSESSMENT, [])
        self.assertIn('"기록상 없음":["처방"]', prompt)
        self.assertIn('"판정":"조건부"', prompt)
        self.assertIn("'판정'이 '충족'인 항목", prompt)
        self.assertNotIn("recorded_none", prompt)
        self.assertNotIn("x_y", prompt)  # internal keys are dropped

    def test_english_style_keeps_data_keys_and_rule_names_in_english(self) -> None:
        prompt = compose_recommendation_input(SNAPSHOT, ASSESSMENT, [], labels="english")
        self.assertIn('"recorded_none":["처방"]', prompt)
        self.assertIn('"status":"possible"', prompt)
        self.assertIn("status이 met인 항목", prompt)
        self.assertIn("dose_assessments는 말뭉치 용량 규칙을", prompt)
        self.assertIn("medication_alerts는 현재 처방", prompt)
        self.assertNotIn('"판정"', prompt)
        self.assertNotIn("x_y", prompt)
        self.assertEqual(prompt_data({"id": 1, "status": "met"}, "english"), {"status": "met"})
        self.assertEqual(korean_view({"id": 1, "status": "met"}), {"판정": "충족"})
        self.assertIn("영문 키·필드 이름", prompt_texts("english").instructions)
        snapshot_prompt = compose_prepared_snapshot_question("처방은?", SNAPSHOT, labels="english")
        self.assertIn("recorded_none에 있는 항목은", snapshot_prompt)

    def test_runtime_setting_selects_the_style(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(RuntimeSettings.from_environment().prompt_data_labels, "korean")
        with patch.dict(os.environ, {"AX_G_PROMPT_DATA_LABELS": "english"}, clear=True):
            self.assertEqual(RuntimeSettings.from_environment().prompt_data_labels, "english")
        with patch.dict(os.environ, {"AX_G_PROMPT_DATA_LABELS": "japanese"}, clear=True), self.assertRaises(ConfigurationError):
            RuntimeSettings.from_environment()


class UndatedPrescriptionTest(unittest.TestCase):
    def test_undated_prescriptions_move_to_their_own_group(self) -> None:
        # A small model gave an undated prescription the latest measurement date, even with an explicit '기록 없음' value.
        summary = {"prescriptions": [
            {"med_name": "A", "dosage": 20, "prescriptionDate": "2026-06-25"},
            {"med_name": "B", "dosage": 500, "prescriptionDate": None},
        ], "blood_pressure": []}
        marked = mark_undated_prescriptions(summary)

        self.assertEqual(list(marked), ["prescriptions", "undated_prescriptions", "blood_pressure"])
        self.assertEqual(marked["prescriptions"], [{"med_name": "A", "dosage": 20, "prescriptionDate": "2026-06-25"}])
        self.assertEqual(marked["undated_prescriptions"], [{"med_name": "B", "dosage": 500, "prescriptionDate": None}])
        self.assertEqual(len(summary["prescriptions"]), 2)  # input untouched
        prompt = compose_prepared_snapshot_question("SOAP로 정리해 주세요", summary)
        self.assertIn('"처방(처방일 기록 없음)":[{"약품명":"B"', prompt)
        self.assertIn("'처방(처방일 기록 없음)'의 약은", prompt)
        self.assertIn('"undated_prescriptions":[', compose_recommendation_input(summary, {}, [], labels="english"))
        only_undated = mark_undated_prescriptions({"prescriptions": [{"med_name": "B"}]})
        self.assertEqual(only_undated, {"undated_prescriptions": [{"med_name": "B"}]})
        self.assertEqual(mark_undated_prescriptions(SNAPSHOT), SNAPSHOT)


class UntimedGlucoseTest(unittest.TestCase):
    def test_glucose_without_timing_says_not_recorded_right_after_the_unit(self) -> None:
        # The 30B model called untimed readings fasting or non-fasting.
        summary = {"blood_sugar": [
            {"measured_at": "t1", "value": 164, "unit": "mg/dL", "extra": 1},
            {"measured_at": "t2", "value": 200, "unit": "mg/dL", "timing_type": "식후 2시간"},
        ]}
        marked = mark_untimed_glucose(summary)
        self.assertEqual(list(marked["blood_sugar"][0]), ["measured_at", "value", "unit", "timing_type", "extra"])
        self.assertEqual(marked["blood_sugar"][0]["timing_type"], "기록 없음")
        self.assertEqual(marked["blood_sugar"][1], summary["blood_sugar"][1])
        self.assertNotIn("timing_type", summary["blood_sugar"][0])  # input untouched
        prompt = compose_prepared_snapshot_question("요약", summary, one_shot_summary=True)
        self.assertIn('"측정 시점":"기록 없음"', prompt)
        self.assertIn("공복·식후나 날짜를 추정해 붙이지 말고", prompt)
        self.assertIn("'측정 시점 기록 없음'으로 쓰세요", compose_prepared_snapshot_question("혈당은?", summary))


class ModelProfileTest(unittest.TestCase):
    def test_profiles_differ_only_in_the_dur_rule(self) -> None:
        compact, capable = prompt_texts("korean", "compact"), prompt_texts("korean", "capable")
        self.assertEqual(compact, prompt_texts("korean"))
        self.assertIn("직접 판단하지 마세요", compact.dur_rule)
        self.assertIn("임상적 의미", capable.dur_rule)
        self.assertIn("선택지마다 끝에 근거를 붙이세요", capable.dur_rule)
        self.assertNotIn("⟪", prompt_texts("english", "capable").dur_rule)
        for field in dataclasses.fields(compact):
            if field.name != "dur_rule":
                self.assertEqual(getattr(compact, field.name), getattr(capable, field.name), field.name)
        prompt = compose_recommendation_input(SNAPSHOT, ASSESSMENT, [], profile="capable")
        self.assertIn(capable.dur_rule, prompt)
        self.assertNotIn(compact.dur_rule, prompt)

    def test_runtime_setting_selects_the_profile(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(RuntimeSettings.from_environment().model_profile, "compact")
        with patch.dict(os.environ, {"AX_G_MODEL_PROFILE": "capable"}, clear=True):
            self.assertEqual(RuntimeSettings.from_environment().model_profile, "capable")
        with patch.dict(os.environ, {"AX_G_MODEL_PROFILE": "large"}, clear=True), self.assertRaises(ConfigurationError):
            RuntimeSettings.from_environment()


class Recording:
    def __init__(self) -> None:
        self.requests: list[OpenAIRequest] = []

    async def stream(self, request: OpenAIRequest):
        self.requests.append(request)
        yield "답변"


class ServiceLabelsTest(unittest.IsolatedAsyncioTestCase):
    async def test_service_uses_the_configured_style_and_marks_the_prompt_version(self) -> None:
        korean, english = Recording(), Recording()
        await collect(ClinicalChatService(korean).stream_prepared_snapshot(question="처방은?", snapshot=SNAPSHOT))
        await collect(ClinicalChatService(english, prompt_labels="english").stream_prepared_snapshot(question="처방은?", snapshot=SNAPSHOT))

        self.assertIn('"기록상 없음"', korean.requests[0].input)
        self.assertIn('"recorded_none"', english.requests[0].input)
        self.assertEqual(english.requests[0].prompt_version, f"{korean.requests[0].prompt_version}-en")
        self.assertEqual(english.requests[0].instructions, prompt_texts("english").instructions)

        capable = Recording()
        await collect(ClinicalChatService(capable, model_profile="capable").stream_prepared_snapshot(question="처방은?", snapshot=SNAPSHOT))
        self.assertEqual(capable.requests[0].prompt_version, f"{korean.requests[0].prompt_version}-capable")


if __name__ == "__main__":
    unittest.main()
