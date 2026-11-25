from __future__ import annotations

import unittest
from datetime import UTC, datetime
from types import SimpleNamespace

from ax_g_ai.services.clinical_grounding import chat_grounding
from ax_g_ai.services.patient_context_store import PreparedRawEmrContext
from ax_g_ai.services.question_facts import is_record_lookup


class FakeService:
    answers_dur_only_itself = True

    def __init__(self, found_on=()):
        self.queries = []
        self.found_on = found_on

    def find_evidence(self, question, retriever=None):
        self.queries.append(question)
        return ("chunk",) if any(word in question for word in self.found_on) else ()


def context():
    return PreparedRawEmrContext(data={"diagnoses": [{"code": "M819", "name_ko": "골다공증"}]}, updated_at=datetime(2026, 11, 1, tzinfo=UTC))


SNAPSHOT = SimpleNamespace(evidence=None, recommender=None)


class RecordLookupTest(unittest.TestCase):
    def test_value_questions_are_lookups(self) -> None:
        for question in ("혈압 기록이 있으면 최근 값을 알려 주세요.", "크레아티닌의 최근 값과 이전 값을 비교해 주세요.",
                         "CrCl 추정치를 구해 주세요.", "음주에 관한 기록을 날짜별로 알려 주세요."):
            self.assertTrue(is_record_lookup(question), question)

    def test_knowledge_recommendation_and_drug_questions_are_not(self) -> None:
        for question in ("당화혈색소 7%의 의미를 알려 주세요.", "최근 혈압 값과 일반적인 관리 목표를 출처와 함께 알려 주세요.",
                         "추천 검사는?", "현재 처방 약끼리 병용금기가 있나요?", "임신 12주인데 처방 기록 확인해 주세요.",
                         "어지럼증의 원인을 감별해 주세요."):
            self.assertFalse(is_record_lookup(question), question)


class ChatGroundingTest(unittest.TestCase):
    def test_record_lookup_does_not_search_general_evidence(self) -> None:
        service = FakeService(found_on=("혈압",))
        grounding = chat_grounding("혈압 기록이 있으면 최근 값을 알려 주세요.", context(), service, SNAPSHOT)
        self.assertEqual(service.queries, [])
        self.assertEqual(grounding.evidence, ())

    def test_diagnosis_hint_only_for_recommendation_questions(self) -> None:
        service = FakeService(found_on=("골다공증",))
        chat_grounding("이 수치의 의미를 설명해 주세요.", context(), service, SNAPSHOT)
        self.assertEqual(service.queries, ["이 수치의 의미를 설명해 주세요."])
        service = FakeService(found_on=("골다공증",))
        grounding = chat_grounding("추천 검사는?", context(), service, SNAPSHOT)
        self.assertEqual(service.queries, ["추천 검사는?", "추천 검사는? 골다공증"])
        self.assertEqual(grounding.evidence, ("chunk",))


if __name__ == "__main__":
    unittest.main()
