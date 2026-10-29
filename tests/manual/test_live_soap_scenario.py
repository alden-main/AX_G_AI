"""통합 테스트 시나리오(LLM SOAP 질의 2건)를 실제 Bridge 경로로 검증하는 수동 테스트.

`soap_scenario_emr.json`은 시나리오 예시 응답에 등장하는 수치로 구성한 원문 EMR fixture다.
fixture → snapshot 검증은 항상 실행되며 네트워크를 쓰지 않는다. OpenAI 호출은
``AX_G_RUN_LIVE_SOAP_SCENARIO=true``와 OpenAI 환경 변수를 명시적으로 제공할 때만 실행한다.
LLM 출력은 문구가 매번 달라지므로 구조·핵심 수치·키워드로만 판정한다.
"""

from __future__ import annotations

from datetime import date
import json
import os
from pathlib import Path
import re
import unittest

from ax_g_ai.adapters.main_bridge import RawEmrPayloadMapper
from ax_g_ai.config import AISettings
from ax_g_ai.adapters.provider import build_provider_client
from ax_g_ai.services.access_policy import AccessRequest, ContextGuard
from ax_g_ai.services.audit import InMemoryAuditSink
from ax_g_ai.services.clinical_chat import ClinicalChatService, compose_prepared_snapshot_question
from ax_g_ai.services.evaluation import ChatVersion, EvaluationGate, EvaluationRecord
from ax_g_ai.services.knowledge import ApprovedKnowledgeStore, KnowledgeDocument


FIXTURE = Path(__file__).with_name("soap_scenario_emr.json")
PATIENT_ID = "scenario-soap-001"

FIRST_QUESTION = "이 환자의 EMR 정보와 검사결과를 바탕으로 SOAP 형식의 진료기록을 작성해줘. 각각 4줄이 넘지않도록 요약해줘."
SECOND_QUESTION = (
    "환자가 저혈당 증상은 따로 없다고 했어.어지럼증 등도 없다고 했고. "
    "이 내용을 반영해서 SOAP 형식의 진료기록을 작성해줘. 이번에도 각각 4줄이 넘지않도록 요약해줘."
)

KEY_VALUES = ("232", "88", "6.9", "70/60")
# 시나리오 예시처럼 한 줄에 한 가지 사실을 담는지 보는 느슨한 상한이다.
MAX_LINE_LENGTH = 140
DISCLAIMERS = ("진단을 확정하지", "의료진의 확인이 필요", "제공된 자료만으로 판단하기 어렵", "샘플 구조")
_SECTION_NAMES = ("Subjective", "Objective", "Assessment", "Plan")
_HEADING = re.compile(r"^[#*\s]*(?:[SOAP]\s*\(\s*)?(Subjective|Objective|Assessment|Plan)\b", re.IGNORECASE)


def _snapshot() -> dict[str, object]:
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    return dict(RawEmrPayloadMapper().map(PATIENT_ID, payload).data)


class _AllowScenarioAccess:
    def is_allowed(self, request: AccessRequest) -> bool:
        return True


class _DiscardSessionState:
    def clear(self, session_id: str) -> None:
        pass


def _service(settings: AISettings) -> ClinicalChatService:
    """런타임과 같은 prepared snapshot 경로를 쓰도록 서비스 경계를 조립한다."""
    today = date.today()
    version = ChatVersion(settings.provider, settings.model, "soap-scenario-v1", "not-used")
    audit = InMemoryAuditSink()
    return ClinicalChatService(
        ContextGuard(_AllowScenarioAccess(), _DiscardSessionState(), audit),
        ApprovedKnowledgeStore((KnowledgeDocument("placeholder", "placeholder", "v1", today, "", True, today, None),)),
        build_provider_client(settings),
        audit,
        EvaluationGate((EvaluationRecord(version, "soap-scenario", today, "local-only", True, "local-only"),)),
        version,
    )


def soap_sections(answer: str) -> dict[str, list[str]]:
    """답변을 S/O/A/P 제목 기준으로 나누고, 각 섹션의 비어 있지 않은 줄을 모은다."""
    sections: dict[str, list[str]] = {}
    current: str | None = None
    for line in answer.splitlines():
        heading = _HEADING.match(line)
        if heading:
            current = heading.group(1).capitalize()
            sections[current] = []
        elif current is not None and line.strip():
            sections[current].append(line.strip())
    return sections


class SoapScenarioFixtureTest(unittest.TestCase):
    """시나리오 수치가 비식별 snapshot과 Provider 입력에 그대로 전달되는지 확인한다."""

    def test_fixture_values_reach_the_provider_prompt(self) -> None:
        prompt = compose_prepared_snapshot_question(FIRST_QUESTION, _snapshot())

        for value in ("232", "180", "88", "6.9", "4.2", "84.12", "No growth", "Nitroglycerin", "HumalogMix25", "Aspirin", "D/W 1000 ml"):
            self.assertIn(value, prompt)
        self.assertIn('"sbp":70,"dbp":60', prompt)
        self.assertNotIn(PATIENT_ID, prompt)
        self.assertNotIn("시나리오환자", prompt)

    def test_section_parser_accepts_scenario_example_headings(self) -> None:
        example = "S (Subjective)\n• a\nO (Objective)\n• b\n• c\nAssessment:\n• d\n**P (Plan)**\n• e"

        self.assertEqual(
            soap_sections(example),
            {"Subjective": ["• a"], "Objective": ["• b", "• c"], "Assessment": ["• d"], "Plan": ["• e"]},
        )


@unittest.skipUnless(
    os.getenv("AX_G_RUN_LIVE_SOAP_SCENARIO", "").lower() == "true",
    "Set AX_G_RUN_LIVE_SOAP_SCENARIO=true to send the SOAP scenario fixture to OpenAI.",
)
class LiveSoapScenarioTest(unittest.TestCase):
    """시나리오 질문 2건을 OpenAI에 보내 SOAP 형식·분량·근거 반영을 판정한다."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.service = _service(AISettings.from_environment())
        cls.snapshot = _snapshot()

    def _ask(self, question: str) -> tuple[str, dict[str, list[str]]]:
        answer = self.service.answer_prepared_snapshot(patient_id=PATIENT_ID, question=question, snapshot=self.snapshot)
        print(f"\n[질문]\n{question}\n\n[AI 답변]\n{answer}\n")
        return answer, soap_sections(answer)

    def _assert_soap_shape(self, answer: str, sections: dict[str, list[str]]) -> None:
        self.assertEqual(list(sections), list(_SECTION_NAMES), "S/O/A/P 네 제목이 순서대로 있어야 합니다.")
        for name, lines in sections.items():
            self.assertTrue(lines, f"{name} 섹션이 비어 있습니다.")
            self.assertLessEqual(len(lines), 4, f"{name} 섹션이 4줄을 넘습니다: {lines}")
            for line in lines:
                self.assertTrue(line.startswith("•"), f"{name} 섹션의 줄이 •로 시작하지 않습니다: {line}")
                self.assertLessEqual(len(line), MAX_LINE_LENGTH, f"{name} 섹션의 줄이 너무 깁니다: {line}")
        for phrase in DISCLAIMERS:
            self.assertNotIn(phrase, answer)
        for keyword in ("변경하십시오", "증량", "감량", "처방하세요"):
            self.assertNotIn(keyword, "\n".join(sections["Plan"]))

    def test_first_question_summarizes_emr_as_soap(self) -> None:
        answer, sections = self._ask(FIRST_QUESTION)

        self._assert_soap_shape(answer, sections)
        objective = "\n".join(sections["Objective"])
        for value in KEY_VALUES:
            self.assertIn(value, objective, f"O 섹션에 핵심 수치 {value}가 없습니다.")
        self.assertRegex("\n".join(sections["Assessment"]), "고혈당|혈당")

    def test_second_question_reflects_clinician_reported_symptoms(self) -> None:
        answer, sections = self._ask(SECOND_QUESTION)

        self._assert_soap_shape(answer, sections)
        subjective = "\n".join(sections["Subjective"])
        self.assertIn("저혈당", subjective)
        self.assertIn("어지럼", subjective)
        self.assertNotIn("주관적 호소 기록 없음", subjective)
        for value in KEY_VALUES:
            self.assertIn(value, answer, f"답변에 핵심 수치 {value}가 없습니다.")


if __name__ == "__main__":
    unittest.main(verbosity=2)
