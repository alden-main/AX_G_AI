"""`emr_test.json`을 실제 AI Provider에 보내 답변을 눈으로 확인하는 수동 테스트.

기본 테스트 묶음에서는 절대 네트워크를 호출하지 않는다. 실행하려면
``AX_G_RUN_LIVE_EMR_CHAT=true``와 ``AI_*``(또는 ``OPENAI_*``) 환경 변수를 명시적으로
제공해야 한다. 질문은 ``AX_G_EMR_CHAT_QUESTION``으로 바꿀 수 있다.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import unittest

from ax_g_ai.adapters.main_bridge import RawEmrPayloadMapper
from ax_g_ai.adapters.provider import build_provider_client
from ax_g_ai.config import AISettings
from ax_g_ai.services.clinical_chat import ClinicalChatService


@unittest.skipUnless(
    os.getenv("AX_G_RUN_LIVE_EMR_CHAT", "").lower() == "true",
    "Set AX_G_RUN_LIVE_EMR_CHAT=true to send the dummy fixture to the AI provider.",
)
class LiveEmrChatTest(unittest.IsolatedAsyncioTestCase):
    """더미 EMR 문맥으로 Provider의 실제 구조화 요약과 스트리밍 답변을 출력한다."""

    async def test_prints_summary_and_answer_for_the_fixture_question(self) -> None:
        fixture = Path(__file__).parents[2] / "emr_test.json"
        payload = json.loads(fixture.read_text(encoding="utf-8"))
        source = payload.get("data", payload)
        context = RawEmrPayloadMapper().map(source["patient"]["patientId"], payload)
        service = ClinicalChatService(build_provider_client(AISettings.from_environment()))
        question = os.getenv("AX_G_EMR_CHAT_QUESTION", "해당 환자의 최근 vital 정보를 요약해줘.")

        summary = await service.summarize_prepared_snapshot(snapshot=context.data)
        answer = "".join([
            delta async for delta in service.stream_prepared_snapshot(question=question, snapshot=context.data)
        ])

        print(f"\n[준비 요약]\n{json.dumps(summary.as_dict(), ensure_ascii=False, indent=2)}")
        print(f"\n[질문]\n{question}\n\n[AI 답변]\n{answer}\n")
        self.assertTrue(answer.strip())


if __name__ == "__main__":
    unittest.main(verbosity=2)
