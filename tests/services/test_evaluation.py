from __future__ import annotations

from datetime import date
import unittest

from ax_g_ai.services.evaluation import (
    ChatVersion,
    EvaluationGate,
    EvaluationNotApprovedError,
    EvaluationRecord,
)


class EvaluationGateTest(unittest.TestCase):
    """D-10이 미승인 챗봇 후보를 기본적으로 차단하는지 검증한다."""

    def test_blocks_a_version_without_a_clinical_approval_record(self) -> None:
        with self.assertRaises(EvaluationNotApprovedError):
            EvaluationGate().require_approved(ChatVersion("ai-tank", "model-1", "prompt-1", "knowledge-1"))

    def test_allows_only_the_exactly_approved_version(self) -> None:
        version = ChatVersion("ai-tank", "model-1", "prompt-1", "knowledge-1")
        gate = EvaluationGate([
            EvaluationRecord(version, "eval-1", date(2026, 9, 1), "reviewer-1", True, "approval-1")
        ])

        gate.require_approved(version)
        with self.assertRaises(EvaluationNotApprovedError):
            gate.require_approved(ChatVersion("ai-tank", "model-2", "prompt-1", "knowledge-1"))


if __name__ == "__main__":
    unittest.main()
