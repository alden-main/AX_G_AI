from __future__ import annotations

import unittest
from datetime import UTC, datetime

from ax_g_ai.api import ClinicalChatResponseBody, EmrSummaryResponseBody
from ax_g_ai.services.clinical_recommendation.schema import Evidence

PASSAGE = {
    "n": 1, "chunk_id": "c1", "claim_id": "k1", "title": "t", "text": "본문", "review_status": "approved",
    "region": None, "region_label": "미상", "priority": "primary", "sources": [{"id": "s", "title": "출처"}],
    "conflicts": [], "updated_at": None,
}
AT = datetime(2026, 1, 1, tzinfo=UTC)


class ResponseBodyTest(unittest.TestCase):
    def test_chat_omits_missing_evidence_but_keeps_nulls_inside_a_passage(self) -> None:
        self.assertNotIn("evidence", ClinicalChatResponseBody(answer="a", emr_updated_at=AT).model_dump())
        evidence = ClinicalChatResponseBody(answer="a", emr_updated_at=AT, evidence=[PASSAGE]).model_dump()["evidence"]  # type: ignore[list-item]
        self.assertEqual(evidence, [PASSAGE])  # the nulls inside the passage stay
        self.assertEqual(list(evidence[0]), [name for name in Evidence.model_fields if name in PASSAGE])  # declaration order

    def test_emr_summary_keeps_its_null_evidence(self) -> None:
        self.assertIsNone(EmrSummaryResponseBody(answer="a", emr_updated_at=AT).model_dump()["evidence"])


if __name__ == "__main__":
    unittest.main()
