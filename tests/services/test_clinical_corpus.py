from __future__ import annotations

import json
import unittest
from pathlib import Path

from ax_g_ai.services.clinical_corpus import CorpusRuleError, PatientFacts, evaluate
from ax_g_ai.corpus.gate import is_citable

REPO_CORPUS = Path(__file__).resolve().parents[2] / "text_data" / "corpus"

APPROVED = {
    "usable": True, "review_status": "approved", "verification": "primary", "claim_status": "개정",
    "ai_tier": "A", "sources": [{"id": "VADOD-CKD-2025", "ai_use": "A"}],
}


class CitableGateTest(unittest.TestCase):
    """The runtime gate (``is_citable``) applied to rule records and passages alike."""

    def test_only_records_passing_every_gate_are_citable(self) -> None:
        records = [
            {**APPROVED, "id": "ok"},
            {**APPROVED, "id": "draft", "review_status": "draft"},
            {**APPROVED, "id": "unverified", "verification": "unverified"},
            {**APPROVED, "id": "retired", "claim_status": "폐기"},
            {**APPROVED, "id": "permission-required", "ai_tier": "permission_required"},
            {**APPROVED, "id": "non-a-source", "sources": [{"id": "KDIGO-CKD-2024", "ai_use": "C"}]},
            {**APPROVED, "id": "mixed-sources", "sources": [{"id": "X", "ai_use": "A"}, {"id": "Y", "ai_use": "U"}]},
            {**APPROVED, "id": "no-sources", "sources": []},
        ]
        self.assertEqual([r["id"] for r in records if is_citable(r, include_unreviewed=False)], ["ok"])
        self.assertEqual([r["id"] for r in records if is_citable(r, include_unreviewed=True)], ["ok", "draft"])

    def test_repository_rules_have_no_approved_records_until_clinical_review(self) -> None:
        approved = skipped = 0
        for path in REPO_CORPUS.glob("*.jsonl"):
            if path.name == "chunks.jsonl":
                continue
            for line in path.read_text(encoding="utf-8").splitlines():
                if is_citable(json.loads(line), include_unreviewed=False):
                    approved += 1
                else:
                    skipped += 1
        self.assertEqual(approved, 0)
        self.assertGreater(skipped, 0)

    def test_repository_ai_tier_records_cite_only_commercially_usable_sources(self) -> None:
        checked = 0
        for path in REPO_CORPUS.glob("*.jsonl"):
            for line in path.read_text(encoding="utf-8").splitlines():
                record = json.loads(line)
                if record.get("ai_tier") != "A":
                    continue
                checked += 1
                if "sources" in record:
                    self.assertTrue(all(source["ai_use"] == "A" for source in record["sources"]), record.get("id", record.get("chunk_id")))
                self.assertNotIn("reference", record)
        self.assertGreater(checked, 30)


class EvaluateConditionTest(unittest.TestCase):
    SGLT2 = {"all": [{"dx": "E11"}, {"var": "eGFR", "op": ">=", "value": 20}]}

    def test_known_facts_give_definite_answers(self) -> None:
        facts = PatientFacts(values={"eGFR": 35.0}, diagnosis_codes=frozenset({"E11.9", "N18.3"}))
        self.assertIs(evaluate(self.SGLT2, facts), True)
        self.assertIs(evaluate(self.SGLT2, PatientFacts(values={"eGFR": 15.0}, diagnosis_codes=frozenset({"E11"}))), False)

    def test_missing_inputs_are_unknown_not_false(self) -> None:
        self.assertIsNone(evaluate(self.SGLT2, PatientFacts(values={}, diagnosis_codes=frozenset({"E11"}))))
        self.assertIsNone(evaluate(self.SGLT2, PatientFacts(values={"eGFR": 35.0})))
        self.assertIsNone(evaluate({"not": {"dx": "E11"}}, PatientFacts(values={})))
        self.assertIsNone(evaluate({"finding": "hematuria"}, PatientFacts(values={}, flags={})))

    def test_kleene_logic_short_circuits_on_definite_values(self) -> None:
        facts = PatientFacts(values={"eGFR": 10.0})
        # A definite False inside all() wins over unknown diagnosis.
        self.assertIs(evaluate(self.SGLT2, facts), False)
        self.assertIs(evaluate({"any": [{"dx": "E11"}, {"var": "eGFR", "op": "<", "value": 30}]}, facts), True)

    def test_medications_and_flags(self) -> None:
        facts = PatientFacts(
            values={}, medications=frozenset({"metformin"}), medication_classes=frozenset({"ACEi", "ARB"}),
            flags={"event:dialysis": False},
        )
        self.assertIs(evaluate({"all": [{"med_class": "ACEi"}, {"med_class": "ARB"}]}, facts), True)
        self.assertIs(evaluate({"med": "metformin"}, facts), True)
        self.assertIs(evaluate({"not": {"event": "dialysis"}}, facts), True)

    def test_unsupported_syntax_raises(self) -> None:
        with self.assertRaises(CorpusRuleError):
            evaluate({"indication_any": ["CKD"]}, PatientFacts(values={}))
        with self.assertRaises(CorpusRuleError):
            evaluate({"var": "eGFR", "op": "~", "value": 1}, PatientFacts(values={"eGFR": 1.0}))

    def test_every_repository_condition_uses_supported_syntax(self) -> None:
        def conditions(node):
            if isinstance(node, dict):
                for key, value in node.items():
                    if key in {"when", "logic", "trigger"} and isinstance(value, dict):
                        yield value
                    elif key == "criteria" and isinstance(value, list):
                        yield from (item for item in value if isinstance(item, dict))
                    else:
                        yield from conditions(value)
            elif isinstance(node, list):
                for item in node:
                    yield from conditions(item)

        count = 0
        for path in REPO_CORPUS.glob("*.jsonl"):
            for line in path.read_text(encoding="utf-8").splitlines():
                for condition in conditions(json.loads(line)):
                    evaluate(condition, PatientFacts(values={}))
                    count += 1
        self.assertGreater(count, 50)


if __name__ == "__main__":
    unittest.main()
