"""Load the reviewed clinical corpus (``text_data/corpus/*.jsonl``) and evaluate its rules.

Only records a clinician approved and whose every cited source is licensed for
commercial AI use (``ai_use: A``) are loaded. Every gate in the generator is
re-checked here, so a hand-edited ``usable: true`` cannot bypass review or licensing.

Rule conditions use three-valued logic: a fact absent from the patient context
evaluates to ``None`` (unknown), never ``False``. Callers must treat ``None`` as
"input needed", mirroring ``clinical_calculations`` which never infers inputs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

USABLE_VERIFICATIONS = frozenset({"primary", "local_db"})
EXCLUDED_CLAIM_STATUSES = frozenset({"폐기", "확인불가"})
CORPUS_FILES = (
    "diagnosis.jsonl", "workup.jsonl", "treatment.jsonl", "dose_rules.jsonl", "lab_thresholds.jsonl",
    "formulas.jsonl", "monitoring.jsonl", "referral.jsonl", "chunks.jsonl",
)
_OPERATORS = {
    "<": lambda a, b: a < b, "<=": lambda a, b: a <= b,
    ">": lambda a, b: a > b, ">=": lambda a, b: a >= b, "==": lambda a, b: a == b,
}
# Condition keys answered from a boolean flag map; absent flags are unknown.
_FLAG_KEYS = ("finding", "event", "risk", "indication", "lab_present", "dx_history")


class CorpusRuleError(ValueError):
    """A corpus condition uses an operator or key this evaluator does not support."""


@dataclass(frozen=True)
class CorpusLoadReport:
    loaded: Mapping[str, int]
    skipped_unapproved: int
    skipped_invalid: int


@dataclass(frozen=True)
class PatientFacts:
    """Facts a rule may test. ``None`` means the whole category is unknown.

    ``medications_complete`` is False when some prescriptions could not be mapped to an
    ingredient: a drug or class missing from the sets is then unknown, not absent.
    ``uncertain_classes`` are classes that stay unknown unless present (non-exhaustive
    member lists, or classes that need dose, duration or route to decide).
    """

    values: Mapping[str, float]
    diagnosis_codes: frozenset[str] | None = None
    medications: frozenset[str] | None = None
    medication_classes: frozenset[str] | None = None
    flags: Mapping[str, bool] | None = None
    medications_complete: bool = True
    uncertain_classes: frozenset[str] = frozenset()


def is_approved(record: Mapping[str, Any]) -> bool:
    """Clinician-approved and backed only by sources licensed for commercial AI use."""
    sources = record.get("sources")
    return (
        record.get("usable") is True
        and record.get("ai_tier") == "A"
        and isinstance(sources, list)
        and bool(sources)
        and all(isinstance(source, Mapping) and source.get("ai_use") == "A" for source in sources)
        and record.get("review_status") == "approved"
        and record.get("verification") in USABLE_VERIFICATIONS
        and record.get("claim_status") not in EXCLUDED_CLAIM_STATUSES
    )


class ClinicalCorpus:
    def __init__(self, records_by_file: Mapping[str, Iterable[Mapping[str, Any]]]) -> None:
        self._records = {name: tuple(records) for name, records in records_by_file.items()}
        self._dose_rules = {record["id"]: record for record in self._records.get("dose_rules.jsonl", ()) if "id" in record}

    @classmethod
    def load(cls, directory: Path) -> tuple["ClinicalCorpus", CorpusLoadReport]:
        records: dict[str, list[Mapping[str, Any]]] = {}
        unapproved = invalid = 0
        for name in CORPUS_FILES:
            path = directory / name
            records[name] = []
            if not path.is_file():
                continue
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        invalid += 1
                        continue
                    if not isinstance(record, dict):
                        invalid += 1
                    elif is_approved(record):
                        records[name].append(record)
                    else:
                        unapproved += 1
        report = CorpusLoadReport({name: len(items) for name, items in records.items()}, unapproved, invalid)
        return cls(records), report

    def records(self, file_name: str) -> tuple[Mapping[str, Any], ...]:
        return self._records.get(file_name, ())

    def dose_rule(self, rule_id: str) -> Mapping[str, Any] | None:
        return self._dose_rules.get(rule_id)

    def classify(self, lab: str, value: float, *, sex: str | None = None) -> list[dict[str, str]]:
        """Return matching threshold bands for ``lab`` with their record ids."""
        matches = []
        for record in self.records("lab_thresholds.jsonl"):
            if record.get("lab") != lab:
                continue
            for band in record.get("bands", ()):
                if "sex" in band and band["sex"] != sex:
                    continue
                if ("min" in band and value < band["min"]) or ("max" in band and value > band["max"]):
                    continue
                matches.append({"record_id": record["id"], "label": band["label"], "context": record.get("context", "")})
        return matches


def evaluate(condition: Mapping[str, Any], facts: PatientFacts) -> bool | None:
    """Evaluate a corpus condition with Kleene three-valued logic."""
    if "all" in condition:
        results = [evaluate(item, facts) for item in condition["all"]]
        return False if False in results else (None if None in results else True)
    if "any" in condition:
        results = [evaluate(item, facts) for item in condition["any"]]
        return True if True in results else (None if None in results else False)
    if "not" in condition:
        result = evaluate(condition["not"], facts)
        return None if result is None else not result
    if condition.get("always") is True:
        return True
    if "var" in condition:
        operator = _OPERATORS.get(condition.get("op"))
        if operator is None:
            raise CorpusRuleError(f"unsupported operator: {condition.get('op')!r}")
        value = facts.values.get(condition["var"])
        return None if value is None else operator(value, condition["value"])
    if "dx" in condition:
        if facts.diagnosis_codes is None:
            return None
        return any(code.startswith(condition["dx"]) for code in facts.diagnosis_codes)
    if "med" in condition:
        if facts.medications is None:
            return None
        return True if condition["med"] in facts.medications else (False if facts.medications_complete else None)
    if "med_class" in condition:
        if facts.medication_classes is None:
            return None
        if condition["med_class"] in facts.medication_classes:
            return True
        return None if not facts.medications_complete or condition["med_class"] in facts.uncertain_classes else False
    for key in _FLAG_KEYS:
        if key in condition:
            return None if facts.flags is None else facts.flags.get(f"{key}:{condition[key]}")
    raise CorpusRuleError(f"unsupported condition keys: {sorted(condition)}")
