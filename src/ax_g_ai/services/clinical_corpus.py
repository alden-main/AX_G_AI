"""Patient facts and the three-valued evaluator for corpus rule conditions.

Records are gated before they reach this module (``clinical_evidence.is_citable``, applied
by ``clinical_recommendation.RuleCorpus``).

Rule conditions use three-valued logic: a fact absent from the patient context
evaluates to ``None`` (unknown), never ``False``. Callers must treat ``None`` as
"input needed", mirroring ``clinical_calculations`` which never infers inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

_OPERATORS = {
    "<": lambda a, b: a < b, "<=": lambda a, b: a <= b,
    ">": lambda a, b: a > b, ">=": lambda a, b: a >= b, "==": lambda a, b: a == b,
}
# Condition keys answered from a boolean flag map; absent flags are unknown.
_FLAG_KEYS = ("finding", "event", "risk", "indication", "lab_present", "dx_history")


class CorpusRuleError(ValueError):
    """A corpus condition uses an operator or key this evaluator does not support."""


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
