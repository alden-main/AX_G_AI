"""Stage-2 rule decisions: which records are ``met`` or ``possible`` for the patient, and in which order.

Evaluation itself is three-valued (``clinical_corpus.evaluate``). A rule is ``met`` when its condition
is true; ``possible`` when it is unknown but anchored to the patient (a recorded diagnosis, a
prescription, a lab value or a clinician finding); otherwise it is dropped.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import replace
from typing import Any, Iterable, Mapping

from ax_g_ai.services.clinical_corpus import PatientFacts, evaluate
from ax_g_ai.services.clinical_recommendation.model import Leaf

FLAG_KINDS = ("finding", "event", "risk", "indication", "lab_present", "dx_history")
_KIND_ORDER = {"diagnosis": 0, "referral": 1, "workup": 2, "monitoring": 3}
_STATUS_ORDER = {"met": 0, "possible": 1}
# Diabetes history derivable from recorded KCD codes (presence only; absence stays unknown).
_DIABETES_KCD = ("E10", "E11", "E12", "E13", "E14")


def evaluate_record(
    record: Mapping[str, Any], facts: PatientFacts
) -> tuple[str, list[Leaf], dict[str, Any]] | None:
    kind = record.get("kind")
    extra: dict[str, Any] = {}
    if kind == "diagnosis":
        condition = record.get("logic")
        if not isinstance(condition, Mapping):
            return None  # criteria text only — not machine-evaluable
    elif kind == "workup":
        condition = record.get("trigger")
    elif kind == "referral":
        criteria = record.get("criteria")
        condition = {"any": list(criteria)} if isinstance(criteria, list) else criteria
    elif kind == "monitoring":
        return evaluate_monitoring(record, facts)
    else:
        return None
    if not isinstance(condition, Mapping):
        return None
    status = rule_status(condition, facts)
    if status is None:
        return None
    return status[0], status[1], extra


def evaluate_monitoring(
    record: Mapping[str, Any], facts: PatientFacts
) -> tuple[str, list[Leaf], dict[str, Any]] | None:
    after = record.get("after")
    rules = [rule for rule in record.get("rules") or () if isinstance(rule, Mapping)]
    if isinstance(after, Mapping):
        # ``after`` mixes keys (e.g. event + med_class): every key must hold.
        condition = {"all": [{key: value} for key, value in after.items()]}
        status = rule_status(condition, facts)
        if status is None:
            return None
        actions = [rule.get("action") for rule in rules
                   if isinstance(rule.get("when"), Mapping) and evaluate(rule["when"], facts) is True and rule.get("action")]
        return status[0], status[1], {"actions": actions}
    fired = []
    leaves: list[Leaf] = []
    for rule in rules:
        when = rule.get("when")
        if not isinstance(when, Mapping) or when.get("always") is True:
            continue
        result = rule_status(when, facts)
        if result is not None and result[0] == "met":
            fired.append(rule.get("action"))
            leaves.extend(result[1])
    if not fired:
        return None  # interval-only monitoring without an anchor stays silent
    return "met", leaves, {"actions": [action for action in fired if action]}


def accepted_findings(findings: Mapping[str, bool], vocabulary: Mapping[str, tuple[str, ...]]) -> tuple[dict[str, bool], list[str]]:
    accepted: dict[str, bool] = {}
    unknown: list[str] = []
    for raw, value in findings.items():
        key = raw.split(":", 1)[1] if ":" in raw and raw.split(":", 1)[0] in FLAG_KINDS else raw
        if key in vocabulary and isinstance(value, bool):
            accepted[key] = value
        else:
            unknown.append(raw)
    return accepted, unknown


def derive_facts(
    base: PatientFacts, findings: Mapping[str, bool], sex: str | None, vocabulary: Mapping[str, tuple[str, ...]],
) -> PatientFacts:
    flags: dict[str, bool] = dict(base.flags or {})
    if sex in {"male", "female"}:
        flags.setdefault("finding:male", sex == "male")
    codes = base.diagnosis_codes or frozenset()
    if any(code.startswith(_DIABETES_KCD) for code in codes):
        flags.setdefault("dx_history:diabetes", True)
    for key, value in findings.items():
        for kind in vocabulary.get(key, ()):
            flags[f"{kind}:{key}"] = value  # clinician input overrides derived flags
    return replace(base, flags=flags)


def rule_status(condition: Mapping[str, Any], facts: PatientFacts) -> tuple[str, list[Leaf]] | None:
    result = evaluate(condition, facts)
    leaves = list(condition_leaves(condition, facts, negated=False))
    if result is True:
        return "met", leaves
    if result is not None:
        return None
    anchors = [leaf for leaf in leaves if leaf.result is True and not leaf.negated]
    if any(not is_weak_anchor(leaf) for leaf in anchors):
        return "possible", leaves
    # Only age/sex hold: report screening rules whose sole unknowns are "not already done"
    # (e.g. adult HBV screening without a prior panel), never symptom-driven rules.
    if anchors and all(leaf.negated for leaf in leaves if leaf.result is None):
        return "possible", leaves
    return None


def is_weak_anchor(leaf: Leaf) -> bool:
    """Demographic anchors (age, derived sex) that alone do not tie a rule to this patient."""
    return (leaf.kind == "var" and leaf.key == "age") or (leaf.kind == "finding" and leaf.key == "male")


def condition_leaves(condition: Mapping[str, Any], facts: PatientFacts, *, negated: bool) -> Iterable[Leaf]:
    if "all" in condition or "any" in condition:
        children = [child for child in condition.get("all") or condition.get("any") or () if isinstance(child, Mapping)]
        if "any" in condition and evaluate(condition, facts) is True:
            # Already satisfied: unknown alternatives are irrelevant (not missing inputs).
            children = [child for child in children if evaluate(child, facts) is True]
        for child in children:
            yield from condition_leaves(child, facts, negated=negated)
        return
    if "not" in condition:
        if isinstance(condition["not"], Mapping):
            yield from condition_leaves(condition["not"], facts, negated=not negated)
        return
    if condition.get("always") is True:
        return
    kind = next((key for key in ("var", "dx", "med", "med_class", *FLAG_KINDS) if key in condition), None)
    if kind is None:
        return
    yield Leaf(kind, str(condition[kind]), condition, evaluate(condition, facts), negated)


def rank_item(item: Mapping[str, Any], leaves: list[Leaf]) -> tuple:
    anchors = [leaf for leaf in leaves if leaf.result is True and not leaf.negated]
    # Anchors other than age/sex (diagnosis, prescription, lab value, clinician finding) rank first.
    strong = sum(1 for leaf in anchors if not is_weak_anchor(leaf))
    known = sum(1 for leaf in leaves if leaf.result is not None)
    return (
        0 if item.get("priority") == "primary" else 1,  # domestic standard (or no competing one) first
        _STATUS_ORDER.get(item["status"], 9), 0 if strong else 1, _KIND_ORDER.get(item["kind"], 9),
        -strong, -(known / len(leaves) if leaves else 0), str(item.get("record_id")),
    )


def flag_vocabulary(records: Iterable[Mapping[str, Any]]) -> dict[str, tuple[str, ...]]:
    found: dict[str, set[str]] = defaultdict(set)

    def walk(value: object) -> None:
        if isinstance(value, Mapping):
            for kind in FLAG_KINDS:
                if isinstance(value.get(kind), str):
                    found[value[kind]].add(kind)
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    for record in records:
        for key in ("logic", "trigger", "criteria", "after", "rules", "initiation"):
            walk(record.get(key))
    return {key: tuple(sorted(kinds)) for key, kinds in found.items() if re.fullmatch(r"[a-z0-9_]+", key)}
