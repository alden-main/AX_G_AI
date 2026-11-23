"""Input and result types of the recommender, and the numbering of cited passages."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any

from ax_g_ai.services.clinical_evidence import EvidenceChunk
from ax_g_ai.services.clinical_recommendation.schema import (
    AssessmentPayload,
    DoseAssessment,
    FindingConflict,
    LabInterpretation,
    MappingIssue,
    MedicationAlerts,
    MissingInput,
    RecommendationItem,
    UnmatchedDoseDrug,
)
from ax_g_ai.services.emr_mapping import EmrMapping


@dataclass(frozen=True)
class Leaf:
    kind: str
    key: str
    condition: Mapping[str, Any]
    result: bool | None
    negated: bool


@dataclass(frozen=True)
class RuleInput:
    """EMR facts for rule evaluation, computed once when the patient context is prepared.

    Holds concept ids, values, KCD codes and ingredient ids only — no identifiers.
    """

    mapping: EmrMapping
    sex: str | None = None
    # (medCode, medName) of current prescriptions — product codes for the DUR check
    medications: tuple[tuple[str | None, str | None], ...] = ()


@dataclass(frozen=True)
class Assessment:
    """One patient's rule evaluation: the payload parts (``schema`` models) and the passages they cite.

    ``evidence`` numbering matches ``evidence_refs``. ``payload`` assembles the API/prompt payload from the parts.
    """

    items: tuple[RecommendationItem, ...]
    lab_interpretations: tuple[LabInterpretation, ...]
    evidence: tuple[EvidenceChunk, ...]
    findings_used: Mapping[str, bool]
    mapping_issues: tuple[MappingIssue, ...] = ()
    unknown_findings: tuple[str, ...] = ()
    # HIRA DUR contraindication/caution alerts for the current prescriptions (None: DUR not configured)
    medication_alerts: MedicationAlerts | None = None
    # Stage-3 dose results (None: dose stage disabled) and requested drugs without a dose rule
    dose_assessments: tuple[DoseAssessment, ...] | None = None
    unmatched_dose_drugs: tuple[UnmatchedDoseDrug, ...] = ()
    # Findings the memo and the question state differently, left out of findings_used: {"pregnancy": {"clinical_note": True, "question": False}}
    finding_conflicts: Mapping[str, Mapping[str, bool]] = field(default_factory=dict)

    @property
    def unreviewed(self) -> bool:
        return (any(item.review_status != "approved" for item in self.items)
                or any(lab.review_status != "approved" for lab in self.lab_interpretations)
                or any(dose.review_status != "approved" for dose in self.dose_assessments or ()))

    @cached_property
    def payload(self) -> AssessmentPayload:
        """``schema.AssessmentPayload``: the parts as they are, and the optional groups only when they apply."""
        optional: dict[str, Any] = {}
        if self.medication_alerts is not None:
            optional["medication_alerts"] = self.medication_alerts
        if self.dose_assessments is not None:
            optional["dose_assessments"] = list(self.dose_assessments)
            if self.unmatched_dose_drugs:
                optional["unmatched_dose_drugs"] = list(self.unmatched_dose_drugs)
        if self.finding_conflicts:
            # Produced by ``question_facts.merge_question_findings`` (plain data shared with the chat path).
            optional["finding_conflicts"] = {key: FindingConflict.model_validate(value) for key, value in self.finding_conflicts.items()}
        return AssessmentPayload(
            items=list(self.items), lab_interpretations=list(self.lab_interpretations), findings_used=dict(self.findings_used),
            unknown_findings=list(self.unknown_findings), mapping_issues=list(self.mapping_issues), **optional,
        )

    def as_dict(self) -> dict[str, object]:
        """JSON-ready payload with exactly the keys that were set (a fresh dict: callers may edit it)."""
        return self.payload.model_dump(mode="json")


# The answer attributes content to its sources (source_titles, region_label); ids and review state are not repeated.
_PROMPT_DROP = frozenset({"claim_id", "chapter", "review_status", "record_id", "mapping_issues"})


def prompt_view(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Assessment payload for the LLM: ids and review state are left out, and full source records (shown on screen)
    shrink to ``source_titles`` so the answer can say which source each statement is based on."""

    def strip(value: Any) -> Any:
        if isinstance(value, Mapping):
            result = {}
            for key, item in value.items():
                if key in _PROMPT_DROP or item in (None, [], {}):
                    continue
                if key == "sources" and isinstance(item, list):
                    titles = [source["title"] for source in item if isinstance(source, Mapping) and source.get("title")]
                    if titles:
                        result["source_titles"] = list(dict.fromkeys(titles))
                    continue
                result[key] = strip(item)
            return result
        if isinstance(value, list):
            return [strip(item) for item in value]
        return value

    return strip(payload)


@dataclass
class EvidenceNumbering:
    """Passages cited by one assessment, numbered in order of first use (``evidence_refs`` = [근거 n])."""

    chunks_by_claim: Mapping[str, tuple[EvidenceChunk, ...]]
    max_evidence: int
    per_item: int = 1
    evidence: list[EvidenceChunk] = field(default_factory=list)
    numbers: dict[str, int] = field(default_factory=dict)

    def refs(self, claim_id: object, limit: int | None = None) -> list[int]:
        refs: list[int] = []
        if not isinstance(claim_id, str):
            return refs
        for chunk in self.chunks_by_claim.get(claim_id, ())[: self.per_item]:
            number = self.numbers.get(chunk.chunk_id)
            if number is None:
                if len(self.evidence) >= (self.max_evidence if limit is None else limit):
                    continue
                self.evidence.append(chunk)
                number = self.numbers[chunk.chunk_id] = len(self.evidence)
            refs.append(number)
        return refs


def unique_missing(items: Iterable[MissingInput]) -> list[MissingInput]:
    """The first of each (kind, key), in order of first appearance: one rule can test an input in several conditions."""
    unique: dict[tuple[str, str], MissingInput] = {}
    for item in items:
        unique.setdefault((item.kind, item.key), item)
    return list(unique.values())
