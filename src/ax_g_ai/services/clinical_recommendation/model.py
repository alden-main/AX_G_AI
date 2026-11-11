"""Input and result types of the recommender, and the numbering of cited passages."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from typing import Any, Iterable, Mapping

from ax_g_ai.services.clinical_evidence import EvidenceChunk
from ax_g_ai.services.clinical_recommendation.schema import AssessmentPayload
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
    """Structured stage-2 result. ``evidence`` numbering matches ``evidence_refs``."""

    items: tuple[Mapping[str, Any], ...]
    lab_interpretations: tuple[Mapping[str, Any], ...]
    evidence: tuple[EvidenceChunk, ...]
    findings_used: Mapping[str, bool]
    mapping_issues: tuple[Mapping[str, Any], ...] = ()
    unknown_findings: tuple[str, ...] = ()
    # HIRA DUR contraindication/caution alerts for the current prescriptions (None: DUR not configured)
    medication_alerts: Mapping[str, Any] | None = None
    # Stage-3 dose results (None: dose stage disabled) and requested drugs without a dose rule
    dose_assessments: tuple[Mapping[str, Any], ...] | None = None
    unmatched_dose_drugs: tuple[Mapping[str, str], ...] = ()

    @property
    def unreviewed(self) -> bool:
        return any(
            item.get("review_status") != "approved"
            for item in (*self.items, *self.lab_interpretations, *(self.dose_assessments or ()))
        )

    @cached_property
    def payload(self) -> AssessmentPayload:
        """The validated payload (``schema.AssessmentPayload``); raises ``ValidationError`` on an unexpected key or type."""
        return AssessmentPayload.model_validate(self._raw_dict())

    def as_dict(self) -> dict[str, object]:
        """JSON-ready payload with exactly the keys that were set (a fresh dict: callers may edit it)."""
        return self.payload.model_dump(mode="json")

    def _raw_dict(self) -> dict[str, object]:
        doses: dict[str, object] = {}
        if self.dose_assessments is not None:
            doses["dose_assessments"] = [dict(item) for item in self.dose_assessments]
            if self.unmatched_dose_drugs:
                doses["unmatched_dose_drugs"] = [dict(item) for item in self.unmatched_dose_drugs]
        return {
            "items": [dict(item) for item in self.items],
            "lab_interpretations": [dict(item) for item in self.lab_interpretations],
            "findings_used": dict(self.findings_used),
            "unknown_findings": list(self.unknown_findings),
            "mapping_issues": [dict(issue) for issue in self.mapping_issues],
            **({"medication_alerts": dict(self.medication_alerts)} if self.medication_alerts is not None else {}),
            **doses,
        }


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


def unique_missing(items: Iterable[dict[str, str]]) -> list[dict[str, str]]:
    seen: set[tuple[str, str]] = set()
    result = []
    for item in items:
        marker = (item["kind"], item["key"])
        if marker not in seen:
            seen.add(marker)
            result.append(item)
    return result
