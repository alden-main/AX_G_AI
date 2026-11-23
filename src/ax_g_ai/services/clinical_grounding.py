"""What a chat answer or a recommendation is grounded on: cited evidence and the rule assessment.

Both use the request's corpus snapshot, so the passages, the rule results and DUR come from one release.

- Chat (``chat_grounding``): evidence for the question, except one that only asks for recorded values
  (``question_facts.is_record_lookup``); for a question about diagnoses, tests, drugs or
  doses (or one that names a drug or states pregnancy) also the stage-2/3 rule results, whose evidence
  comes first so their ``[근거 n]`` references stay valid.
- Recommendation (``recommendation_grounding``): the clinician's explicit findings over the memo (its
  LLM extraction over the deterministic reading) over the question, then the full assessment.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

from ax_g_ai.errors import RecommendationUnavailableError
from ax_g_ai.services.clinical_chat import ClinicalChatService
from ax_g_ai.services.clinical_evidence import EvidenceChunk
from ax_g_ai.services.clinical_recommendation import Assessment, ClinicalRecommender
from ax_g_ai.services.corpus_snapshot import CorpusSnapshot
from ax_g_ai.services.patient_context_store import PreparedRawEmrContext
from ax_g_ai.services.question_facts import asks_for_recommendation, is_record_lookup, merge_question_findings, question_findings

logger = logging.getLogger("uvicorn.error")

# Rule evidence passages put in front of a chat answer's own evidence.
RULE_EVIDENCE_IN_CHAT = 4


@dataclass(frozen=True)
class Grounding:
    evidence: tuple[EvidenceChunk, ...]
    assessment: dict[str, object] | None  # assessment payload (``docs/assessment_schema.md``), None: not used


def require_recommender(snapshot: CorpusSnapshot) -> ClinicalRecommender:
    if snapshot.recommender is None or len(snapshot.recommender) == 0:
        raise RecommendationUnavailableError()
    return snapshot.recommender


def with_rule_input(context: PreparedRawEmrContext, snapshot: CorpusSnapshot) -> PreparedRawEmrContext:
    """Rule facts mapped with this snapshot's rules: kept when the release matches, else mapped again."""
    recommender = snapshot.recommender
    if recommender is None or (context.rule_input is not None and context.rule_release == snapshot.release_id):
        return context
    try:
        rule_input = recommender.prepare(context.rule_source)
    except Exception as error:  # rule facts are optional; never block EMR preparation or chat
        logger.warning("clinical_rule_mapping_failed error_type=%s", type(error).__name__)
        return replace(context, rule_input=None)
    return replace(context, rule_input=rule_input, rule_release=snapshot.release_id)


def chat_grounding(question: str, context: PreparedRawEmrContext, service: ClinicalChatService, snapshot: CorpusSnapshot) -> Grounding:
    """Evidence (and, for diagnosis/test/medication questions, rule results) for one chat question.

    Synchronous: retrieval may call the embedding server and database, so callers run it in a thread.
    """
    searched: tuple[EvidenceChunk, ...] = ()
    if not is_record_lookup(question):
        searched = service.find_evidence(question, snapshot.evidence)
        if not searched and asks_for_recommendation(question):
            # A generic recommendation question ("추천 검사는?") has no topic words: add the recorded diagnoses.
            # Only then — for other questions the diagnoses pull in passages unrelated to what was asked.
            hint = diagnosis_hint(context)
            if hint:
                searched = service.find_evidence(f"{question} {hint}", snapshot.evidence)
    recommender = snapshot.recommender
    if recommender is None or context.rule_input is None:
        return Grounding(searched, None)
    stated = recommender.question_inputs(question)
    if not stated and not asks_for_recommendation(question):
        return Grounding(searched, None)
    assessment = recommender.assess(context.rule_input, stated.findings, stated.dose_drugs, stated.proposed_drugs)
    alerts = (assessment.medication_alerts or {}).get("alerts")
    if not assessment.items and not assessment.lab_interpretations and not alerts and not assessment.dose_assessments:
        return Grounding(searched, None)
    rule_chunks = assessment.evidence[:RULE_EVIDENCE_IN_CHAT]
    return Grounding(merge_evidence(rule_chunks, searched), _renumber(assessment, len(rule_chunks)))


async def recommendation_grounding(
    context: PreparedRawEmrContext, service: ClinicalChatService, snapshot: CorpusSnapshot, *,
    clinical_note: str | None, question: str | None, findings: Mapping[str, bool] | None, dose_drugs: Sequence[str] | None,
) -> tuple[PreparedRawEmrContext, Grounding]:
    """The full assessment for a recommendation request; raises ``RecommendationUnavailableError`` without rules."""
    rules = require_recommender(snapshot)
    context = with_rule_input(context, snapshot)
    if context.rule_input is None:
        # EMR mapping failed: evaluate with no EMR facts.
        context = replace(context, rule_input=rules.prepare({}))
    query = " ".join(part for part in (question, clinical_note, diagnosis_hint(context)) if part)
    # The note extraction (an LLM call) and the evidence search do not depend on each other: run them together.
    extracted, searched = await asyncio.gather(
        service.extract_findings(note=clinical_note, vocabulary=rules.finding_vocabulary) if clinical_note else _value({}),
        asyncio.to_thread(service.find_evidence, query, snapshot.evidence) if query else _value(()),
    )
    stated = rules.question_inputs(clinical_note, question)
    # Explicit clinician input wins over the memo (its extraction over the deterministic reading), which wins over
    # the question text unless the question states a change since the memo ("출산 후"). A disagreement without such
    # a cue is left unknown and reported as a conflict to confirm.
    note_findings, conflicts = merge_question_findings({**question_findings(clinical_note), **extracted}, question)
    merged: dict[str, bool] = {**note_findings, **(findings or {})}
    drugs = tuple(dict.fromkeys((*(dose_drugs or ()), *stated.dose_drugs)))
    assessment = rules.assess(context.rule_input, merged, drugs, stated.proposed_drugs, conflicts)  # type: ignore[arg-type]
    evidence = merge_evidence(assessment.evidence, searched)
    logger.info(
        "clinical_recommendation_assessed items=%d met=%d findings=%d unknown_findings=%d evidence=%d unreviewed=%s doses=%d",
        len(assessment.items), sum(1 for item in assessment.items if item.status == "met"),
        len(assessment.findings_used), len(assessment.unknown_findings), len(evidence), assessment.unreviewed,
        len(assessment.dose_assessments or ()),
    )
    return context, Grounding(evidence, assessment.as_dict())


def diagnosis_hint(context: PreparedRawEmrContext) -> str:
    """The recorded diagnosis names, as extra search words for a question without a topic."""
    diagnoses = context.data.get("diagnoses")
    names = [item.get("name_ko") for item in diagnoses if isinstance(item, Mapping)] if isinstance(diagnoses, list) else []
    return " ".join(name for name in names if isinstance(name, str))[:300]


def merge_evidence(first: Sequence[EvidenceChunk], second: Sequence[EvidenceChunk]) -> tuple[EvidenceChunk, ...]:
    """Concatenate keeping ``first``'s numbering (rule evidence refs point into it)."""
    seen = {chunk.chunk_id for chunk in first}
    return (*first, *(chunk for chunk in second if chunk.chunk_id not in seen))


def _renumber(assessment: Assessment, kept: int) -> dict[str, object]:
    """Assessment payload whose evidence refs only point at the first ``kept`` passages."""
    payload = assessment.as_dict()
    for group in ("items", "lab_interpretations", "dose_assessments"):
        items = payload.get(group)
        for item in items if isinstance(items, list) else ():
            item["evidence_refs"] = [ref for ref in item.get("evidence_refs", ()) if ref <= kept]
    return payload


async def _value[T](value: T) -> T:
    return value
