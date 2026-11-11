"""Stage 2 of corpus use: evaluate diagnosis/workup/referral/monitoring rules against EMR facts.

Rules come from the same generated corpus as the evidence passages
(``text_data/corpus`` or the active PostgreSQL release) and pass the same gate as cited
evidence (``clinical_evidence.is_citable``): every source licensed for commercial AI use,
checked against the primary text, not retired. Clinician-approved rules are always used;
draft rules only when ``include_unreviewed`` is enabled. Every result carries its sources and
``review_status`` (data only — no warning text: the service is a reference tool by design).

Stage 3 (``doses=True``): ``dose_rules`` are applied to the current prescriptions and to the
drugs the clinician names (``dose_drugs``) — weight-based formulas, the renal-function band
the patient falls in, dose-condition rules, and the prescribed daily amount against a numeric
band maximum (``clinical_dosing``). Values are reference calculations that always carry their
inputs, formula and sources. Treatment choice (``treatment.jsonl``) stays out.

Evaluation is three-valued (``clinical_corpus.evaluate``). A rule is reported as
- ``met``: the condition is true for the facts we have;
- ``possible``: the condition is unknown, but part of it is already true for this patient
  (an anchor such as a recorded diagnosis, a prescription or a lab value). The unknown
  inputs are returned as ``missing`` so the clinician can confirm them.
Rules that are false, or unknown with nothing anchoring them to the patient, are dropped.

Modules: ``corpus`` (gated records), ``rules`` (stage-2 decisions), ``doses`` (stage 3),
``presentation`` (payload entries and labels), ``model`` (input/result types).
"""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import Any, Iterable, Mapping

from ax_g_ai.corpus.files import DOSE_FILE
from ax_g_ai.services.clinical_corpus import CorpusRuleError
from ax_g_ai.services.clinical_recommendation.corpus import RuleCorpus, read_rows
from ax_g_ai.services.clinical_recommendation.doses import DoseStage
from ax_g_ai.services.clinical_recommendation.model import Assessment, EvidenceNumbering, Leaf, RuleInput, prompt_view
from ax_g_ai.services.clinical_recommendation.presentation import Presenter
from ax_g_ai.services.clinical_recommendation.rules import (
    FLAG_KINDS, accepted_findings, derive_facts, evaluate_record, flag_vocabulary, rank_item,
)
from ax_g_ai.services.emr_mapping import SiteCodeMap, map_emr

__all__ = ["FLAG_KINDS", "Assessment", "ClinicalRecommender", "Leaf", "RuleCorpus", "RuleInput", "prompt_view", "read_rows"]

logger = logging.getLogger("uvicorn.error")


class ClinicalRecommender:
    """Evaluate gated corpus rules against an EMR mapping and the clinician's findings."""

    def __init__(
        self, corpus: RuleCorpus, site_map: SiteCodeMap | None = None, *,
        max_possible: int = 12, max_evidence: int = 10, evidence_per_item: int = 1, dur: Any = None,
        doses: bool = False, max_dose_items: int = 12, max_dose_evidence: int = 6,
    ) -> None:
        self._corpus = corpus
        self._dur = dur  # services.dur_check.DurChecker
        self._site_map = site_map
        self._max_possible = max_possible
        self._max_evidence = max_evidence
        self._evidence_per_item = evidence_per_item
        self._doses = doses
        self._present = Presenter(corpus)
        self._dose_stage = DoseStage(
            corpus, self._present, corpus.rules.get(DOSE_FILE, ()) if doses else (),
            max_items=max_dose_items, max_evidence=max_dose_evidence,
        )
        self._vocabulary = flag_vocabulary(
            record for name, records in corpus.rules.items() if name != DOSE_FILE or doses for record in records
        )

    def __len__(self) -> int:
        return len(self._corpus)

    @property
    def unreviewed_count(self) -> int:
        return self._corpus.unreviewed_count

    @property
    def dose_rule_count(self) -> int:
        return self._dose_stage.rule_count

    @property
    def dose_drugs(self) -> tuple[str, ...]:
        """Ingredient / class ids that have a dose rule (for ``dose_drugs`` input); empty when stage 3 is off."""
        return tuple(sorted(self._dose_stage.targets))

    @property
    def finding_vocabulary(self) -> tuple[str, ...]:
        """Finding/event keys the loaded rules can test (for clinician input or note extraction)."""
        return tuple(sorted(self._vocabulary))

    def prepare(self, source: Mapping[str, Any]) -> RuleInput:
        """Map the raw Bridge EMR object (not the envelope) onto corpus concepts."""
        mapping = map_emr(source, self._corpus.catalog, self._site_map)
        context = source.get("calculationContext")
        context = context if isinstance(context, Mapping) else {}
        sex = context.get("sex") if context.get("sex") in {"male", "female"} else None
        values = dict(mapping.facts.values)
        weight, height = context.get("weight_kg"), context.get("height_cm")
        if "BMI" not in values and _plausible(weight, 0.3, 500) and _plausible(height, 30, 300):
            values["BMI"] = round(float(weight) / (float(height) / 100) ** 2, 1)  # type: ignore[arg-type]
            mapping = replace(mapping, facts=replace(mapping.facts, values=values))
        medications = tuple(
            (_text_or_none(item.get("medCode")), _text_or_none(item.get("medName")))
            for item in source.get("medicineDtailList") or () if isinstance(item, Mapping)
        )
        return RuleInput(mapping, sex, medications)

    def assess(
        self, rule_input: RuleInput, findings: Mapping[str, bool] | None = None, dose_drugs: Iterable[str] = (),
    ) -> Assessment:
        mapping, sex = rule_input.mapping, rule_input.sex
        accepted, unknown = accepted_findings(findings or {}, self._vocabulary)
        facts = derive_facts(mapping.facts, accepted, sex, self._vocabulary)
        numbering = EvidenceNumbering(self._corpus.chunks_by_claim, self._max_evidence, self._evidence_per_item)
        candidates: list[tuple[tuple, dict[str, Any], Mapping[str, Any]]] = []
        for name in ("diagnosis.jsonl", "referral.jsonl", "workup.jsonl", "monitoring.jsonl"):
            for record in self._corpus.rules.get(name, ()):
                try:
                    result = evaluate_record(record, facts)
                except CorpusRuleError as error:
                    logger.warning("clinical_rule_unsupported record_id=%s error=%s", record.get("id"), error)
                    continue
                if result is not None:
                    status, leaves, extra = result
                    item = self._present.item(record, status, leaves, facts, extra)
                    candidates.append((rank_item(item, leaves), item, record))
        candidates.sort(key=lambda entry: entry[0])
        items: list[dict[str, Any]] = []
        possible = references = 0
        for _rank_key, item, record in candidates:
            if item["priority"] == "reference":
                references += 1
                if references > self._max_possible:
                    continue
            elif item["status"] == "possible":
                possible += 1
                if possible > self._max_possible:
                    continue
            item["evidence_refs"] = numbering.refs(record.get("claim_id"))
            items.append(item)
        labs = []
        interpretations = sorted(self._present.lab_interpretations(mapping, facts, sex), key=lambda pair: pair[0]["priority"] != "primary")
        for interpretation, record in interpretations:
            interpretation["evidence_refs"] = numbering.refs(record.get("claim_id"))
            labs.append(interpretation)
        issues = tuple(
            {key: value for key, value in vars(issue).items() if value is not None} for issue in mapping.issues
        )
        alerts = None
        if self._dur is not None:
            pregnant = accepted.get("pregnancy")
            alerts = self._dur.check(rule_input.medications, age_years=facts.values.get("age"), sex=sex, pregnant=pregnant).as_dict()
        doses: tuple[Mapping[str, Any], ...] | None = None
        unmatched: tuple[Mapping[str, str], ...] = ()
        if self._doses:
            requested, unmatched = self._dose_stage.resolve(dose_drugs)
            doses = tuple(self._dose_stage.assessments(mapping, facts, requested, numbering))
        return Assessment(tuple(items), tuple(labs), tuple(numbering.evidence), dict(accepted), issues, tuple(unknown), alerts,
                          doses, unmatched)


def _text_or_none(value: object) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _plausible(value: object, low: float, high: float) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and low <= float(value) <= high
