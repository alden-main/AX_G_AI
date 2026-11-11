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
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable, Mapping

from ax_g_ai.corpus.files import CHUNKS_FILE, CONCEPT_FILES, DOSE_FILE, RULE_FILES
from ax_g_ai.corpus.gate import is_citable
from ax_g_ai.jsonl import read_jsonl
from ax_g_ai.services.clinical_corpus import CorpusRuleError, PatientFacts, evaluate
from ax_g_ai.services.clinical_dosing import (
    MissingInput, band_daily_max_mg, evaluate_formula, formula_variables, prescribed_daily_mg, range_label, rounded,
    select_band, unit_for,
)
from ax_g_ai.services.clinical_evidence import (
    REGION_LABELS, EvidenceChunk, chunk_from_record, mark_superseded, region_of, sorted_sources,
)
from ax_g_ai.services.emr_mapping import ConceptCatalog, EmrMapping, MedicationMatch, SiteCodeMap, map_emr

logger = logging.getLogger("uvicorn.error")

FLAG_KINDS = ("finding", "event", "risk", "indication", "lab_present", "dx_history")
_LOGICAL = ("all", "any", "not")
_KIND_ORDER = {"diagnosis": 0, "referral": 1, "workup": 2, "monitoring": 3}
_STATUS_ORDER = {"met": 0, "possible": 1}
# Diabetes history derivable from recorded KCD codes (presence only; absence stays unknown).
_DIABETES_KCD = ("E10", "E11", "E12", "E13", "E14")
_CATEGORY = {
    "labs": "검체검사(혈액·소변 등)", "panels": "검체검사 묶음", "derived": "계산 지표", "scores": "점수·위험도 평가",
    "vitals": "활력징후·신체계측", "measurements": "측정", "clinical": "임상 평가", "tasks": "평가 항목", "med_derived": "약물 관련 지표",
}
_EXAM_CATEGORY = {"imaging": "영상검사", "endoscopy": "내시경검사", "biopsy": "조직검사", "function": "기능·진찰 검사"}
_TEST_KEY_LABEL = {"review": "검토", "calc": "계산", "score": "점수·위험도 평가", "screen": "선별 평가"}

# ── stage 3 (dose) presentation ──────────────────────────────────────────────
ADULT_AGE = 18
# Record keys the dose stage evaluates or shows elsewhere; the rest is shown as reference text.
_DOSE_HANDLED = frozenset({
    "id", "claim_id", "chapter", "topic", "ai_tier", "claim_status", "verification", "review_status", "usable", "sources",
    "conflicts", "kind", "source_priority", "concepts", "indication", "ingredient", "ingredient_class", "rules", "initiation",
    "bands", "calculation", "renal_param", "renal_method", "alert_note", "preferred_over", "region",
})
_DOSE_LABELS = {
    "standard_dose": "표준 용량", "starting_dose": "시작 용량", "titration": "증량", "maintenance": "유지 용량",
    "max_dose": "최대 용량", "target": "목표", "target_daily_mg": "목표 1일 용량(mg)", "monitoring": "감시",
    "renal": "신기능", "renal_adjustment": "신기능 조정", "hepatic": "간기능 조정", "administration": "투여 방법",
    "taper": "감량", "duration": "기간", "max_duration": "최대 기간", "pretreatment": "시작 전 검사", "regimens": "요법",
    "contraindications": "금기", "route": "투여 경로", "bolus": "부하 용량", "infusion": "주입",
    "maintenance_mg_weekly": "주간 유지 용량(mg)", "max_mg_weekly": "주간 최대 용량(mg)", "sick_day": "아픈 날 수칙",
    "interactions": "상호작용", "pediatric": "소아", "diluent_volume_ml_ge41kg": "희석 용량(41 kg 이상, mL)",
    "setting": "투여 환경", "timing": "시점", "limit": "한도", "options": "선택지", "equivalence": "등가 용량",
    "other_holds": "중단 고려", "other": "기타", "induction": "유도", "iv": "정주", "nasal": "비강",
    "short_acting": "속효성", "therapeutic_doses": "치료 용량", "intensity_table": "강도표", "max_total_g_day": "1일 최대 총량(g)",
}
_BAND_LABELS = {
    "start": "시작", "titration": "증량", "max_daily_mg": "1일 최대(mg)", "daily_mg": "1일 용량(mg)", "regimen": "분할",
    "action": "용량·조치", "usual_0_5": "통상(0.5 mg 요법)", "refractory_or_decompensated_1mg": "불응·비대상성(1 mg 요법)",
}
_ALERT_ORDER = {"contraindicated": 0, "high": 1, "caution": 2, "medium": 2, "info": 3}
_VALUE_HINTS = {
    "CrCl": "크레아티닌 청소율(Cockcroft-Gault) — 연령·성별·체중·혈청 크레아티닌이 있으면 계산",
    "eGFR": "추정 사구체여과율 — 연령·성별·혈청 크레아티닌이 있으면 계산",
    "weight_kg": "체중(kg)",
    "age": "연령",
}


@dataclass(frozen=True)
class Leaf:
    kind: str
    key: str
    condition: Mapping[str, Any]
    result: bool | None
    negated: bool


@dataclass(frozen=True)
class RuleCorpus:
    """Gated rule records, the evidence passages of their claims and the concept catalog."""

    rules: Mapping[str, tuple[Mapping[str, Any], ...]]
    chunks_by_claim: Mapping[str, tuple[EvidenceChunk, ...]]
    catalog: ConceptCatalog
    skipped: int = 0
    # claim or record id → claims of loaded higher-priority (domestic) records that supersede it
    superseded: Mapping[str, tuple[str, ...]] = field(default_factory=dict)

    @classmethod
    def from_rows(
        cls, rows: Mapping[str, Iterable[Mapping[str, Any]]], *, include_unreviewed: bool, require_license: bool = False,
    ) -> "RuleCorpus":
        def citable(record: Mapping[str, Any]) -> bool:
            return is_citable(record, include_unreviewed=include_unreviewed, require_license=require_license)

        rules: dict[str, tuple[Mapping[str, Any], ...]] = {}
        skipped = 0
        for name in RULE_FILES:
            kept = []
            for record in rows.get(name, ()):
                if citable(record):
                    kept.append(record)
                else:
                    skipped += 1
            rules[name] = tuple(kept)
        chunks: dict[str, list[EvidenceChunk]] = defaultdict(list)
        for chunk in mark_superseded(chunk_from_record(record) for record in rows.get(CHUNKS_FILE, ()) if citable(record)):
            chunks[chunk.claim_id].append(chunk)
        superseded: dict[str, list[str]] = defaultdict(list)
        for records in rules.values():
            for record in records:
                for target in record.get("preferred_over") or ():
                    if target != record.get("claim_id") and record.get("claim_id") not in superseded[target]:
                        superseded[target].append(record.get("claim_id"))
        catalog = ConceptCatalog.from_rows(*(rows.get(name, ()) for name in CONCEPT_FILES))
        return cls(rules, {claim: tuple(items) for claim, items in chunks.items()}, catalog, skipped,
                   {target: tuple(claims) for target, claims in superseded.items()})

    @classmethod
    def load(cls, corpus_dir: Path, *, include_unreviewed: bool, require_license: bool = False) -> "RuleCorpus":
        return cls.from_rows(read_rows(corpus_dir), include_unreviewed=include_unreviewed, require_license=require_license)

    def __len__(self) -> int:
        return sum(len(records) for records in self.rules.values())

    @property
    def unreviewed_count(self) -> int:
        return sum(1 for records in self.rules.values() for record in records if record.get("review_status") != "approved")


def read_rows(corpus_dir: Path) -> dict[str, list[dict[str, Any]]]:
    return {name: read_jsonl(corpus_dir / name, strict=False) for name in (*RULE_FILES, *CONCEPT_FILES, CHUNKS_FILE)}


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

    def as_dict(self) -> dict[str, object]:
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


@dataclass(frozen=True)
class RuleInput:
    """EMR facts for rule evaluation, computed once when the patient context is prepared.

    Holds concept ids, values, KCD codes and ingredient ids only — no identifiers.
    """

    mapping: EmrMapping
    sex: str | None = None
    # (medCode, medName) of current prescriptions — product codes for the DUR check
    medications: tuple[tuple[str | None, str | None], ...] = ()


@dataclass
class _Builder:
    evidence: list[EvidenceChunk] = field(default_factory=list)
    numbers: dict[str, int] = field(default_factory=dict)


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
        self._max_dose_items = max_dose_items
        self._max_dose_evidence = max_dose_evidence
        self._dose_rules = corpus.rules.get(DOSE_FILE, ()) if doses else ()
        self._vocabulary = _flag_vocabulary(
            record for name, records in corpus.rules.items() if name != DOSE_FILE or doses for record in records
        )
        self._dose_targets = frozenset(
            target for record in self._dose_rules if (target := record.get("ingredient") or record.get("ingredient_class"))
        )

    def __len__(self) -> int:
        return len(self._corpus)

    @property
    def unreviewed_count(self) -> int:
        return self._corpus.unreviewed_count

    @property
    def dose_rule_count(self) -> int:
        return len(self._dose_rules)

    @property
    def dose_drugs(self) -> tuple[str, ...]:
        """Ingredient / class ids that have a dose rule (for ``dose_drugs`` input); empty when stage 3 is off."""
        return tuple(sorted(self._dose_targets))

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
        accepted, unknown = self._accepted_findings(findings or {})
        facts = self._facts(mapping.facts, accepted, sex)
        builder = _Builder()
        candidates: list[tuple[tuple, dict[str, Any], Mapping[str, Any]]] = []
        for name in ("diagnosis.jsonl", "referral.jsonl", "workup.jsonl", "monitoring.jsonl"):
            for record in self._corpus.rules.get(name, ()):
                try:
                    result = self._evaluate_record(record, facts)
                except CorpusRuleError as error:
                    logger.warning("clinical_rule_unsupported record_id=%s error=%s", record.get("id"), error)
                    continue
                if result is not None:
                    status, leaves, extra = result
                    item = self._item(record, status, leaves, facts, extra)
                    candidates.append((_rank(item, leaves), item, record))
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
            item["evidence_refs"] = self._refs(builder, record.get("claim_id"))
            items.append(item)
        labs = []
        interpretations = sorted(self._lab_interpretations(mapping, facts, sex), key=lambda pair: pair[0]["priority"] != "primary")
        for interpretation, record in interpretations:
            interpretation["evidence_refs"] = self._refs(builder, record.get("claim_id"))
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
            requested, unmatched = self._resolve_dose_drugs(dose_drugs)
            doses = tuple(self._dose_assessments(mapping, facts, requested, builder))
        return Assessment(tuple(items), tuple(labs), tuple(builder.evidence), dict(accepted), issues, tuple(unknown), alerts,
                          doses, unmatched)

    # ── stage 3: doses ───────────────────────────────────────────────────────
    def _resolve_dose_drugs(self, names: Iterable[str]) -> tuple[set[str], tuple[Mapping[str, str], ...]]:
        """Dose-rule targets for the clinician's drug list (ids, or Korean/English ingredient names)."""
        catalog = self._corpus.catalog
        targets: set[str] = set()
        unmatched: list[dict[str, str]] = []
        for raw in names:
            text = raw.strip() if isinstance(raw, str) else ""
            if not text:
                continue
            key = text.lower().replace(" ", "_")
            found: set[str] = set()
            if text in self._dose_targets or key in self._dose_targets:
                found.add(text if text in self._dose_targets else key)
            elif key in catalog.classes:
                found.update(catalog.classes[key].get("corpus_ids") or (key,))
            elif (cid := catalog.drug_by_corpus_id.get(key)) is not None:
                found.update(self._aliases(cid))
            else:
                for ingredient in catalog.match_ingredients(f"({text})")[0]:
                    found.update(self._aliases(ingredient))
            for concept_id in list(found):  # an ingredient also anchors its classes' rules (e.g. basal insulin)
                for class_id in (catalog.ingredients.get(catalog.drug_by_corpus_id.get(concept_id, concept_id)) or {}).get("classes") or ():
                    found.update(catalog.classes.get(class_id, {}).get("corpus_ids") or (class_id,))
            if not found:
                unmatched.append({"input": text, "reason": "성분을 확인할 수 없음"})
            elif not found & self._dose_targets:
                unmatched.append({"input": text, "reason": "말뭉치에 이 약의 용량 규칙 없음"})
            targets.update(found)
        return targets, tuple(unmatched)

    def _aliases(self, concept_id: str) -> tuple[str, ...]:
        row = self._corpus.catalog.ingredients.get(concept_id) or self._corpus.catalog.products.get(concept_id) or {}
        return tuple(row.get("corpus_ids") or (concept_id,))

    def _dose_assessments(
        self, mapping: EmrMapping, facts: PatientFacts, requested: set[str], builder: _Builder,
    ) -> list[dict[str, Any]]:
        medications = facts.medications or frozenset()
        classes = facts.medication_classes or frozenset()
        candidates: list[tuple[tuple, dict[str, Any], Mapping[str, Any]]] = []
        for record in self._dose_rules:
            ingredient, klass = record.get("ingredient"), record.get("ingredient_class")
            target = ingredient or klass
            if not isinstance(target, str):
                continue  # regimen-only rules (no single drug) are not anchored to a prescription
            prescribed = (ingredient in medications) if ingredient else (klass in classes)
            if not prescribed and target not in requested:
                continue
            try:
                item = self._dose_item(record, mapping, facts, prescribed=prescribed, requested=target in requested)
            except CorpusRuleError as error:
                logger.warning("clinical_dose_rule_unsupported record_id=%s error=%s", record.get("id"), error)
                continue
            rank = (0 if item["priority"] == "primary" else 1, 0 if prescribed else 1,
                    _ALERT_ORDER.get(item.get("alert"), 9), str(item.get("record_id")))
            candidates.append((rank, item, record))
        candidates.sort(key=lambda entry: entry[0])
        result = []
        # Dose values must be citable: they get their own passages beyond the stage-2 evidence cap.
        limit = len(builder.evidence) + self._max_dose_evidence
        for _rank, item, record in candidates[: self._max_dose_items]:
            item["evidence_refs"] = self._refs(builder, record.get("claim_id"), limit=limit)
            result.append(item)
        return result

    def _dose_item(
        self, record: Mapping[str, Any], mapping: EmrMapping, facts: PatientFacts, *, prescribed: bool, requested: bool,
    ) -> dict[str, Any]:
        ingredient, klass = record.get("ingredient"), record.get("ingredient_class")
        target = str(ingredient or klass)
        age = facts.values.get("age")
        missing: list[dict[str, str]] = []
        item: dict[str, Any] = {
            "record_id": record.get("id"), "claim_id": record.get("claim_id"), "kind": "dose", "topic": record.get("topic"),
            "drug": {"id": target, "name": self._drug_name(target), "type": "ingredient" if ingredient else "class"},
            "anchor": [name for name, flag in (("prescribed", prescribed), ("requested", requested)) if flag],
        }
        if record.get("indication"):
            item["indication"] = record["indication"]
        prescriptions = [self._dose_prescription(med) for med in mapping.medications if self._med_matches(med, ingredient, klass)]
        if prescriptions:
            item["prescriptions"] = prescriptions
        adult = age is not None and age >= ADULT_AGE
        if age is None:
            missing.append(self._missing_value("age"))
        elif not adult:
            item["not_applied"] = f"{ADULT_AGE}세 미만 — 성인 기준 용량 계산·신기능 구간을 적용하지 않음"

        if adult and isinstance(record.get("calculation"), Mapping):
            item["calculations"] = self._dose_calculations(record["calculation"], facts, missing)
        if isinstance(record.get("renal_param"), str):
            item["renal"] = self._dose_renal(record, mapping, facts, adult, missing)
            band = (item["renal"].get("band") or {}).get("_raw")
            if band is not None:
                item["renal"]["band"].pop("_raw")
                self._dose_check(prescriptions, band, item["renal"])
        rules = self._dose_rules_for(record, facts)
        if rules:
            item["rules"] = rules
        reference = [
            {"key": key, "label": _DOSE_LABELS.get(key, key.replace("_", " ")), "value": value}
            for key, value in record.items() if key not in _DOSE_HANDLED and value not in (None, "", [], {})
        ]
        if reference:
            item["reference"] = reference
        if record.get("alert_note"):
            item["alert_note"] = record["alert_note"]
        if record.get("conflicts"):
            item["conflicts"] = list(record["conflicts"])
        item["missing"] = _unique(missing)
        item["alert"] = _most_severe(
            [rule.get("alert") for rule in rules if rule["status"] == "met"]
            + [((item.get("renal") or {}).get("band") or {}).get("alert")]
            + ["high" for prescription in prescriptions if (prescription.get("check") or {}).get("status") == "exceeds"]
        )
        item["review_status"] = record.get("review_status")
        item.update(self._provenance(record))
        return item

    def _drug_name(self, target: str) -> str:
        catalog = self._corpus.catalog
        cid = catalog.drug_by_corpus_id.get(target, target)
        row = catalog.ingredients.get(cid) or catalog.products.get(cid) or catalog.classes.get(target) or {}
        names = row.get("ko") or ()
        return (names[0] if names else None) or row.get("name_ko") or target.replace("_", " ")

    def _med_matches(self, med: MedicationMatch, ingredient: object, klass: object) -> bool:
        catalog = self._corpus.catalog
        if isinstance(ingredient, str):
            return any(ingredient in self._aliases(item) for item in med.ingredients)
        if isinstance(klass, str):
            if klass in med.classes:
                return True
            for item in med.ingredients:
                for class_id in (catalog.ingredients.get(item) or {}).get("classes") or ():
                    if klass in (catalog.classes.get(class_id, {}).get("corpus_ids") or (class_id,)):
                        return True
        return False

    @staticmethod
    def _dose_prescription(med: MedicationMatch) -> dict[str, Any]:
        entry: dict[str, Any] = {"med_code": med.med_code, "med_name": med.med_name}
        if med.dose:
            entry["dose"] = dict(med.dose)
        daily = prescribed_daily_mg(med.dose)
        if daily is not None:
            entry["daily_mg"] = daily.mg_per_day
            entry["daily_basis"] = daily.basis
        return entry

    def _dose_calculations(
        self, calculation: Mapping[str, Any], facts: PatientFacts, missing: list[dict[str, str]],
    ) -> list[dict[str, Any]]:
        variables: dict[str, Any] = dict(facts.values)
        results = []
        for key, spec in calculation.items():
            if not isinstance(spec, Mapping) or not isinstance(spec.get("formula"), str):
                continue
            formula = spec["formula"]
            entry: dict[str, Any] = {"name": key, "formula": formula}
            unit = unit_for(key, spec.get("unit"))
            if unit:
                entry["unit"] = unit
            for extra in ("infuse_h", "note"):
                if spec.get(extra) is not None:
                    entry[extra] = spec[extra]
            try:
                value = evaluate_formula(formula, variables)
            except MissingInput as gap:
                needed = self._missing_value(gap.name)
                entry["missing"] = [needed]
                missing.append(needed)
                results.append(entry)
                continue
            variables[key] = value
            entry["inputs"] = {name: rounded(variables[name]) for name in formula_variables(formula) if name in variables}
            entry["range" if isinstance(value, tuple) else "value"] = rounded(value)
            results.append(entry)
        return results

    def _dose_renal(
        self, record: Mapping[str, Any], mapping: EmrMapping, facts: PatientFacts, adult: bool, missing: list[dict[str, str]],
    ) -> dict[str, Any]:
        param = record["renal_param"]
        renal: dict[str, Any] = {"param": param, "label": self._concept_name(param)}
        if record.get("renal_method"):
            renal["method"] = record["renal_method"]
        value = facts.values.get(param)
        if value is None:
            needed = self._missing_value(param)
            renal["missing"] = [needed]
            missing.append(needed)
            return renal
        lab = next((item for item in mapping.labs if item.concept == param), None)
        renal["value"] = value
        if lab is not None:
            renal["unit"] = lab.unit
            if lab.test_date:
                renal["test_date"] = lab.test_date
            renal["basis"] = lab.raw_unit if lab.matched_by == "derived" else "검사실 보고값"
            if lab.matched_by == "derived" and "이상체중" in str(record.get("renal_method") or ""):
                renal["method_note"] = "규칙은 이상체중 기준 CrCl — 실제 체중으로 계산한 값이라 비만 환자에서는 높게 나올 수 있음"
        if not adult or not isinstance(record.get("bands"), list):
            return renal
        band = select_band(record["bands"], value)
        if band is None:
            renal["band"] = None
            renal["band_note"] = "환자 값이 규칙의 어느 구간에도 속하지 않음"
            return renal
        renal["band"] = {
            "range": range_label(band["range"], renal.get("unit")),
            "dose": [{"key": key, "label": _BAND_LABELS.get(key, key.replace("_", " ")), "value": item}
                     for key, item in band.items() if key not in ("range", "alert")],
            **({"alert": band["alert"]} if band.get("alert") not in (None, "none") else {}),
            "_raw": band,
        }
        return renal

    @staticmethod
    def _dose_check(prescriptions: list[dict[str, Any]], band: Mapping[str, Any], renal: Mapping[str, Any]) -> None:
        """Compare each prescribed daily amount with the band's numeric daily maximum, when both exist."""
        limit = band_daily_max_mg(band)
        if limit is None:
            return
        for prescription in prescriptions:
            daily = prescription.get("daily_mg")
            if daily is None:
                continue
            prescription["check"] = {
                "limit_daily_mg": limit,
                "limit_basis": f"{renal['param']} {renal['band']['range']} 구간 1일 최대",
                "status": "exceeds" if daily > limit else "within",
            }

    def _dose_rules_for(self, record: Mapping[str, Any], facts: PatientFacts) -> list[dict[str, Any]]:
        """Dose-condition rules: met, or possible with the unknown inputs (kept per rule, not in the item's
        ``missing``, which lists only what the calculation itself needs)."""
        results = []
        for phase, rules in (("initiation", record.get("initiation")), ("rules", record.get("rules"))):
            for rule in rules if isinstance(rules, list) else ():
                when = rule.get("when") if isinstance(rule, Mapping) else None
                if not isinstance(when, Mapping):
                    continue
                result = evaluate(when, facts)
                if result is False:
                    continue
                leaves = list(_leaves(when, facts, negated=False))
                entry: dict[str, Any] = {"status": "met" if result else "possible"}
                if phase == "initiation":
                    entry["phase"] = "시작 시"
                entry.update({key: value for key, value in rule.items() if key != "when" and value is not None})
                entry["matched"] = [self._describe(leaf, facts) for leaf in leaves if leaf.result is True and not leaf.negated]
                gaps = _unique(self._missing(leaf) for leaf in leaves if leaf.result is None)
                if gaps:
                    entry["missing"] = gaps
                results.append(entry)
        return results

    def _missing_value(self, key: str) -> dict[str, str]:
        return {"kind": "value", "key": key, "label": _VALUE_HINTS.get(key) or self._concept_name(key)}

    # ── facts ──────────────────────────────────────────────────────────────
    def _accepted_findings(self, findings: Mapping[str, bool]) -> tuple[dict[str, bool], list[str]]:
        accepted: dict[str, bool] = {}
        unknown: list[str] = []
        for raw, value in findings.items():
            key = raw.split(":", 1)[1] if ":" in raw and raw.split(":", 1)[0] in FLAG_KINDS else raw
            if key in self._vocabulary and isinstance(value, bool):
                accepted[key] = value
            else:
                unknown.append(raw)
        return accepted, unknown

    def _facts(self, base: PatientFacts, findings: Mapping[str, bool], sex: str | None) -> PatientFacts:
        flags: dict[str, bool] = dict(base.flags or {})
        if sex in {"male", "female"}:
            flags.setdefault("finding:male", sex == "male")
        codes = base.diagnosis_codes or frozenset()
        if any(code.startswith(_DIABETES_KCD) for code in codes):
            flags.setdefault("dx_history:diabetes", True)
        for key, value in findings.items():
            for kind in self._vocabulary.get(key, ()):
                flags[f"{kind}:{key}"] = value  # clinician input overrides derived flags
        return replace(base, flags=flags)

    # ── rule evaluation ────────────────────────────────────────────────────
    def _evaluate_record(
        self, record: Mapping[str, Any], facts: PatientFacts
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
            return self._evaluate_monitoring(record, facts)
        else:
            return None
        if not isinstance(condition, Mapping):
            return None
        status = _status(condition, facts)
        if status is None:
            return None
        return status[0], status[1], extra

    def _evaluate_monitoring(
        self, record: Mapping[str, Any], facts: PatientFacts
    ) -> tuple[str, list[Leaf], dict[str, Any]] | None:
        after = record.get("after")
        rules = [rule for rule in record.get("rules") or () if isinstance(rule, Mapping)]
        if isinstance(after, Mapping):
            # ``after`` mixes keys (e.g. event + med_class): every key must hold.
            condition = {"all": [{key: value} for key, value in after.items()]}
            status = _status(condition, facts)
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
            result = _status(when, facts)
            if result is not None and result[0] == "met":
                fired.append(rule.get("action"))
                leaves.extend(result[1])
        if not fired:
            return None  # interval-only monitoring without an anchor stays silent
        return "met", leaves, {"actions": [action for action in fired if action]}

    # ── output ─────────────────────────────────────────────────────────────
    def _item(
        self, record: Mapping[str, Any], status: str, leaves: list[Leaf], facts: PatientFacts, extra: Mapping[str, Any]
    ) -> dict[str, Any]:
        kind = record.get("kind")
        item: dict[str, Any] = {
            "record_id": record.get("id"), "claim_id": record.get("claim_id"), "kind": kind, "status": status,
            "topic": record.get("topic"), "chapter": record.get("chapter"),
        }
        if kind == "diagnosis":
            condition = record.get("condition") or {}
            item["diagnosis"] = {"name": condition.get("name"), "kcd": list(condition.get("kcd") or ())}
            recorded = facts.diagnosis_codes or frozenset()
            item["already_recorded"] = any(code.startswith(prefix) for prefix in condition.get("kcd") or () for code in recorded)
            if record.get("severity"):
                item["severity"] = record.get("severity")
        if kind == "referral":
            item["referral_to"] = record.get("to")
            if record.get("purpose"):
                item["purpose"] = record.get("purpose")
        if kind == "workup":
            item["tests"] = [self._test(test) for test in record.get("tests") or () if isinstance(test, Mapping)]
        if kind == "monitoring":
            item["tests"] = [self._test({"lab": name}) for name in record.get("check") or ()]
            item["interval"] = {key: record[key] for key in record if key.startswith(("interval_", "within_", "timing_"))}
            if extra.get("actions"):
                item["actions"] = list(extra["actions"])
        if record.get("strength"):
            item["strength"] = record.get("strength")
        for key in ("note", "notes"):
            if record.get(key):
                item["note"] = record.get(key)
        item["matched"] = [self._describe(leaf, facts) for leaf in leaves if leaf.result is True and not leaf.negated]
        item["missing"] = _unique(self._missing(leaf) for leaf in leaves if leaf.result is None)
        item["review_status"] = record.get("review_status")
        item.update(self._provenance(record))
        if record.get("conflicts"):
            item["conflicts"] = list(record["conflicts"])
        return item

    def _provenance(self, record: Mapping[str, Any]) -> dict[str, Any]:
        """Region, priority (domestic standard first) and display sources of a rule record."""
        sources = [source for source in record.get("sources") or () if isinstance(source, Mapping)]
        region = region_of(sources)
        claim = record.get("claim_id")
        by = [c for key in (record.get("id"), claim) for c in self._corpus.superseded.get(str(key), ()) if c != claim]
        result: dict[str, Any] = {
            "region": region, "region_label": REGION_LABELS.get(region or "", "미상"),
            "priority": "reference" if by else "primary", "sources": sorted_sources(sources),
        }
        if by:
            result["superseded_by"] = list(dict.fromkeys(by))
        return result

    def _test(self, test: Mapping[str, Any]) -> dict[str, Any]:
        key = next((name for name in ("exam", "lab", "score", "calc", "screen", "review") if name in test), None)
        concept_id = str(test.get(key)) if key else ""
        concept = self._corpus.catalog.tests.get(concept_id, {})
        section = concept.get("section")
        if concept.get("kind") == "vaccination":
            category = "예방접종"
        elif section == "exams":
            category = _EXAM_CATEGORY.get(concept.get("kind"), "검사")
        elif key == "review":
            category = "검토"
        else:
            category = _CATEGORY.get(section) or _TEST_KEY_LABEL.get(key or "", "검사")
        result = {"id": concept_id, "name": concept.get("name_ko") or concept_id.replace("_", " "), "category": category}
        if test.get("reason"):
            result["reason"] = test["reason"]
        return result

    def _describe(self, leaf: Leaf, facts: PatientFacts) -> str:
        condition = leaf.condition
        if leaf.kind == "var":
            name = self._concept_name(leaf.key)
            value = facts.values.get(leaf.key)
            return f"{name} {condition.get('op')} {condition.get('value')} (환자 값 {value:g})" if value is not None else name
        if leaf.kind == "dx":
            codes = sorted(code for code in facts.diagnosis_codes or () if code.startswith(leaf.key))
            return f"진단 기록 {leaf.key} ({', '.join(codes)})"
        if leaf.kind in ("med", "med_class"):
            return f"처방 {leaf.key}"
        return f"{leaf.kind}:{leaf.key}"

    def _missing(self, leaf: Leaf) -> dict[str, str]:
        if leaf.kind == "var":
            return {"kind": "value", "key": leaf.key, "label": self._concept_name(leaf.key)}
        if leaf.kind in ("med", "med_class"):
            return {"kind": leaf.kind, "key": leaf.key, "label": f"{leaf.key} 처방 여부(성분 미확인 처방 있음)"}
        if leaf.kind == "dx":
            return {"kind": "dx", "key": leaf.key, "label": f"진단 기록 {leaf.key}"}
        return {"kind": leaf.kind, "key": leaf.key, "label": leaf.key.replace("_", " ")}

    def _concept_name(self, concept_id: str) -> str:
        concept = self._corpus.catalog.tests.get(concept_id)
        if concept is None:
            concept = next((row for row in self._corpus.catalog.tests.values() if concept_id in (row.get("corpus_ids") or ())), {})
        return concept.get("name_ko") or concept_id.replace("_", " ")

    def _lab_interpretations(
        self, mapping: EmrMapping, facts: PatientFacts, sex: str | None
    ) -> list[tuple[dict[str, Any], Mapping[str, Any]]]:
        by_concept = {lab.concept: lab for lab in mapping.labs}
        result = []
        for record in self._corpus.rules.get("lab_thresholds.jsonl", ()):
            lab = record.get("lab")
            value = facts.values.get(lab) if isinstance(lab, str) else None
            if value is None:
                continue
            for band in record.get("bands") or ():
                if "sex" in band and band["sex"] != sex:
                    continue
                if ("min" in band and value < band["min"]) or ("max" in band and value > band["max"]):
                    continue
                source = by_concept.get(lab)
                result.append(({
                    "record_id": record.get("id"), "claim_id": record.get("claim_id"), "lab": lab,
                    "name": self._concept_name(lab), "value": value, "unit": record.get("unit"),
                    "test_date": source.test_date if source else None, "band": band.get("label"),
                    "context": record.get("context") or record.get("topic"),
                    "review_status": record.get("review_status"),
                    **{key: value for key, value in self._provenance(record).items() if key != "sources"},
                }, record))
        return result

    def _refs(self, builder: _Builder, claim_id: object, limit: int | None = None) -> list[int]:
        refs: list[int] = []
        if not isinstance(claim_id, str):
            return refs
        for chunk in self._corpus.chunks_by_claim.get(claim_id, ())[: self._evidence_per_item]:
            number = builder.numbers.get(chunk.chunk_id)
            if number is None:
                if len(builder.evidence) >= (self._max_evidence if limit is None else limit):
                    continue
                builder.evidence.append(chunk)
                number = builder.numbers[chunk.chunk_id] = len(builder.evidence)
            refs.append(number)
        return refs


def _status(condition: Mapping[str, Any], facts: PatientFacts) -> tuple[str, list[Leaf]] | None:
    result = evaluate(condition, facts)
    leaves = list(_leaves(condition, facts, negated=False))
    if result is True:
        return "met", leaves
    if result is not None:
        return None
    anchors = [leaf for leaf in leaves if leaf.result is True and not leaf.negated]
    if any(not _is_weak(leaf) for leaf in anchors):
        return "possible", leaves
    # Only age/sex hold: report screening rules whose sole unknowns are "not already done"
    # (e.g. adult HBV screening without a prior panel), never symptom-driven rules.
    if anchors and all(leaf.negated for leaf in leaves if leaf.result is None):
        return "possible", leaves
    return None


def _is_weak(leaf: Leaf) -> bool:
    """Demographic anchors (age, derived sex) that alone do not tie a rule to this patient."""
    return (leaf.kind == "var" and leaf.key == "age") or (leaf.kind == "finding" and leaf.key == "male")


def _leaves(condition: Mapping[str, Any], facts: PatientFacts, *, negated: bool) -> Iterable[Leaf]:
    if "all" in condition or "any" in condition:
        children = [child for child in condition.get("all") or condition.get("any") or () if isinstance(child, Mapping)]
        if "any" in condition and evaluate(condition, facts) is True:
            # Already satisfied: unknown alternatives are irrelevant (not missing inputs).
            children = [child for child in children if evaluate(child, facts) is True]
        for child in children:
            yield from _leaves(child, facts, negated=negated)
        return
    if "not" in condition:
        if isinstance(condition["not"], Mapping):
            yield from _leaves(condition["not"], facts, negated=not negated)
        return
    if condition.get("always") is True:
        return
    kind = next((key for key in ("var", "dx", "med", "med_class", *FLAG_KINDS) if key in condition), None)
    if kind is None:
        return
    yield Leaf(kind, str(condition[kind]), condition, evaluate(condition, facts), negated)


def _rank(item: Mapping[str, Any], leaves: list[Leaf]) -> tuple:
    anchors = [leaf for leaf in leaves if leaf.result is True and not leaf.negated]
    # Anchors other than age/sex (diagnosis, prescription, lab value, clinician finding) rank first.
    strong = sum(1 for leaf in anchors if not _is_weak(leaf))
    known = sum(1 for leaf in leaves if leaf.result is not None)
    return (
        0 if item.get("priority") == "primary" else 1,  # domestic standard (or no competing one) first
        _STATUS_ORDER.get(item["status"], 9), 0 if strong else 1, _KIND_ORDER.get(item["kind"], 9),
        -strong, -(known / len(leaves) if leaves else 0), str(item.get("record_id")),
    )


def _flag_vocabulary(records: Iterable[Mapping[str, Any]]) -> dict[str, tuple[str, ...]]:
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


def _most_severe(alerts: Iterable[object]) -> str | None:
    ranked = [alert for alert in alerts if isinstance(alert, str) and alert in _ALERT_ORDER]
    return min(ranked, key=_ALERT_ORDER.__getitem__) if ranked else None


def _text_or_none(value: object) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _plausible(value: object, low: float, high: float) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and low <= float(value) <= high


def _unique(items: Iterable[dict[str, str]]) -> list[dict[str, str]]:
    seen: set[tuple[str, str]] = set()
    result = []
    for item in items:
        marker = (item["kind"], item["key"])
        if marker not in seen:
            seen.add(marker)
            result.append(item)
    return result
