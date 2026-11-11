"""Stage 3: corpus ``dose_rules`` applied to current prescriptions and to drugs the clinician names.

Weight-based formulas, the renal-function band the patient falls in, dose-condition rules, and the
prescribed daily amount against a numeric band maximum (arithmetic in ``clinical_dosing``). Values are
reference calculations that always carry their inputs, formula and sources.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable, Mapping

from ax_g_ai.services.clinical_corpus import CorpusRuleError, PatientFacts, evaluate
from ax_g_ai.services.clinical_dosing import (
    MissingInput, band_daily_max_mg, evaluate_formula, formula_variables, prescribed_daily_mg, range_label, rounded,
    select_band, unit_for,
)
from ax_g_ai.services.clinical_recommendation.corpus import RuleCorpus
from ax_g_ai.services.clinical_recommendation.model import EvidenceNumbering, unique_missing
from ax_g_ai.services.clinical_recommendation.presentation import Presenter
from ax_g_ai.services.clinical_recommendation.rules import condition_leaves
from ax_g_ai.services.emr_mapping import EmrMapping, MedicationMatch

logger = logging.getLogger("uvicorn.error")

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


class DoseStage:
    """Dose items for the targets anchored to the patient (prescribed) or requested by the clinician."""

    def __init__(
        self, corpus: RuleCorpus, present: Presenter, dose_rules: tuple[Mapping[str, Any], ...], *,
        max_items: int, max_evidence: int,
    ) -> None:
        self._corpus = corpus
        self._present = present
        self._dose_rules = dose_rules
        self._max_dose_items = max_items
        self._max_dose_evidence = max_evidence
        self._dose_targets = frozenset(
            target for record in dose_rules if (target := record.get("ingredient") or record.get("ingredient_class"))
        )

    @property
    def rule_count(self) -> int:
        return len(self._dose_rules)

    @property
    def targets(self) -> frozenset[str]:
        return self._dose_targets

    def resolve(self, names: Iterable[str]) -> tuple[set[str], tuple[Mapping[str, str], ...]]:
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
                found.update(self._present.aliases(cid))
            else:
                for ingredient in catalog.match_ingredients(f"({text})")[0]:
                    found.update(self._present.aliases(ingredient))
            for concept_id in list(found):  # an ingredient also anchors its classes' rules (e.g. basal insulin)
                for class_id in (catalog.ingredients.get(catalog.drug_by_corpus_id.get(concept_id, concept_id)) or {}).get("classes") or ():
                    found.update(catalog.classes.get(class_id, {}).get("corpus_ids") or (class_id,))
            if not found:
                unmatched.append({"input": text, "reason": "성분을 확인할 수 없음"})
            elif not found & self._dose_targets:
                unmatched.append({"input": text, "reason": "말뭉치에 이 약의 용량 규칙 없음"})
            targets.update(found)
        return targets, tuple(unmatched)

    def assessments(
        self, mapping: EmrMapping, facts: PatientFacts, requested: set[str], numbering: EvidenceNumbering,
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
        limit = len(numbering.evidence) + self._max_dose_evidence
        for _rank, item, record in candidates[: self._max_dose_items]:
            item["evidence_refs"] = numbering.refs(record.get("claim_id"), limit=limit)
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
            "drug": {"id": target, "name": self._present.drug_name(target), "type": "ingredient" if ingredient else "class"},
            "anchor": [name for name, flag in (("prescribed", prescribed), ("requested", requested)) if flag],
        }
        if record.get("indication"):
            item["indication"] = record["indication"]
        prescriptions = [self._dose_prescription(med) for med in mapping.medications if self._med_matches(med, ingredient, klass)]
        if prescriptions:
            item["prescriptions"] = prescriptions
        adult = age is not None and age >= ADULT_AGE
        if age is None:
            missing.append(self._present.missing_value("age"))
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
        item["missing"] = unique_missing(missing)
        item["alert"] = most_severe(
            [rule.get("alert") for rule in rules if rule["status"] == "met"]
            + [((item.get("renal") or {}).get("band") or {}).get("alert")]
            + ["high" for prescription in prescriptions if (prescription.get("check") or {}).get("status") == "exceeds"]
        )
        item["review_status"] = record.get("review_status")
        item.update(self._present.provenance(record))
        return item

    def _med_matches(self, med: MedicationMatch, ingredient: object, klass: object) -> bool:
        catalog = self._corpus.catalog
        if isinstance(ingredient, str):
            return any(ingredient in self._present.aliases(item) for item in med.ingredients)
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
                needed = self._present.missing_value(gap.name)
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
        renal: dict[str, Any] = {"param": param, "label": self._present.concept_name(param)}
        if record.get("renal_method"):
            renal["method"] = record["renal_method"]
        value = facts.values.get(param)
        if value is None:
            needed = self._present.missing_value(param)
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
                leaves = list(condition_leaves(when, facts, negated=False))
                entry: dict[str, Any] = {"status": "met" if result else "possible"}
                if phase == "initiation":
                    entry["phase"] = "시작 시"
                entry.update({key: value for key, value in rule.items() if key != "when" and value is not None})
                entry["matched"] = [self._present.describe(leaf, facts) for leaf in leaves if leaf.result is True and not leaf.negated]
                gaps = unique_missing(self._present.missing(leaf) for leaf in leaves if leaf.result is None)
                if gaps:
                    entry["missing"] = gaps
                results.append(entry)
        return results


def most_severe(alerts: Iterable[object]) -> str | None:
    ranked = [alert for alert in alerts if isinstance(alert, str) and alert in _ALERT_ORDER]
    return min(ranked, key=_ALERT_ORDER.__getitem__) if ranked else None
