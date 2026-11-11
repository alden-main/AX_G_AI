"""Display form of rule results: items, lab interpretations, provenance and the labels of tests and inputs.

Korean labels live here (and in ``doses`` for dose fields); evaluation code never builds text.
"""

from __future__ import annotations

from typing import Any, Mapping

from ax_g_ai.services.clinical_corpus import PatientFacts
from ax_g_ai.services.clinical_evidence import REGION_LABELS, region_of, sorted_sources
from ax_g_ai.services.clinical_recommendation.corpus import RuleCorpus
from ax_g_ai.services.clinical_recommendation.model import Leaf, unique_missing
from ax_g_ai.services.emr_mapping import EmrMapping

_CATEGORY = {
    "labs": "검체검사(혈액·소변 등)", "panels": "검체검사 묶음", "derived": "계산 지표", "scores": "점수·위험도 평가",
    "vitals": "활력징후·신체계측", "measurements": "측정", "clinical": "임상 평가", "tasks": "평가 항목", "med_derived": "약물 관련 지표",
}
_EXAM_CATEGORY = {"imaging": "영상검사", "endoscopy": "내시경검사", "biopsy": "조직검사", "function": "기능·진찰 검사"}
_TEST_KEY_LABEL = {"review": "검토", "calc": "계산", "score": "점수·위험도 평가", "screen": "선별 평가"}
_VALUE_HINTS = {
    "CrCl": "크레아티닌 청소율(Cockcroft-Gault) — 연령·성별·체중·혈청 크레아티닌이 있으면 계산",
    "eGFR": "추정 사구체여과율 — 연령·성별·혈청 크레아티닌이 있으면 계산",
    "weight_kg": "체중(kg)",
    "age": "연령",
}


class Presenter:
    """Builds the payload entries of an assessment from corpus records and patient facts."""

    def __init__(self, corpus: RuleCorpus) -> None:
        self._corpus = corpus

    def item(
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
            item["tests"] = [self.test_entry(test) for test in record.get("tests") or () if isinstance(test, Mapping)]
        if kind == "monitoring":
            item["tests"] = [self.test_entry({"lab": name}) for name in record.get("check") or ()]
            item["interval"] = {key: record[key] for key in record if key.startswith(("interval_", "within_", "timing_"))}
            if extra.get("actions"):
                item["actions"] = list(extra["actions"])
        if record.get("strength"):
            item["strength"] = record.get("strength")
        for key in ("note", "notes"):
            if record.get(key):
                item["note"] = record.get(key)
        item["matched"] = [self.describe(leaf, facts) for leaf in leaves if leaf.result is True and not leaf.negated]
        item["missing"] = unique_missing(self.missing(leaf) for leaf in leaves if leaf.result is None)
        item["review_status"] = record.get("review_status")
        item.update(self.provenance(record))
        if record.get("conflicts"):
            item["conflicts"] = list(record["conflicts"])
        return item

    def provenance(self, record: Mapping[str, Any]) -> dict[str, Any]:
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

    def test_entry(self, test: Mapping[str, Any]) -> dict[str, Any]:
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

    def describe(self, leaf: Leaf, facts: PatientFacts) -> str:
        condition = leaf.condition
        if leaf.kind == "var":
            name = self.concept_name(leaf.key)
            value = facts.values.get(leaf.key)
            return f"{name} {condition.get('op')} {condition.get('value')} (환자 값 {value:g})" if value is not None else name
        if leaf.kind == "dx":
            codes = sorted(code for code in facts.diagnosis_codes or () if code.startswith(leaf.key))
            return f"진단 기록 {leaf.key} ({', '.join(codes)})"
        if leaf.kind in ("med", "med_class"):
            return f"처방 {leaf.key}"
        return f"{leaf.kind}:{leaf.key}"

    def missing(self, leaf: Leaf) -> dict[str, str]:
        if leaf.kind == "var":
            return {"kind": "value", "key": leaf.key, "label": self.concept_name(leaf.key)}
        if leaf.kind in ("med", "med_class"):
            return {"kind": leaf.kind, "key": leaf.key, "label": f"{leaf.key} 처방 여부(성분 미확인 처방 있음)"}
        if leaf.kind == "dx":
            return {"kind": "dx", "key": leaf.key, "label": f"진단 기록 {leaf.key}"}
        return {"kind": leaf.kind, "key": leaf.key, "label": leaf.key.replace("_", " ")}

    def missing_value(self, key: str) -> dict[str, str]:
        return {"kind": "value", "key": key, "label": _VALUE_HINTS.get(key) or self.concept_name(key)}

    def concept_name(self, concept_id: str) -> str:
        concept = self._corpus.catalog.tests.get(concept_id)
        if concept is None:
            concept = next((row for row in self._corpus.catalog.tests.values() if concept_id in (row.get("corpus_ids") or ())), {})
        return concept.get("name_ko") or concept_id.replace("_", " ")

    def drug_name(self, target: str) -> str:
        catalog = self._corpus.catalog
        cid = catalog.drug_by_corpus_id.get(target, target)
        row = catalog.ingredients.get(cid) or catalog.products.get(cid) or catalog.classes.get(target) or {}
        names = row.get("ko") or ()
        return (names[0] if names else None) or row.get("name_ko") or target.replace("_", " ")

    def aliases(self, concept_id: str) -> tuple[str, ...]:
        row = self._corpus.catalog.ingredients.get(concept_id) or self._corpus.catalog.products.get(concept_id) or {}
        return tuple(row.get("corpus_ids") or (concept_id,))

    def lab_interpretations(
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
                    "name": self.concept_name(lab), "value": value, "unit": record.get("unit"),
                    "test_date": source.test_date if source else None, "band": band.get("label"),
                    "context": record.get("context") or record.get("topic"),
                    "review_status": record.get("review_status"),
                    **{key: value for key, value in self.provenance(record).items() if key != "sources"},
                }, record))
        return result
