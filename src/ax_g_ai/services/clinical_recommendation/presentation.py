"""규칙 결과의 표시 형태: 추천 항목, 검사값 해석, 출처 정보, 검사·입력의 표시 이름.

설계상 위치: 임상 추천 패키지의 표시 계층. ``ClinicalRecommender.assess``와 ``doses.DoseStage``가 ``Presenter``로
``schema`` 모델(페이로드 항목)을 만든다.

역할: 한국어 표시 이름은 여기(용량 필드는 ``doses``)에만 두고, 평가 코드는 텍스트를 만들지 않는다. 규칙 레코드와
환자 사실로 항목·검사 목록·충족 조건 설명·누락 입력·출처(지역, 우선순위, 대체 관계)를 만든다.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

from ax_g_ai.services.clinical_corpus import PatientFacts
from ax_g_ai.services.clinical_evidence import REGION_LABELS, region_of, sorted_sources
from ax_g_ai.services.clinical_recommendation.corpus import RuleCorpus
from ax_g_ai.services.clinical_recommendation.model import Leaf, unique_missing
from ax_g_ai.services.clinical_recommendation.schema import (
    Diagnosis,
    ItemKind,
    LabInterpretation,
    MissingInput,
    RecommendationItem,
    Status,
    TestEntry,
)
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
    """말뭉치 레코드와 환자 사실로 평가 페이로드 항목을 만드는 표시기.

    역할: 말뭉치 카탈로그로 개념·약 이름을 찾고, 2단계 항목·검사값 해석·출처 정보와 충족·누락 조건 표시를 만든다.
    상태를 갖지 않으므로 2단계와 3단계가 함께 쓴다.
    """

    def __init__(self, corpus: RuleCorpus) -> None:
        self._corpus = corpus

    def item(
        self, record: Mapping[str, Any], status: Status, leaves: list[Leaf], facts: PatientFacts, extra: Mapping[str, Any]
    ) -> RecommendationItem:
        """``met``·``possible`` 규칙 하나의 페이로드 항목을 만든다.

        역할: 종류별 선택 필드(진단: 진단명·KCD·이미 기록 여부·중증도, 협진: 대상·목적, 추천 검사·추적: 검사 목록·
        주기·조치)와 권고 강도·메모·충돌을 해당할 때만 넣고, 충족 조건·누락 입력·출처 정보를 붙인다.
        입력:
            record: 규칙 레코드.
            status: ``met`` 또는 ``possible``.
            leaves: 규칙의 잎 조건.
            facts: 환자 사실.
            extra: ``rules.evaluate_record``의 추가 정보(추적 규칙의 ``actions``).
        출력: ``RecommendationItem``.
        """
        # ``rules.evaluate_record``는 이 네 종류에만 상태를 준다. 모델이 다시 검증한다.
        kind = cast(ItemKind, record.get("kind"))
        # 일부 종류·레코드에만 있는 필드: 값이 있을 때만 넘겨 페이로드에서 나머지는 빠진다.
        optional: dict[str, Any] = {}
        if kind == "diagnosis":
            condition = record.get("condition") or {}
            kcd = list(condition.get("kcd") or ())
            recorded = facts.diagnosis_codes or frozenset()
            optional["diagnosis"] = Diagnosis(name=condition.get("name"), kcd=kcd)
            optional["already_recorded"] = any(code.startswith(prefix) for prefix in kcd for code in recorded)
            if record.get("severity"):
                optional["severity"] = record.get("severity")
        if kind == "referral":
            optional["referral_to"] = record.get("to")
            if record.get("purpose"):
                optional["purpose"] = record.get("purpose")
        if kind == "workup":
            optional["tests"] = [self.test_entry(test) for test in record.get("tests") or () if isinstance(test, Mapping)]
        if kind == "monitoring":
            optional["tests"] = [self.test_entry({"lab": name}) for name in record.get("check") or ()]
            optional["interval"] = {key: record[key] for key in record if key.startswith(("interval_", "within_", "timing_"))}
            if extra.get("actions"):
                optional["actions"] = list(extra["actions"])
        if record.get("strength"):
            optional["strength"] = record.get("strength")
        for key in ("note", "notes"):
            if record.get(key):
                optional["note"] = record.get(key)
        conflicts: dict[str, Any] = {"conflicts": list(record["conflicts"])} if record.get("conflicts") else {}
        return RecommendationItem(
            record_id=record.get("id"), claim_id=record.get("claim_id"), kind=kind, status=status,
            topic=record.get("topic"), chapter=record.get("chapter"),
            **optional,
            matched=[self.describe(leaf, facts) for leaf in leaves if leaf.result is True and not leaf.negated],
            missing=unique_missing(self.missing(leaf) for leaf in leaves if leaf.result is None),
            review_status=record.get("review_status"),
            **self.provenance(record),
            **conflicts,
        )

    def provenance(self, record: Mapping[str, Any]) -> dict[str, Any]:
        """규칙 레코드의 지역, 우선순위(국내 기준 먼저), 표시용 출처를 만든다.

        역할: 출처로 지역을 정하고, 이 레코드(ID 또는 claim)를 대체하는 상위 기준이 있으면 ``reference``와
        ``superseded_by``를, 없으면 ``primary``를 준다.
        입력:
            record: 규칙 레코드.
        출력: ``Provenance`` 필드 이름 → 값 dict(``region``, ``region_label``, ``priority``, ``sources``, ``superseded_by``).
        """
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

    def test_entry(self, test: Mapping[str, Any]) -> TestEntry:
        """규칙의 검사 지정 하나를 표시용 검사 항목으로 바꾼다.

        역할: ``exam``·``lab``·``score``·``calc``·``screen``·``review`` 중 있는 키로 개념을 찾고, 카탈로그의 구역·종류로
        분류(예방접종, 영상검사, 검체검사 등)를 정한다. 이름이 없으면 ID에서 만든다.
        입력:
            test: 규칙의 검사 지정(예: ``{"lab": "HbA1c", "reason": ...}``).
        출력: ``TestEntry``.
        """
        key = next((name for name in ("exam", "lab", "score", "calc", "screen", "review") if name in test), None)
        concept_id = str(test.get(key)) if key else ""
        concept = self._corpus.catalog.tests.get(concept_id, {})
        section = concept.get("section")
        if concept.get("kind") == "vaccination":
            category = "예방접종"
        elif section == "exams":
            category = _EXAM_CATEGORY.get(str(concept.get("kind")), "검사")
        elif key == "review":
            category = "검토"
        else:
            category = _CATEGORY.get(str(section)) or _TEST_KEY_LABEL.get(key or "", "검사")
        return TestEntry(
            id=concept_id, name=concept.get("name_ko") or concept_id.replace("_", " "), category=category,
            **({"reason": test["reason"]} if test.get("reason") else {}),
        )

    def describe(self, leaf: Leaf, facts: PatientFacts) -> str:
        """충족된 잎 조건을 화면용 문장으로 설명한다.

        역할: 수치는 "이름 연산자 기준값 (환자 값 n)", 진단은 기록된 KCD 코드, 처방은 "처방 키"로, 그 밖은
        ``kind:key``로 쓴다.
        입력:
            leaf: 잎 조건.
            facts: 환자 사실.
        출력: 설명 문자열.
        """
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

    def missing(self, leaf: Leaf) -> MissingInput:
        """알 수 없는 잎 조건을 의료진이 확인할 누락 입력으로 바꾼다."""
        if leaf.kind == "var":
            return MissingInput(kind="value", key=leaf.key, label=self.concept_name(leaf.key))
        if leaf.kind in ("med", "med_class"):
            return MissingInput(kind=leaf.kind, key=leaf.key, label=f"{leaf.key} 처방 여부(성분 미확인 처방 있음)")
        if leaf.kind == "dx":
            return MissingInput(kind="dx", key=leaf.key, label=f"진단 기록 {leaf.key}")
        return MissingInput(kind=leaf.kind, key=leaf.key, label=leaf.key.replace("_", " "))

    def missing_value(self, key: str) -> MissingInput:
        """계산에 필요한 수치 입력의 누락 항목(CrCl·eGFR 등은 계산 방법 안내를 이름으로 쓴다)."""
        return MissingInput(kind="value", key=key, label=_VALUE_HINTS.get(key) or self.concept_name(key))

    def concept_name(self, concept_id: str) -> str:
        """검사 개념의 한글 이름(``corpus_ids`` 별칭으로도 찾음). 없으면 ID의 밑줄을 공백으로 바꿔 쓴다."""
        concept = self._corpus.catalog.tests.get(concept_id)
        if concept is None:
            concept = next((row for row in self._corpus.catalog.tests.values() if concept_id in (row.get("corpus_ids") or ())), {})
        return concept.get("name_ko") or concept_id.replace("_", " ")

    def drug_name(self, target: str) -> str:
        """성분·제품·계열 ID의 한글 이름. 없으면 ID의 밑줄을 공백으로 바꿔 쓴다."""
        catalog = self._corpus.catalog
        cid = catalog.drug_by_corpus_id.get(target, target)
        row = catalog.ingredients.get(cid) or catalog.products.get(cid) or catalog.classes.get(target) or {}
        names = row.get("ko") or ()
        return (names[0] if names else None) or row.get("name_ko") or target.replace("_", " ")

    def aliases(self, concept_id: str) -> tuple[str, ...]:
        """성분·제품 개념의 말뭉치 ID 별칭(``corpus_ids``). 없으면 자기 ID 하나."""
        row = self._corpus.catalog.ingredients.get(concept_id) or self._corpus.catalog.products.get(concept_id) or {}
        return tuple(row.get("corpus_ids") or (concept_id,))

    def lab_interpretations(
        self, mapping: EmrMapping, facts: PatientFacts, sex: str | None
    ) -> list[tuple[LabInterpretation, Mapping[str, Any]]]:
        """환자 검사값이 속한 기준 구간을 ``lab_thresholds.jsonl`` 규칙으로 찾는다.

        역할: 값이 있는 검사마다 규칙의 구간 중 성별 조건과 ``min``·``max`` 범위에 맞는 모든 구간을 해석 항목으로
        만든다. 출처 정보에서 ``sources``는 빼고, 검사일은 EMR 매핑에서 가져온다.
        입력:
            mapping: EMR 매핑(검사일 확인용).
            facts: 환자 사실(기준 단위 검사값).
            sex: ``male``·``female`` 또는 ``None``.
        출력: (``LabInterpretation``, 원 규칙 레코드) 목록. 근거 번호는 호출자가 붙인다.
        """
        by_concept = {lab.concept: lab for lab in mapping.labs}
        result = []
        for record in self._corpus.rules.get("lab_thresholds.jsonl", ()):
            lab = record.get("lab")
            if not isinstance(lab, str):
                continue
            value = facts.values.get(lab)
            if value is None:
                continue
            for band in record.get("bands") or ():
                if "sex" in band and band["sex"] != sex:
                    continue
                if ("min" in band and value < band["min"]) or ("max" in band and value > band["max"]):
                    continue
                source = by_concept.get(lab)
                provenance = {key: item for key, item in self.provenance(record).items() if key != "sources"}
                result.append((LabInterpretation(
                    record_id=record.get("id"), claim_id=record.get("claim_id"), lab=lab,
                    name=self.concept_name(lab), value=value, unit=record.get("unit"),
                    test_date=source.test_date if source else None, band=band.get("label"),
                    context=record.get("context") or record.get("topic"),
                    review_status=record.get("review_status"),
                    **provenance,
                ), record))
        return result
