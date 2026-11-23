"""3단계: 말뭉치 ``dose_rules``를 현재 처방과 의료진이 지정한 약에 적용한다.

설계상 위치: 임상 추천 패키지의 용량 단계. ``ClinicalRecommender``가 ``doses=True``일 때 ``DoseStage``를 써서
``assess`` 결과의 ``dose_assessments``·``unmatched_dose_drugs``를 만든다.

역할: 체중 기반 공식, 환자가 속한 신기능 구간, 용량 조건 규칙, 처방 1일 용량과 구간 수치 최대량의 대조를 계산한다
(산술은 ``clinical_dosing``). 값은 참고 계산이며 항상 입력·공식·출처를 함께 담는다. 용량 필드의 한국어 표시 이름도
여기 둔다.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from typing import Any, Literal

from ax_g_ai.services.clinical_corpus import CorpusRuleError, PatientFacts, evaluate
from ax_g_ai.services.clinical_dosing import MissingInput as MissingFormulaInput
from ax_g_ai.services.clinical_dosing import (
    band_daily_max_mg,
    evaluate_formula,
    formula_variables,
    prescribed_daily_mg,
    range_label,
    rounded,
    select_band,
    unit_for,
)
from ax_g_ai.services.clinical_recommendation.corpus import RuleCorpus
from ax_g_ai.services.clinical_recommendation.model import EvidenceNumbering, unique_missing
from ax_g_ai.services.clinical_recommendation.presentation import Presenter
from ax_g_ai.services.clinical_recommendation.rules import condition_leaves
from ax_g_ai.services.clinical_recommendation.schema import (
    DoseAssessment,
    DoseCalculation,
    DoseCheck,
    DoseConditionRule,
    DoseDrug,
    DosePrescription,
    DoseRenal,
    LabeledValue,
    MissingInput,
    RenalBand,
    UnmatchedDoseDrug,
)
from ax_g_ai.services.emr_mapping import EmrMapping, MedicationMatch

logger = logging.getLogger("uvicorn.error")

# ── 3단계(용량) 표시 ──────────────────────────────────────────────
ADULT_AGE = 18
# 용량 단계가 평가하거나 다른 곳에 보이는 레코드 키. 나머지는 참고 텍스트로 보인다.
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
    """환자에게 처방 중이거나 의료진이 요청한 약에 대한 용량 항목을 만드는 3단계 평가기.

    역할: 용량 규칙의 대상(성분·계열) 집합을 두고, 의료진의 약 목록을 대상으로 해석하며(``resolve``), 대상마다
    계산·신기능 구간·조건 규칙·처방 대조를 담은 ``DoseAssessment``를 만든다(``assessments``).
    """

    def __init__(
        self, corpus: RuleCorpus, present: Presenter, dose_rules: tuple[Mapping[str, Any], ...], *,
        max_items: int, max_evidence: int,
    ) -> None:
        """용량 단계를 구성한다.

        역할: 용량 규칙과 상한을 저장하고, 규칙의 ``ingredient`` 또는 ``ingredient_class``로 대상 집합을 만든다.
        입력:
            corpus: 규칙 말뭉치(카탈로그 조회용).
            present: 표시용 ``Presenter``.
            dose_rules: 관문을 통과한 ``dose_rules`` 레코드(3단계가 꺼지면 빈 튜플).
            max_items: 용량 항목 최대 수.
            max_evidence: 2단계 근거 위에 추가로 허용하는 용량 근거 문단 수.
        출력: 없음.
        """
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

    def resolve(self, names: Iterable[str]) -> tuple[set[str], tuple[UnmatchedDoseDrug, ...]]:
        """의료진의 약 목록(ID 또는 한글·영문 성분명)을 용량 규칙 대상으로 해석한다.

        역할: 이름마다 규칙 대상 ID, 계열 ID, 말뭉치 ID, 성분명 매칭 순으로 개념을 찾고, 성분이면 그 성분이 속한
        계열의 규칙도 대상에 넣는다. 찾지 못하거나 용량 규칙이 없는 이름은 이유와 함께 미매칭으로 보고한다.
        입력:
            names: 요청한 약 이름·ID. 빈 값은 건너뛴다.
        출력: (대상 개념 ID 집합, ``UnmatchedDoseDrug`` 튜플).
        """
        catalog = self._corpus.catalog
        targets: set[str] = set()
        unmatched: list[UnmatchedDoseDrug] = []
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
            for concept_id in list(found):  # 성분은 그 성분이 속한 계열의 규칙에도 이어진다(예: 기저 인슐린)
                for class_id in (catalog.ingredients.get(catalog.drug_by_corpus_id.get(concept_id, concept_id)) or {}).get("classes") or ():
                    found.update(catalog.classes.get(class_id, {}).get("corpus_ids") or (class_id,))
            if not found:
                unmatched.append(UnmatchedDoseDrug(input=text, reason="성분을 확인할 수 없음"))
            elif not found & self._dose_targets:
                unmatched.append(UnmatchedDoseDrug(input=text, reason="말뭉치에 이 약의 용량 규칙 없음"))
            targets.update(found)
        return targets, tuple(unmatched)

    def assessments(
        self, mapping: EmrMapping, facts: PatientFacts, requested: set[str], numbering: EvidenceNumbering,
    ) -> list[DoseAssessment]:
        """처방 중이거나 요청된 대상의 용량 항목을 정렬해 상한까지 돌려준다.

        역할: 단일 약 대상이 있는 규칙 중 처방 중이거나 요청된 것만 항목으로 만들고(평가할 수 없는 규칙은 경고 로그 후
        건너뜀), 국내 기준 → 처방 중 → 경고 단계 → 레코드 ID 순으로 정렬한다. 정렬 순서대로 근거 번호를 붙이되,
        2단계 근거 상한과 별도로 ``max_evidence``개까지 더 허용한다.
        입력:
            mapping: EMR 매핑(처방·검사 정보).
            facts: 환자 사실.
            requested: ``resolve``로 얻은 요청 대상.
            numbering: 2단계와 함께 쓰는 근거 번호 매기기.
        출력: ``DoseAssessment`` 목록(최대 ``max_items``개).
        """
        medications = facts.medications or frozenset()
        classes = facts.medication_classes or frozenset()
        candidates: list[tuple[tuple, DoseAssessment, Mapping[str, Any]]] = []
        for record in self._dose_rules:
            ingredient, klass = record.get("ingredient"), record.get("ingredient_class")
            target = ingredient or klass
            if not isinstance(target, str):
                continue  # 요법만 있는 규칙(단일 약 없음)은 처방과 이어지지 않는다
            prescribed = (ingredient in medications) if ingredient else (klass in classes)
            if not prescribed and target not in requested:
                continue
            try:
                item = self._dose_item(record, mapping, facts, prescribed=prescribed, requested=target in requested)
            except CorpusRuleError as error:
                logger.warning("clinical_dose_rule_unsupported record_id=%s error=%s", record.get("id"), error)
                continue
            rank = (0 if item.priority == "primary" else 1, 0 if prescribed else 1,
                    _ALERT_ORDER.get(str(item.alert), 9), str(item.record_id))
            candidates.append((rank, item, record))
        candidates.sort(key=lambda entry: entry[0])
        result = []
        # 용량 값은 인용할 수 있어야 하므로 2단계 근거 상한을 넘어 자기 근거 문단을 받는다.
        limit = len(numbering.evidence) + self._max_dose_evidence
        for _rank, item, record in candidates[: self._max_dose_items]:
            # 정렬 순서대로 번호를 매기므로 정렬한 뒤에 지정한다.
            item.evidence_refs = numbering.refs(record.get("claim_id"), limit=limit)
            result.append(item)
        return result

    def _dose_item(
        self, record: Mapping[str, Any], mapping: EmrMapping, facts: PatientFacts, *, prescribed: bool, requested: bool,
    ) -> DoseAssessment:
        """약 하나의 용량 항목을 만든다. 선택 필드는 값이 있을 때만 넘겨 페이로드에서 나머지는 빠진다.

        역할: 맞는 처방과 1일 용량을 모으고, 연령이 없으면 누락으로, 18세 미만이면 성인 기준 미적용으로 표시한다.
        성인이면 공식 계산과 신기능 구간(및 처방 1일 최대량 대조)을 하고, 용량 조건 규칙과 나머지 참고 필드를 붙인다.
        경고는 충족된 조건 규칙·신기능 구간·최대량 초과(``high``) 중 가장 높은 단계로 정한다.
        입력:
            record: ``dose_rules`` 레코드.
            mapping: EMR 매핑.
            facts: 환자 사실.
            prescribed: 대상이 현재 처방 중인지.
            requested: 의료진이 요청한 대상인지.
        출력: ``DoseAssessment``.
        예외: 규칙 조건을 평가할 수 없으면 ``CorpusRuleError``(호출자가 처리).
        """
        ingredient, klass = record.get("ingredient"), record.get("ingredient_class")
        target = str(ingredient or klass)
        age = facts.values.get("age")
        missing: list[MissingInput] = []
        optional: dict[str, Any] = {}
        if record.get("indication"):
            optional["indication"] = record["indication"]
        prescriptions = [self._dose_prescription(med) for med in mapping.medications if self._med_matches(med, ingredient, klass)]
        if prescriptions:
            optional["prescriptions"] = prescriptions
        adult = age is not None and age >= ADULT_AGE
        if age is None:
            missing.append(self._present.missing_value("age"))
        elif not adult:
            optional["not_applied"] = f"{ADULT_AGE}세 미만 — 성인 기준 용량 계산·신기능 구간을 적용하지 않음"

        if adult and isinstance(record.get("calculation"), Mapping):
            optional["calculations"] = self._dose_calculations(record["calculation"], facts, missing)
        renal: DoseRenal | None = None
        if isinstance(record.get("renal_param"), str):
            renal, band = self._dose_renal(record, mapping, facts, adult, missing)
            if band is not None:
                self._dose_check(prescriptions, band, renal)
            optional["renal"] = renal
        rules = self._dose_rules_for(record, facts)
        if rules:
            optional["rules"] = rules
        reference = [
            LabeledValue(key=key, label=_DOSE_LABELS.get(key, key.replace("_", " ")), value=value)
            for key, value in record.items() if key not in _DOSE_HANDLED and value not in (None, "", [], {})
        ]
        if reference:
            optional["reference"] = reference
        if record.get("alert_note"):
            optional["alert_note"] = record["alert_note"]
        if record.get("conflicts"):
            optional["conflicts"] = list(record["conflicts"])
        alert = most_severe(
            [(rule.model_extra or {}).get("alert") for rule in rules if rule.status == "met"]
            + [renal.band.alert if renal is not None and renal.band is not None else None]
            + ["high" for prescription in prescriptions if prescription.check is not None and prescription.check.status == "exceeds"]
        )
        anchor: list[Literal["prescribed", "requested"]] = []
        if prescribed:
            anchor.append("prescribed")
        if requested:
            anchor.append("requested")
        return DoseAssessment(
            record_id=record.get("id"), claim_id=record.get("claim_id"), kind="dose", topic=record.get("topic"),
            drug=DoseDrug(id=target, name=self._present.drug_name(target), type="ingredient" if ingredient else "class"),
            anchor=anchor,
            **optional,
            missing=unique_missing(missing), alert=alert, review_status=record.get("review_status"),
            **self._present.provenance(record),
        )

    def _med_matches(self, med: MedicationMatch, ingredient: object, klass: object) -> bool:
        """처방 하나가 규칙의 성분 또는 계열에 해당하는지 판정한다.

        역할: 성분 규칙이면 처방 성분의 별칭에 그 성분이 있는지, 계열 규칙이면 처방의 계열이나 처방 성분이 속한
        계열(말뭉치 ID 별칭 포함)에 그 계열이 있는지 본다.
        입력:
            med: EMR 처방 매핑 결과.
            ingredient: 규칙의 ``ingredient``(문자열이 아니면 무시).
            klass: 규칙의 ``ingredient_class``(문자열이 아니면 무시).
        출력: 해당하면 True.
        """
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
    def _dose_prescription(med: MedicationMatch) -> DosePrescription:
        """처방 하나의 표시 항목(EMR 원 용량, 명확할 때만 계산한 1일 mg)."""
        optional: dict[str, Any] = {"dose": dict(med.dose)} if med.dose else {}
        daily = prescribed_daily_mg(med.dose)
        if daily is not None:
            optional.update(daily_mg=daily.mg_per_day, daily_basis=daily.basis)
        return DosePrescription(med_code=med.med_code, med_name=med.med_name, **optional)

    def _dose_calculations(
        self, calculation: Mapping[str, Any], facts: PatientFacts, missing: list[MissingInput],
    ) -> list[DoseCalculation]:
        """규칙의 용량 공식을 환자 값으로 계산한다.

        역할: 공식마다 단위·주입 시간·메모를 붙여 계산하고, 결과를 다음 공식의 변수로 쓴다. 입력이 없으면 그 공식은
        ``missing``만 담고 누락 입력을 ``missing`` 목록에도 더한다. 계산하면 쓴 입력값과 값(또는 범위)을 담는다.
        입력:
            calculation: 레코드의 ``calculation``(이름 → ``formula`` 등).
            facts: 환자 사실.
            missing: 항목의 누락 입력 목록(이 함수가 추가한다).
        출력: ``DoseCalculation`` 목록.
        """
        variables: dict[str, Any] = dict(facts.values)
        results = []
        for key, spec in calculation.items():
            if not isinstance(spec, Mapping) or not isinstance(spec.get("formula"), str):
                continue
            formula = spec["formula"]
            optional: dict[str, Any] = {}
            unit = unit_for(key, spec.get("unit"))
            if unit:
                optional["unit"] = unit
            for extra in ("infuse_h", "note"):
                if spec.get(extra) is not None:
                    optional[extra] = spec[extra]
            try:
                value = evaluate_formula(formula, variables)
            except MissingFormulaInput as gap:
                needed = self._present.missing_value(gap.name)
                missing.append(needed)
                results.append(DoseCalculation(name=key, formula=formula, **optional, missing=[needed]))
                continue
            variables[key] = value
            inputs = {name: rounded(variables[name]) for name in formula_variables(formula) if name in variables}
            result: dict[str, Any] = {"range" if isinstance(value, tuple) else "value": rounded(value)}
            results.append(DoseCalculation(name=key, formula=formula, **optional, inputs=inputs, **result))
        return results

    def _dose_renal(
        self, record: Mapping[str, Any], mapping: EmrMapping, facts: PatientFacts, adult: bool, missing: list[MissingInput],
    ) -> tuple[DoseRenal, Mapping[str, Any] | None]:
        """표시할 신기능 값과 구간, 그리고 해당 구간이 있으면 규칙의 구간 원본(1일 최대량 대조용)을 돌려준다.

        역할: ``renal_param``(CrCl·eGFR) 값이 없으면 누락으로 표시한다. 있으면 단위·검사일·근거(검사실 보고값 또는
        계산식)를 붙이고, 이상체중 기준 규칙인데 실제 체중으로 계산한 값이면 주의 문구를 단다. 성인이고 구간이
        있으면 환자 값이 속한 구간을 고르고, 어느 구간에도 속하지 않으면 그 사실을 적는다.
        입력:
            record: ``dose_rules`` 레코드.
            mapping: EMR 매핑(검사 단위·검사일 확인용).
            facts: 환자 사실.
            adult: 성인(18세 이상)인지.
            missing: 항목의 누락 입력 목록(이 함수가 추가한다).
        출력: (``DoseRenal``, 규칙 구간 dict 또는 ``None``).
        """
        param = record["renal_param"]
        renal = DoseRenal(param=param, label=self._present.concept_name(param),
                          **({"method": record["renal_method"]} if record.get("renal_method") else {}))
        value = facts.values.get(param)
        if value is None:
            needed = self._present.missing_value(param)
            renal.missing = [needed]
            missing.append(needed)
            return renal, None
        lab = next((item for item in mapping.labs if item.concept == param), None)
        renal.value = value
        if lab is not None:
            renal.unit = lab.unit
            if lab.test_date:
                renal.test_date = lab.test_date
            renal.basis = lab.raw_unit if lab.matched_by == "derived" else "검사실 보고값"
            if lab.matched_by == "derived" and "이상체중" in str(record.get("renal_method") or ""):
                renal.method_note = "규칙은 이상체중 기준 CrCl — 실제 체중으로 계산한 값이라 비만 환자에서는 높게 나올 수 있음"
        if not adult or not isinstance(record.get("bands"), list):
            return renal, None
        band = select_band(record["bands"], value)
        if band is None:
            renal.band = None
            renal.band_note = "환자 값이 규칙의 어느 구간에도 속하지 않음"
            return renal, None
        renal.band = RenalBand(
            range=range_label(band["range"], renal.unit),
            dose=[LabeledValue(key=key, label=_BAND_LABELS.get(key, key.replace("_", " ")), value=item)
                  for key, item in band.items() if key not in ("range", "alert")],
            **({"alert": band["alert"]} if band.get("alert") not in (None, "none") else {}),
        )
        return renal, band

    @staticmethod
    def _dose_check(prescriptions: list[DosePrescription], band: Mapping[str, Any], renal: DoseRenal) -> None:
        """처방마다 1일 용량을 구간의 수치 1일 최대량과 대조한다(둘 다 있을 때만).

        역할: 대조 결과(``exceeds``·``within``)와 기준을 각 처방의 ``check``에 적는다.
        입력:
            prescriptions: 표시용 처방 항목(이 함수가 ``check``를 채운다).
            band: 규칙의 신기능 구간 원본.
            renal: 표시용 신기능 결과(구간 범위 이름용).
        출력: 없음.
        """
        limit = band_daily_max_mg(band)
        if limit is None or renal.band is None:
            return
        for prescription in prescriptions:
            daily = prescription.daily_mg
            if daily is None:
                continue
            prescription.check = DoseCheck(
                limit_daily_mg=limit, limit_basis=f"{renal.param} {renal.band.range} 구간 1일 최대",
                status="exceeds" if daily > limit else "within",
            )

    def _dose_rules_for(self, record: Mapping[str, Any], facts: PatientFacts) -> list[DoseConditionRule]:
        """용량 조건 규칙을 평가해 ``met`` 또는 ``possible``(알 수 없는 입력 포함)인 것을 돌려준다.

        역할: ``initiation``(시작 시)과 ``rules``의 규칙마다 ``when``을 평가해 거짓이 아닌 것만 남긴다. 알 수 없는 입력은
        규칙별로 두고 항목의 ``missing``에는 넣지 않는다(항목의 ``missing``은 계산 자체에 필요한 것만 담는다).
        규칙 자신의 필드(``alert``, ``dose`` …)는 그대로 따라온다(``DoseConditionRule``은 추가 키를 허용). 그래서 dict로
        모은 뒤 다 채워지면 한 번에 검증한다.
        입력:
            record: ``dose_rules`` 레코드.
            facts: 환자 사실.
        출력: ``DoseConditionRule`` 목록.
        """
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
                results.append(DoseConditionRule.model_validate(entry))
        return results


def most_severe(alerts: Iterable[object]) -> str | None:
    """알려진 경고 단계 중 가장 높은 것(contraindicated > high > caution·medium > info). 없으면 ``None``."""
    ranked = [alert for alert in alerts if isinstance(alert, str) and alert in _ALERT_ORDER]
    return min(ranked, key=_ALERT_ORDER.__getitem__) if ranked else None
