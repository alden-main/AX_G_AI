"""Pydantic schema of the assessment payload (API ``assessment``, SSE ``meta.assessment``, prompt input).

The recommender builds plain dicts; ``Assessment.payload`` validates them against these models,
so a misspelt or unexpected key fails at creation (``extra="forbid"``) and ``/docs`` shows the
real structure. Serialization keeps exactly the keys that were set: an optional key that does
not apply is omitted, a key set to ``null`` stays ``null`` (the JSON is the same as before the
schema existed; see ``docs/assessment_schema.md``).

Types: values the code computes (status, priority, kinds, evidence numbers) are strict; text that
comes from corpus records is ``Text`` (a string or a list of strings); structures defined by the
corpus itself (extra fields of dose-condition rules, reference values) stay open.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal, Self

from pydantic import (
    BaseModel, ConfigDict, Field, ModelWrapValidatorHandler, PrivateAttr, SerializerFunctionWrapHandler, model_serializer,
    model_validator,
)

Text = str | list[str]
Number = int | float
Status = Literal["met", "possible"]
Priority = Literal["primary", "reference"]
Region = Literal["KR", "US", "EU", "INTL"]


class PayloadModel(BaseModel):
    """Forbids unknown keys; serializes only the keys that were set, in the order they were given.

    The order matters: the payload is also the prompt input, and an unchanged byte order keeps the
    inference server's prefix cache warm.
    """

    model_config = ConfigDict(extra="forbid")
    _key_order: tuple[str, ...] = PrivateAttr(default=())

    @model_validator(mode="wrap")
    @classmethod
    def _remember_key_order(cls, data: Any, handler: ModelWrapValidatorHandler[Self]) -> Self:
        model = handler(data)
        if isinstance(data, Mapping):
            model._key_order = tuple(data)
        return model

    @model_serializer(mode="wrap")
    def _set_keys_in_order(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        data = handler(self)
        present = self.model_fields_set | set(self.model_extra or {})
        order = [key for key in self._key_order if key in data] + [key for key in data if key not in self._key_order]
        return {key: data[key] for key in order if key in present}


class Source(PayloadModel):
    """인용한 출처(국내 출처가 먼저 온다). 출처 표시에 필요한 항목만 담는다."""

    id: str | None = None
    title: str | None = None
    citation: str | None = None
    locator: str | None = Field(None, description="원문 안의 위치(쪽·표·권고 번호 등)")
    url: str | None = None
    license: str | None = None
    region: Region | None = None


class MissingInput(PayloadModel):
    """``possible`` 결과를 확정하려면 의료진이 확인해야 하는 입력."""

    kind: str = Field(description="value(수치) · finding · event · risk · indication · lab_present · dx_history · dx · med · med_class")
    key: str = Field(description="소견 키·개념 ID·KCD 접두어 등. 소견이면 요청 ``findings``의 키로 그대로 쓸 수 있다")
    label: str = Field(description="화면 표시용 이름")


class Provenance(PayloadModel):
    region: Region | None = Field(None, description="출처 중 우선순위가 가장 높은 지역(KR > US > EU > INTL)")
    region_label: str | None = Field(None, description="국내 · 미국 · 유럽 · 국제·기타 · 미상")
    priority: Priority | None = Field(None, description="reference: 같은 주제의 국내 기준에 밀려 참고로만 보이는 항목")
    superseded_by: list[str] | None = Field(None, description="이 항목을 대체하는 상위(국내) 기준의 claim ID. priority=reference일 때만")
    sources: list[Source] | None = Field(None, description="출처(국내 먼저)")


class TestEntry(PayloadModel):
    """추천 검사 하나."""

    id: str = Field(description="검사 개념 ID")
    name: str
    category: str = Field(description="검체검사 · 영상검사 · 내시경검사 · 점수·위험도 평가 · 예방접종 등")
    reason: Text | None = None


class Diagnosis(PayloadModel):
    name: str | None = None
    kcd: list[str] = Field(description="해당 KCD 코드(접두어)")


class RecommendationItem(Provenance):
    """의심 진단 · 협진 · 추천 검사 · 추적 검사 규칙 중 환자에게 ``met`` 또는 ``possible``인 항목."""

    record_id: str | None
    claim_id: str | None
    kind: Literal["diagnosis", "referral", "workup", "monitoring"] = Field(
        description="diagnosis(의심 진단) · referral(협진) · workup(추천 검사) · monitoring(추적 검사)")
    status: Status = Field(description="met: 조건 충족 / possible: 일부만 확인됨 — 나머지는 missing")
    topic: Text | None
    chapter: Text | None
    diagnosis: Diagnosis | None = Field(None, description="kind=diagnosis일 때만")
    already_recorded: bool | None = Field(None, description="kind=diagnosis일 때만: 이미 기록된 상병인지")
    severity: Text | None = None
    referral_to: Text | None = Field(None, description="kind=referral일 때만: 협진 대상")
    purpose: Text | None = None
    tests: list[TestEntry] | None = Field(None, description="kind=workup·monitoring일 때만")
    interval: dict[str, Any] | None = Field(None, description="kind=monitoring일 때만: 규칙의 interval_* · within_* · timing_* 값")
    actions: list[Any] | None = Field(None, description="kind=monitoring: 충족된 추적 조건의 조치")
    strength: Text | None = Field(None, description="권고 강도")
    note: Text | None = None
    matched: list[str] = Field(description="충족된 조건(환자 값 포함)")
    missing: list[MissingInput] = Field(description="확인되지 않은 조건")
    review_status: str | None = Field(description="approved · draft(경고 문구 없이 데이터로만 둔다)")
    conflicts: list[Any] | None = Field(None, description="같은 주제의 다른 출처와 다른 점")
    evidence_refs: list[int] | None = Field(None, description="응답 evidence의 번호([근거 n])")


class LabInterpretation(Provenance):
    """검사값이 속한 기준 구간."""

    record_id: str | None
    claim_id: str | None
    lab: str = Field(description="검사 개념 ID")
    name: str
    value: Number = Field(description="기준 단위로 환산한 환자 값")
    unit: str | None
    test_date: str | None
    band: str | None = Field(description="구간 이름")
    context: Text | None
    review_status: str | None
    evidence_refs: list[int] | None = None


class MappingIssue(PayloadModel):
    """규칙에 반영되지 못한 EMR 항목(사이트 코드 매핑으로 해결)."""

    kind: str = Field(description="lab_unmapped · lab_unit_unknown · lab_unit_unsupported · lab_out_of_range · lab_non_numeric · "
                                  "lab_not_single · med_unmapped · med_partial")
    code: str | None = None
    name: str | None = None
    detail: str
    suggestion: str | None = Field(None, description="매핑할 개념 ID 제안")


# ── DUR ───────────────────────────────────────────────────────────────────
class DurProduct(PayloadModel):
    med_code: str | None
    med_name: str | None
    ingredient_name: str | None
    proposed: bool | None = Field(None, description="true: 현재 처방이 아니라 질문에서 언급한 약(추가 시 점검)")


class DurNotice(PayloadModel):
    no: str | None = Field(description="고시·공고 번호")
    date: str | None


class DurAlert(PayloadModel):
    """심평원 DUR 목록과 맞는 현재 처방."""

    type: Literal["병용금기", "연령금기", "임부금기", "노인주의"]
    status: Status = Field(description="possible: 임신 여부 미확인(가임기 여성)")
    products: list[DurProduct]
    detail: str | None
    notice: DurNotice
    condition_note: str | None = Field(None, description="목록의 비고(평가하지 않고 그대로 보임)")
    age_condition: str | None = Field(None, description="연령금기: 예) 12세 미만")
    grade: str | None = Field(None, description="임부금기 등급")
    category: str | None = Field(None, description="노인주의 구분")
    note: str | None = None
    missing: list[MissingInput] | None = None


class DurUnmapped(PayloadModel):
    """DUR 목록에 없는 제품코드(점검 제외)."""

    med_code: str | None
    med_name: str | None
    detail: str


class DurSource(PayloadModel):
    id: str | None = None
    title: str | None = None
    url: str | None = None
    license: str | None = None
    region: str | None = None


class DurProposed(PayloadModel):
    """질문에서 언급해 함께 점검한 약(현재 처방에 같은 성분이 있으면 처방으로만 점검)."""

    name: str = Field(description="질문에 쓰인 이름")
    ingredient_name: str


class MedicationAlerts(PayloadModel):
    """DUR 점검 결과(공공누리 제1유형 — source로 출처 표시)."""

    alerts: list[DurAlert]
    unmapped: list[DurUnmapped]
    source: DurSource
    proposed: list[DurProposed] | None = Field(None, description="질문에서 언급해 추가 시 점검한 약")


# ── stage 3: doses ────────────────────────────────────────────────────────
class DoseDrug(PayloadModel):
    id: str
    name: str
    type: Literal["ingredient", "class"]


class DoseCheck(PayloadModel):
    """처방 1일 용량과 신기능 구간 1일 최대량의 대조."""

    limit_daily_mg: Number
    limit_basis: str
    status: Literal["exceeds", "within"]


class DosePrescription(PayloadModel):
    med_code: str | None
    med_name: str | None
    dose: dict[str, Any] | None = Field(None, description="EMR의 dosage · unit · frequency · interval · durationDays 원값")
    daily_mg: Number | None = Field(None, description="단위·횟수가 명확할 때만 계산한 1일 용량(mg)")
    daily_basis: str | None = None
    check: DoseCheck | None = None


class DoseCalculation(PayloadModel):
    """체중 기반 등 용량 공식의 계산값. 입력이 없으면 missing만 온다."""

    name: str
    formula: str
    unit: str | None = None
    infuse_h: Any = None
    note: Any = None
    missing: list[MissingInput] | None = None
    inputs: dict[str, Number | list[Number]] | None = Field(None, description="계산에 쓴 입력값")
    value: Number | None = None
    range: list[Number] | None = Field(None, description="[최소, 최대] 범위 공식의 결과")


class LabeledValue(PayloadModel):
    key: str
    label: str
    value: Any


class RenalBand(PayloadModel):
    range: str = Field(description="예) 30 이상 45 미만 mL/min/1.73m2")
    dose: list[LabeledValue]
    alert: str | None = None


class DoseRenal(PayloadModel):
    """신기능(CrCl·eGFR) 값과 해당 구간."""

    param: str = Field(description="CrCl · eGFR")
    label: str
    method: str | None = None
    missing: list[MissingInput] | None = None
    value: Number | None = None
    unit: str | None = None
    test_date: str | None = None
    basis: str | None = Field(None, description="검사실 보고값 또는 계산식")
    method_note: str | None = None
    band: RenalBand | None = Field(None, description="null: 어느 구간에도 속하지 않음(band_note)")
    band_note: str | None = None


class DoseConditionRule(PayloadModel):
    """충족(met) 또는 일부 확인(possible)된 용량 조건 규칙."""

    model_config = ConfigDict(extra="allow")

    status: Status
    phase: str | None = Field(None, description="시작 시(initiation 규칙)")
    matched: list[str]
    missing: list[MissingInput] | None = None


class DoseAssessment(Provenance):
    """처방 중이거나 의료진이 지정한 약의 참고 용량(3단계). 값마다 입력·공식·출처가 함께 온다."""

    record_id: str | None
    claim_id: str | None
    kind: Literal["dose"]
    topic: Text | None
    drug: DoseDrug
    anchor: list[Literal["prescribed", "requested"]] = Field(description="prescribed: 현재 처방 / requested: 요청 dose_drugs")
    indication: Text | None = None
    prescriptions: list[DosePrescription] | None = None
    not_applied: str | None = Field(None, description="성인 기준을 적용하지 않은 이유(18세 미만)")
    calculations: list[DoseCalculation] | None = None
    renal: DoseRenal | None = None
    rules: list[DoseConditionRule] | None = None
    reference: list[LabeledValue] | None = Field(None, description="규칙의 나머지 용량 정보(참고 텍스트)")
    alert_note: Text | None = None
    conflicts: list[Any] | None = None
    missing: list[MissingInput] = Field(description="계산에 필요한데 없는 입력")
    alert: str | None = Field(description="충족된 경고 중 가장 높은 단계(contraindicated > high > caution·medium > info)")
    review_status: str | None
    evidence_refs: list[int] | None = None


class UnmatchedDoseDrug(PayloadModel):
    input: str
    reason: str = Field(description="성분을 확인할 수 없음 · 말뭉치에 이 약의 용량 규칙 없음")


class AssessmentPayload(PayloadModel):
    """환자 한 명의 규칙 평가 결과(2단계, DUR, 켜져 있으면 3단계 용량)."""

    items: list[RecommendationItem] = Field(description="국내 기준 우선, met 우선 순서")
    lab_interpretations: list[LabInterpretation]
    findings_used: dict[str, bool] = Field(description="적용한 소견(요청 findings가 메모 추출보다 우선)")
    unknown_findings: list[str] = Field(description="규칙이 쓰지 않는 소견 키(무시됨)")
    mapping_issues: list[MappingIssue]
    medication_alerts: MedicationAlerts | None = Field(None, description="DUR 점검이 설정된 경우에만")
    dose_assessments: list[DoseAssessment] | None = Field(None, description="용량 단계(AX_G_CLINICAL_DOSES_ENABLED)가 켜진 경우에만")
    unmatched_dose_drugs: list[UnmatchedDoseDrug] | None = Field(None, description="용량 단계에서 요청 약 중 규칙이 없는 것(있을 때만)")
