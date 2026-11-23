"""평가 페이로드(API ``assessment``, SSE ``meta.assessment``, 프롬프트 입력)의 Pydantic 스키마.

설계상 위치: 임상 추천 패키지의 출력 계약. ``presentation.Presenter``(2단계 항목, 검사값 해석),
``services.dur_check.DurChecker``(DUR 경고), ``doses.DoseStage``(용량)가 이 모델을 직접 만들고,
``model.Assessment.payload``가 조립한다. API 응답 모델과 ``schema_doc``(``docs/assessment_schema.md`` 표 생성)이 읽는다.

역할: 생산자가 모델을 직접 만들므로 mypy가 필드 이름을 검사하고, 잘못된 값은 만드는 자리에서 실패한다
(``extra="forbid"``). ``/docs``에 실제 구조가 보인다. 말뭉치나 다른 모듈에서 오는 열린 구조(용량 조건 규칙,
DUR 출처, 메모·질문 소견 충돌)만 경계에서 dict로부터 검증한다.

직렬화는 지정된 키만 정확히 남긴다(``PayloadModel``): 해당하지 않는 선택 키는 빠지고, ``null``로 지정한 키는
``null``로 남는다(``docs/assessment_schema.md`` 참고). 키 순서는 생산자가 넘긴 순서와 관계없이 필드 선언 순서
(상속한 출처 필드 먼저)다.

타입: 코드가 계산하는 값(상태, 우선순위, 종류, 근거 번호)은 엄격하게, 말뭉치 레코드에서 오는 텍스트는 ``Text``
(문자열 또는 문자열 목록)로, 말뭉치가 직접 정의하는 구조(용량 조건 규칙의 추가 필드, 참고 값)는 열어 둔다.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, SerializerFunctionWrapHandler, model_serializer

Text = str | list[str]
Number = int | float
Status = Literal["met", "possible"]
Priority = Literal["primary", "reference"]
Region = Literal["KR", "US", "EU", "INTL"]
ItemKind = Literal["diagnosis", "referral", "workup", "monitoring"]


class PayloadModel(BaseModel):
    """알 수 없는 키를 막고, 지정된 키(생성 시 넘겼거나 ``evidence_refs``처럼 나중에 대입한 키)만 선언 순서로 직렬화하는 기반 모델.

    역할: 모든 페이로드 모델의 공통 기반. 호출하는 곳에서 ``exclude_unset``을 쓰지 않고 직렬화기를 두는 이유는,
    페이로드가 API 응답 모델 안에 들어가는데 그 응답 모델의 기본값(``source``)은 계속 써야 하고, 직렬화기는
    모든 중첩 단계에 적용되기 때문이다.
    """

    model_config = ConfigDict(extra="forbid")

    @model_serializer(mode="wrap")
    def _set_keys_only(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        """기본 직렬화 결과에서 지정된 필드와 추가 키만 남긴다."""
        present = self.model_fields_set | set(self.model_extra or {})
        return {key: value for key, value in handler(self).items() if key in present}


class Source(PayloadModel):
    """인용한 출처(국내 출처가 먼저 온다). 출처 표시에 필요한 항목만 담는다.

    역할: ``Provenance.sources``의 원소. 화면 출처 표시와 ``prompt_view``의 ``source_titles``에 쓰인다.
    """

    id: str | None = None
    title: str | None = None
    citation: str | None = None
    locator: str | None = Field(default=None, description="원문 안의 위치(쪽·표·권고 번호 등)")
    url: str | None = None
    license: str | None = None
    region: Region | None = None


class MissingInput(PayloadModel):
    """``possible`` 결과를 확정하려면 의료진이 확인해야 하는 입력.

    역할: 2단계 항목·DUR 경고·용량 항목의 ``missing`` 원소. 2단계·용량 항목에서는 ``Presenter.missing``·``missing_value``가 만든다.
    """

    kind: str = Field(description="value(수치) · finding · event · risk · indication · lab_present · dx_history · dx · med · med_class")
    key: str = Field(description="소견 키·개념 ID·KCD 접두어 등. 소견이면 요청 ``findings``의 키로 그대로 쓸 수 있다")
    label: str = Field(description="화면 표시용 이름")


class Provenance(PayloadModel):
    """규칙 항목의 출처 정보(지역, 우선순위, 대체 관계, 출처 목록).

    역할: 2단계 항목·검사값 해석·용량 항목·근거 문단이 상속하는 공통 출처 필드(``Presenter.provenance``가 채운다).
    """

    region: Region | None = Field(default=None, description="출처 중 우선순위가 가장 높은 지역(KR > US > EU > INTL)")
    region_label: str | None = Field(default=None, description="국내 · 미국 · 유럽 · 국제·기타 · 미상")
    priority: Priority | None = Field(default=None, description="reference: 같은 주제의 국내 기준에 밀려 참고로만 보이는 항목")
    superseded_by: list[str] | None = Field(default=None, description="이 항목을 대체하는 상위(국내) 기준의 claim ID. priority=reference일 때만")
    sources: list[Source] | None = Field(default=None, description="출처(국내 먼저)")


class TestEntry(PayloadModel):
    """추천 검사 하나.

    역할: 추천 검사·추적 항목의 ``tests`` 원소(``Presenter.test_entry``가 만든다).
    """

    id: str = Field(description="검사 개념 ID")
    name: str
    category: str = Field(description="검체검사 · 영상검사 · 내시경검사 · 점수·위험도 평가 · 예방접종 등")
    reason: Text | None = None


class Diagnosis(PayloadModel):
    """의심 진단의 이름과 KCD 코드.

    역할: ``RecommendationItem.diagnosis``(kind=diagnosis일 때만).
    """

    name: str | None = None
    kcd: list[str] = Field(description="해당 KCD 코드(접두어)")


class RecommendationItem(Provenance):
    """의심 진단 · 협진 · 추천 검사 · 추적 검사 규칙 중 환자에게 ``met`` 또는 ``possible``인 항목.

    역할: ``AssessmentPayload.items``의 원소. ``Presenter.item``이 만들고, 근거 번호는 정렬 후 ``evidence_refs``에 대입된다.
    """

    record_id: str | None
    claim_id: str | None
    kind: ItemKind = Field(
        description="diagnosis(의심 진단) · referral(협진) · workup(추천 검사) · monitoring(추적 검사)")
    status: Status = Field(description="met: 조건 충족 / possible: 일부만 확인됨 — 나머지는 missing")
    topic: Text | None
    chapter: Text | None
    diagnosis: Diagnosis | None = Field(default=None, description="kind=diagnosis일 때만")
    already_recorded: bool | None = Field(default=None, description="kind=diagnosis일 때만: 이미 기록된 상병인지")
    severity: Text | None = None
    referral_to: Text | None = Field(default=None, description="kind=referral일 때만: 협진 대상")
    purpose: Text | None = None
    tests: list[TestEntry] | None = Field(default=None, description="kind=workup·monitoring일 때만")
    interval: dict[str, Any] | None = Field(default=None, description="kind=monitoring일 때만: 규칙의 interval_* · within_* · timing_* 값")
    actions: list[Any] | None = Field(default=None, description="kind=monitoring: 충족된 추적 조건의 조치")
    strength: Text | None = Field(default=None, description="권고 강도")
    note: Text | None = None
    matched: list[str] = Field(description="충족된 조건(환자 값 포함)")
    missing: list[MissingInput] = Field(description="확인되지 않은 조건")
    review_status: str | None = Field(description="approved · draft(경고 문구 없이 데이터로만 둔다)")
    conflicts: list[Any] | None = Field(default=None, description="같은 주제의 다른 출처와 다른 점")
    evidence_refs: list[int] | None = Field(default=None, description="응답 evidence의 번호([근거 n])")


class LabInterpretation(Provenance):
    """검사값이 속한 기준 구간.

    역할: ``AssessmentPayload.lab_interpretations``의 원소(``Presenter.lab_interpretations``가 만든다).
    """

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
    """규칙에 반영되지 못한 EMR 항목(사이트 코드 매핑으로 해결).

    역할: ``emr_mapping``의 매핑 문제를 페이로드로 옮긴 것. 사이트 코드 맵 보완을 돕는다.
    """

    kind: str = Field(description="lab_unmapped · lab_unit_unknown · lab_unit_unsupported · lab_out_of_range · lab_non_numeric · "
                                  "lab_not_single · med_unmapped · med_partial")
    code: str | None = None
    name: str | None = None
    detail: str
    suggestion: str | None = Field(default=None, description="매핑할 개념 ID 제안")


# ── DUR 점검 ───────────────────────────────────────────────────────────────
class DurProduct(PayloadModel):
    """DUR 경고에 해당하는 제품 하나.

    역할: ``DurAlert.products``의 원소. 질문에서 언급한 약이면 ``proposed``가 true다.
    """

    med_code: str | None
    med_name: str | None
    ingredient_name: str | None
    proposed: bool | None = Field(default=None, description="true: 현재 처방이 아니라 질문에서 언급한 약(추가 시 점검)")


class DurNotice(PayloadModel):
    """DUR 항목의 근거 고시·공고.

    역할: ``DurAlert.notice``.
    """

    no: str | None = Field(description="고시·공고 번호")
    date: str | None


class DurAlert(PayloadModel):
    """심평원 DUR 목록과 맞는 현재 처방.

    역할: ``MedicationAlerts.alerts``의 원소(``DurChecker``가 만든다).
    """

    type: Literal["병용금기", "연령금기", "임부금기", "노인주의"]
    status: Status = Field(description="possible: 임신 여부 미확인(가임기 여성)")
    products: list[DurProduct]
    detail: str | None
    notice: DurNotice
    condition_note: str | None = Field(default=None, description="목록의 비고(평가하지 않고 그대로 보임)")
    age_condition: str | None = Field(default=None, description="연령금기: 예) 12세 미만")
    grade: str | None = Field(default=None, description="임부금기 등급")
    category: str | None = Field(default=None, description="노인주의 구분")
    note: str | None = None
    missing: list[MissingInput] | None = None


class DurUnmapped(PayloadModel):
    """DUR 목록에 없는 제품코드(점검 제외).

    역할: ``MedicationAlerts.unmapped``의 원소. 점검하지 못한 처방을 알린다.
    """

    med_code: str | None
    med_name: str | None
    detail: str


class DurSource(PayloadModel):
    """DUR 목록의 출처 표시 정보.

    역할: ``MedicationAlerts.source``. DUR 저장소의 출처 메타데이터를 경계에서 검증한다.
    """

    id: str | None = None
    title: str | None = None
    url: str | None = None
    license: str | None = None
    region: str | None = None


class DurProposed(PayloadModel):
    """질문에서 언급해 함께 점검한 약(현재 처방에 같은 성분이 있으면 처방으로만 점검).

    역할: ``MedicationAlerts.proposed``의 원소.
    """

    name: str = Field(description="질문에 쓰인 이름")
    ingredient_name: str


class MedicationAlerts(PayloadModel):
    """DUR 점검 결과(공공누리 제1유형 — source로 출처 표시).

    역할: ``AssessmentPayload.medication_alerts``. DUR 점검기가 설정된 경우에만 온다.
    """

    alerts: list[DurAlert]
    unmapped: list[DurUnmapped]
    source: DurSource
    proposed: list[DurProposed] | None = Field(default=None, description="질문에서 언급해 추가 시 점검한 약")


# ── 3단계: 용량 ────────────────────────────────────────────────────────
class DoseDrug(PayloadModel):
    """용량 항목의 대상 약(성분 또는 계열).

    역할: ``DoseAssessment.drug``.
    """

    id: str
    name: str
    type: Literal["ingredient", "class"]


class DoseCheck(PayloadModel):
    """처방 1일 용량과 신기능 구간 1일 최대량의 대조.

    역할: ``DosePrescription.check``(``DoseStage._dose_check``가 채운다).
    """

    limit_daily_mg: Number
    limit_basis: str
    status: Literal["exceeds", "within"]


class DosePrescription(PayloadModel):
    """용량 대상에 해당하는 현재 처방 하나.

    역할: ``DoseAssessment.prescriptions``의 원소. EMR 원 용량과 계산한 1일 용량, 신기능 구간 대조 결과를 담는다.
    """

    med_code: str | None
    med_name: str | None
    dose: dict[str, Any] | None = Field(default=None, description="EMR의 dosage · unit · frequency · interval · durationDays 원값")
    daily_mg: Number | None = Field(default=None, description="단위·횟수가 명확할 때만 계산한 1일 용량(mg)")
    daily_basis: str | None = None
    check: DoseCheck | None = None


class DoseCalculation(PayloadModel):
    """체중 기반 등 용량 공식의 계산값. 입력이 없으면 missing만 온다.

    역할: ``DoseAssessment.calculations``의 원소.
    """

    name: str
    formula: str
    unit: str | None = None
    infuse_h: Any = None
    note: Any = None
    missing: list[MissingInput] | None = None
    inputs: dict[str, Number | list[Number]] | None = Field(default=None, description="계산에 쓴 입력값")
    value: Number | None = None
    range: list[Number] | None = Field(default=None, description="[최소, 최대] 범위 공식의 결과")


class LabeledValue(PayloadModel):
    """표시 이름이 붙은 키·값 한 쌍.

    역할: 용량 규칙의 참고 필드(``DoseAssessment.reference``)와 신기능 구간 용량(``RenalBand.dose``)의 원소.
    """

    key: str
    label: str
    value: Any


class RenalBand(PayloadModel):
    """환자 신기능 값이 속한 규칙 구간.

    역할: ``DoseRenal.band``. 구간 범위 이름, 구간의 용량·조치, 경고 단계를 담는다.
    """

    range: str = Field(description="예) 30 이상 45 미만 mL/min/1.73m2")
    dose: list[LabeledValue]
    alert: str | None = None


class DoseRenal(PayloadModel):
    """신기능(CrCl·eGFR) 값과 해당 구간.

    역할: ``DoseAssessment.renal``(``DoseStage._dose_renal``이 만든다).
    """

    param: str = Field(description="CrCl · eGFR")
    label: str
    method: str | None = None
    missing: list[MissingInput] | None = None
    value: Number | None = None
    unit: str | None = None
    test_date: str | None = None
    basis: str | None = Field(default=None, description="검사실 보고값 또는 계산식")
    method_note: str | None = None
    band: RenalBand | None = Field(default=None, description="null: 어느 구간에도 속하지 않음(band_note)")
    band_note: str | None = None


class DoseConditionRule(PayloadModel):
    """충족(met) 또는 일부 확인(possible)된 용량 조건 규칙.

    역할: ``DoseAssessment.rules``의 원소. 규칙 레코드의 다른 필드를 그대로 허용하는 열린 모델이다.
    """

    model_config = ConfigDict(extra="allow")

    status: Status
    phase: str | None = Field(default=None, description="시작 시(initiation 규칙)")
    matched: list[str]
    missing: list[MissingInput] | None = None


class DoseAssessment(Provenance):
    """처방 중이거나 의료진이 지정한 약의 참고 용량(3단계). 값마다 입력·공식·출처가 함께 온다.

    역할: ``AssessmentPayload.dose_assessments``의 원소(``DoseStage``가 만든다).
    """

    record_id: str | None
    claim_id: str | None
    kind: Literal["dose"]
    topic: Text | None
    drug: DoseDrug
    anchor: list[Literal["prescribed", "requested"]] = Field(description="prescribed: 현재 처방 / requested: 요청 dose_drugs")
    indication: Text | None = None
    prescriptions: list[DosePrescription] | None = None
    not_applied: str | None = Field(default=None, description="성인 기준을 적용하지 않은 이유(18세 미만)")
    calculations: list[DoseCalculation] | None = None
    renal: DoseRenal | None = None
    rules: list[DoseConditionRule] | None = None
    reference: list[LabeledValue] | None = Field(default=None, description="규칙의 나머지 용량 정보(참고 텍스트)")
    alert_note: Text | None = None
    conflicts: list[Any] | None = None
    missing: list[MissingInput] = Field(description="계산에 필요한데 없는 입력")
    alert: str | None = Field(description="충족된 경고 중 가장 높은 단계(contraindicated > high > caution·medium > info)")
    review_status: str | None
    evidence_refs: list[int] | None = None


class UnmatchedDoseDrug(PayloadModel):
    """용량 평가를 요청했지만 대상으로 해석하지 못한 약.

    역할: ``AssessmentPayload.unmatched_dose_drugs``의 원소(``DoseStage.resolve``가 만든다).
    """

    input: str
    reason: str = Field(description="성분을 확인할 수 없음 · 말뭉치에 이 약의 용량 규칙 없음")


class FindingConflict(PayloadModel):
    """진료 메모와 질문이 서로 다르게 적은 소견. 어느 쪽도 적용하지 않고(findings_used에서 빠짐) 확인을 요청한다.

    역할: ``AssessmentPayload.finding_conflicts``의 값. ``question_facts.merge_question_findings``의 결과를 검증한다.
    """

    clinical_note: bool = Field(description="진료 메모(메모 추출 포함)의 값")
    question: bool = Field(description="질문 텍스트의 값")


class Evidence(Provenance):
    """인용한 최신화 근거 문단 하나(채팅·추천 응답의 ``evidence``, SSE ``meta.evidence``). ``n``은 답변의 ``[근거 n]``.

    역할: 응답의 근거 목록 원소. 평가 항목의 ``evidence_refs`` 번호와 같은 번호 체계를 쓴다.
    """

    n: int = Field(description="답변의 [근거 n]·평가 항목의 evidence_refs 번호")
    chunk_id: str
    claim_id: str
    title: str
    text: str
    review_status: str = Field(description="approved · draft (데이터로만 둔다)")
    conflicts: list[str] = Field(description="이 문단과 내용이 충돌하는 claim ID")
    updated_at: str | None = None


class AssessmentPayload(PayloadModel):
    """환자 한 명의 규칙 평가 결과(2단계, DUR, 켜져 있으면 3단계 용량).

    역할: API ``assessment``·SSE ``meta.assessment``·프롬프트 입력의 최상위 모델(``Assessment.payload``가 조립한다).
    """

    items: list[RecommendationItem] = Field(description="국내 기준 우선, met 우선 순서")
    lab_interpretations: list[LabInterpretation]
    findings_used: dict[str, bool] = Field(description="적용한 소견(요청 findings가 메모 추출보다 우선)")
    unknown_findings: list[str] = Field(description="규칙이 쓰지 않는 소견 키(무시됨)")
    mapping_issues: list[MappingIssue]
    medication_alerts: MedicationAlerts | None = Field(default=None, description="DUR 점검이 설정된 경우에만")
    dose_assessments: list[DoseAssessment] | None = Field(default=None, description="용량 단계(AX_G_CLINICAL_DOSES_ENABLED)가 켜진 경우에만")
    unmatched_dose_drugs: list[UnmatchedDoseDrug] | None = Field(default=None, description="용량 단계에서 요청 약 중 규칙이 없는 것(있을 때만)")
    finding_conflicts: dict[str, FindingConflict] | None = Field(
        None, description="메모와 질문이 서로 다른 소견(있을 때만). 질문에 출산·유산·새 임신 같은 상태 변화 표현이 있으면 충돌이 아니라 질문 값을 쓴다",
    )
