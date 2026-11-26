"""프롬프트에 싣는 데이터(EMR 요약·규칙 평가·근거)의 영문 키와 상태값을 한국어 이름으로 바꾼다.

설계상 위치: 프롬프트 계층. ``prompts.clinical``의 ``compose_*`` 함수가 JSON을 프롬프트에 실을 때만 쓴다. API 응답과
화면용 데이터의 키는 바꾸지 않는다.

역할: 작은 자체 모델은 입력 JSON의 영문 키(``recorded_none``, ``medication_alerts``)와 상태값(``met``)을 답변에 그대로
옮기는 일이 잦다. 입력에 없는 단어는 출력에도 나오지 않으므로 모델이 보는 이름을 의료진의 한국어 용어로 바꾼다. 규칙 문구
(``prompts.clinical``)도 같은 이름을 가리킨다(⟪이름⟫ 표시자, ``render_rule``). 사전에 없는 키는 그대로 두고, 모델에 필요 없는
내부 식별자는 뺀다. 영어로 학습한 모델이 영문 키를 더 정확히 읽는지 비교할 수 있도록 영문 키 표기(``english``)도 둔다.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any, Final, Literal

# 프롬프트 데이터 표기 방식. korean: 한국어 항목 이름(기본), english: 영문 키(비교 실험용, ``AX_G_PROMPT_DATA_LABELS``).
PromptLabels = Literal["korean", "english"]

# 모델이 쓸 일이 없는 내부 식별자·코드(답변에 옮겨 적는 일만 생긴다). 표시 이름(``label``·``name``)은 남는다.
_DROP: Final = frozenset({"id", "key", "region", "review_notice", "chunk_id", "claim_id", "record_id", "review_status", "updated_at"})

KEY_NAMES: Final[Mapping[str, str]] = {
    # EMR 요약(main_bridge.RawEmrPayloadMapper)
    "calculation_context": "환자 기본 정보", "age_years": "나이(세)", "sex": "성별", "weight_kg": "체중(kg)", "height_cm": "키(cm)",
    "serum_creatinine_mg_dl": "혈청 크레아티닌(mg/dL)", "blood_pressure": "혈압", "blood_sugar": "혈당", "oxygen_saturation": "산소포화도",
    "measured_at": "측정 시각", "sbp": "수축기", "dbp": "이완기", "timing_type": "측정 시점", "lab_results": "검사 결과",
    "exam_code": "검사 코드", "item_name": "검사명", "prescriptions": "처방", "undated_prescriptions": "처방(처방일 기록 없음)", "med_code": "제품코드", "med_name": "약품명",
    "dosage": "1회 용량", "frequency": "1일 횟수", "duration_days": "처방 일수", "durationDays": "처방 일수", "instructions": "용법",
    "prescriptionDate": "처방일", "prescription_date": "처방일", "diagnoses": "진단", "name_ko": "진단명", "code": "코드",
    "calculation_results": "계산값", "calculation_input_needed": "계산에 필요한 입력", "method": "계산 방법",
    "additional_clinical_data": "추가 임상 기록", "recorded_none": "기록상 없음",
    "glycatedHemoglobinList": "당화혈색소", "hba1cValue": "당화혈색소(%)", "testDate": "검사일", "measuredAt": "측정 시각",
    # 공통
    "name": "이름", "value": "값", "unit": "단위", "category": "구분", "kind": "종류", "status": "판정", "type": "유형",
    "label": "항목", "text": "본문", "title": "제목", "note": "비고", "date": "일자", "reason": "이유", "detail": "사유",
    # 규칙 평가(assessment)
    "items": "규칙 평가 항목", "topic": "주제", "diagnosis": "진단", "kcd": "KCD 코드", "tests": "추천 검사",
    "referral_to": "협진 대상", "matched": "충족 조건", "missing": "확인 필요 정보", "already_recorded": "이미 기록된 진단",
    "evidence_refs": "근거 번호", "source_titles": "출처명", "sources": "출처", "locator": "위치", "region_label": "지역",
    "priority": "우선순위", "superseded_by": "대체한 국내 기준", "conflicts": "지침 간 차이", "lab_interpretations": "검사값 해석",
    "lab": "검사", "test_date": "검사일", "band": "구간", "context": "해석 기준", "findings_used": "적용한 소견",
    "unknown_findings": "알 수 없는 소견", "finding_conflicts": "소견 충돌", "clinical_note": "진료 메모", "question": "질문",
    "actions": "조치", "strength": "권고 강도", "suggestion": "제안", "purpose": "목적", "interval": "간격", "severity": "중증도",
    # DUR
    "medication_alerts": "DUR 점검 결과", "alerts": "경고", "products": "제품", "ingredient_name": "성분명",
    "proposed": "추가 예정 약", "notice": "고시", "no": "번호", "condition_note": "비고", "age_condition": "연령 조건",
    "grade": "등급", "unmapped": "DUR 점검 제외", "source": "출처", "license": "이용 조건", "url": "링크", "citation": "인용",
    # 용량
    "dose_assessments": "용량 결과", "drug": "약", "anchor": "대상", "indication": "적응증", "dose": "처방 용량",
    "daily_mg": "1일 용량(mg)", "daily_basis": "1일 용량 계산 근거", "check": "기준 대조", "limit_daily_mg": "기준 1일 최대(mg)",
    "limit_basis": "기준 근거", "rules": "용량 조건", "alert": "경고 단계", "action": "조치", "reference": "참고 용량",
    "calculations": "계산", "inputs": "입력값", "input": "입력", "formula": "공식", "range": "범위", "renal": "신기능 구간",
    "basis": "근거", "band_note": "구간 비고", "not_applied": "적용 안 함", "alert_note": "경고 비고", "param": "변수",
    "phase": "단계", "infuse_h": "주입 시간(시간)", "method_note": "방법 비고",
    "unmatched_dose_drugs": "용량 규칙 없는 약",
    # 근거 문단
    "n": "번호",
}

# 키별 상태값. 같은 키라도 값이 겹치지 않으므로 키마다 합친 사전 하나를 쓴다.
VALUE_NAMES: Final[Mapping[str, Mapping[str, str]]] = {
    "status": {"met": "충족", "possible": "조건부", "exceeds": "기준 초과", "within": "기준 이내"},
    "priority": {"primary": "우선", "reference": "참고"},
    "sex": {"male": "남성", "female": "여성"},
    "kind": {
        "diagnosis": "진단", "referral": "협진", "workup": "검사", "monitoring": "추적", "dose": "용량",
        "value": "수치", "finding": "소견", "event": "사건", "risk": "위험인자", "indication": "적응증",
        "lab_present": "검사 시행 여부", "dx_history": "진단 이력", "dx": "진단", "med": "약", "med_class": "약 계열",
    },
    "alert": {"contraindicated": "금기", "high": "높음", "caution": "주의", "medium": "중간", "info": "정보"},
    "type": {"ingredient": "성분", "class": "계열"},
    "anchor": {"prescribed": "현재 처방", "requested": "요청 약"},
}


# 소견 키(규칙 어휘)를 키로 쓰는 묶음. 소견 키는 snake_case 영문이므로 알려진 것은 한국어로, 나머지는 공백으로 띄운
# 표시 이름(``presentation``의 확인 필요 정보 ``label``과 같은 형태)으로 바꾼다.
_FINDING_GROUPS: Final = frozenset({"findings_used", "finding_conflicts"})
FINDING_NAMES: Final[Mapping[str, str]] = {"pregnancy": "임신"}


def finding_name(key: str) -> str:
    """소견 키의 프롬프트 표시 이름."""
    return FINDING_NAMES.get(key, key.replace("_", " "))


def korean_view(value: Any) -> Any:
    """``value``의 사전 키와 알려진 상태값을 한국어로 바꾼 새 값을 돌려준다(중첩 사전·목록에 재귀 적용).

    역할: 내부 식별자(``_DROP``)는 빼고, ``VALUE_NAMES``에 있는 상태값은 키에 맞게 바꾼다(목록 값도 원소마다).
        사전에 없는 키·값은 그대로 둔다.
    입력:
        value: JSON으로 직렬화할 값.
    출력: 바꾼 새 값(입력은 고치지 않는다).
    """
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if key in _DROP:
                continue
            names = VALUE_NAMES.get(key)
            if names is not None:
                item = [names.get(entry, entry) if isinstance(entry, str) else entry for entry in item] if isinstance(item, list) \
                    else names.get(item, item) if isinstance(item, str) else item
            if key in _FINDING_GROUPS and isinstance(item, Mapping):
                item = {finding_name(str(finding)): entry for finding, entry in item.items()}
            result[KEY_NAMES.get(key, key)] = korean_view(item)
        return result
    if isinstance(value, list):
        return [korean_view(item) for item in value]
    return value


# 규칙 문구의 ⟪이름⟫ 표시자가 가리키는 영문 키·값(english 표기). 각 이름은 KEY_NAMES 또는 VALUE_NAMES의 한국어 이름과 같다
# (테스트가 확인한다). 같은 한국어 이름이 여러 키에 쓰이는 경우(예: '처방')가 있어 규칙에서 쓰는 뜻을 명시한다.
RULE_KEYS: Final[Mapping[str, str]] = {
    "환자 기본 정보": "calculation_context", "기록상 없음": "recorded_none", "측정 시점": "timing_type",
    "출처": "sources", "제목": "title", "출처명": "source_titles", "번호": "n", "지침 간 차이": "conflicts", "지역": "region_label",
    "우선순위": "priority", "참고": "reference", "판정": "status", "충족": "met", "조건부": "possible", "충족 조건": "matched",
    "확인 필요 정보": "missing", "이미 기록된 진단": "already_recorded", "근거 번호": "evidence_refs", "검사값 해석": "lab_interpretations",
    "적용한 소견": "findings_used", "대체한 국내 기준": "superseded_by", "소견 충돌": "finding_conflicts",
    "DUR 점검 결과": "medication_alerts", "용량 결과": "dose_assessments", "대상": "anchor", "현재 처방": "prescribed",
    "요청 약": "requested", "계산": "calculations", "값": "value", "범위": "range", "입력값": "inputs", "공식": "formula",
    "신기능 구간": "renal", "구간": "band", "근거": "basis", "검사일": "test_date", "처방": "prescriptions", "기준 대조": "check",
    "기준 초과": "exceeds", "1일 용량(mg)": "daily_mg", "1일 용량 계산 근거": "daily_basis", "기준 1일 최대(mg)": "limit_daily_mg",
    "기준 근거": "limit_basis", "적용 안 함": "not_applied", "용량 조건": "rules", "경고 비고": "alert_note", "적응증": "indication",
    "경고 단계": "alert", "금기": "contraindicated", "높음": "high", "처방(처방일 기록 없음)": "undated_prescriptions",
}
_RULE_MARKER = re.compile(r"⟪([^⟫]+)⟫")


def render_rule(template: str, labels: PromptLabels) -> str:
    """규칙 문구의 ⟪이름⟫을 ``korean``이면 '이름'(따옴표), ``english``이면 영문 키로 바꾼다.

    예외: 표시자 이름이 ``RULE_KEYS``에 없으면 ``KeyError``(규칙 문구와 이름표가 어긋남).
    """
    def replace(match: re.Match[str]) -> str:
        key = RULE_KEYS[match.group(1)]  # 이름표에 없는 표시자는 KeyError
        return f"'{match.group(1)}'" if labels == "korean" else key

    return _RULE_MARKER.sub(replace, template)


def prompt_data(value: Any, labels: PromptLabels) -> Any:
    """프롬프트에 실을 데이터: ``korean``이면 ``korean_view``, ``english``이면 내부 식별자만 뺀 원래 키."""
    return korean_view(value) if labels == "korean" else _without_internal(value)


def _without_internal(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _without_internal(item) for key, item in value.items() if key not in _DROP}
    if isinstance(value, list):
        return [_without_internal(item) for item in value]
    return value
