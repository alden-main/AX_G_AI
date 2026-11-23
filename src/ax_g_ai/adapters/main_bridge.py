"""Bridge가 전달한 원문 EMR을 Provider에 보낼 비식별 최소 문맥으로 바꾸는 PoC 전용 변환기.

설계상 위치: 어댑터 계층의 EMR 입력 경계. ``runtime``이 ``RawEmrPayloadMapper``를 만들고, ``api``의 환자 문맥 준비
엔드포인트가 ``map``을 호출하며, 결과(``PreparedRawEmrContext``)는 문맥 저장소에 보관된다.

역할: 이 모듈이 Provider로 가는 환자 문맥의 유일한 비식별 경계다. 알려진 목록은 아는 필드를
정규화하고, 그 밖의 필드(알려진 목록 항목 안의 새 필드 포함)는 직접 식별자를 제거한 뒤 원래 이름으로 보존한다.
빈 목록으로 온 항목은 ``recorded_none``에 남겨 '기록 없음'과 '전달되지 않음'을 구분한다. 규칙 매핑이 읽는
허용 필드(``rule_source``)는 따로 추려 문맥에 함께 담는다.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any
from zoneinfo import ZoneInfo

from ax_g_ai.errors import PatientContextUnavailableError, PatientMismatchError
from ax_g_ai.services.clinical_calculations import calculate_review_aids
from ax_g_ai.services.deidentified_snapshot import deidentify, is_identifier_key
from ax_g_ai.services.patient_context_store import PreparedRawEmrContext
from ax_g_ai.services.ports import ReferenceLookup

# 규칙 매핑(``emr_mapping.map_emr``, ``ClinicalRecommender.prepare``)이 읽는 EMR 필드. 허용 목록이므로 식별자는
# 통과하지 못한다. ``deidentify``와 달리 빈 목록과 null 값을 남기는데, 규칙은 이를 "기록됨: 없음"과 "기록 안 됨"으로
# 구분해 읽는다.
_RULE_LIST_FIELDS: dict[str, tuple[str, ...]] = {
    "labResultList": ("examCode", "itemName", "unit", "value", "testDate"),
    "bloodSugarList": ("glucoseValue", "measuredAt"),
    "glycatedHemoglobinList": ("hba1cValue", "testDate"),
    "bloodPressureList": ("sbp", "dbp", "measuredAt"),
    "oxygenSaturationList": ("spo2Value", "measuredAt"),
    "medicineDtailList": ("medCode", "medName", "dosage", "unit", "frequency", "interval", "durationDays"),
    "diagnosisList": ("diagnosisCode", "kcdCode"),
}
_RULE_CONTEXT_FIELDS = ("age_years", "sex", "weight_kg", "height_cm", "serum_creatinine_mg_dl")
# EMR이 빈 목록으로 보낼 수 있는 목록의 프롬프트 표시 이름(``recorded_none``: "전달 안 됨"이 아니라 "없음으로 기록됨").
_LIST_LABELS = {
    "bloodPressureList": "혈압", "bloodSugarList": "혈당", "oxygenSaturationList": "산소포화도", "labResultList": "검사 결과",
    "glycatedHemoglobinList": "당화혈색소", "medicineDtailList": "처방", "diagnosisList": "진단",
}


def rule_source(source: Mapping[str, Any]) -> dict[str, Any]:
    """규칙 매핑이 읽는 EMR 객체 부분을 JSON으로 바로 저장할 수 있는 데이터로 추린다.

    역할: ``_RULE_LIST_FIELDS``의 목록마다 Mapping 항목의 허용 필드만 남기고(빈 목록도 유지), ``calculationContext``에서는
    ``_RULE_CONTEXT_FIELDS``만 남긴다. 식별자 필드는 허용 목록에 없으므로 들어가지 않는다.
    입력:
        source: Bridge EMR 객체(``emr_source``로 봉투를 벗긴 것).
    출력: 허용 필드만 담은 dict. 목록이 아닌 목록 필드와 Mapping이 아닌 ``calculationContext``는 빠진다.
    """
    result: dict[str, Any] = {}
    for key, fields in _RULE_LIST_FIELDS.items():
        items = source.get(key)
        if isinstance(items, list):
            result[key] = [{name: item[name] for name in fields if name in item} for item in items if isinstance(item, Mapping)]
    context = source.get("calculationContext")
    if isinstance(context, Mapping):
        result["calculationContext"] = {name: context[name] for name in _RULE_CONTEXT_FIELDS if name in context}
    return result


def emr_source(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    """Bridge가 EMR 객체를 그대로 보냈든 ``{"data": ...}`` 봉투에 담아 보냈든 EMR 객체 자체를 돌려준다."""
    raw_data = payload.get("data")
    return raw_data if isinstance(raw_data, Mapping) else payload


class RawEmrPayloadMapper:
    """Bridge 원문 EMR에서 Provider 전송용 비식별 임상 문맥을 만든다.

    역할: 혈압·혈당·산소포화도·검사·처방·진단 목록을 정규화하고 계산 보조값(``calculate_review_aids``)을 붙이며,
    선택적인 참조 DB(``ReferenceLookup``)로 약품명·효능·주의사항과 상병 한글명을 보강해 ``PreparedRawEmrContext``를
    만든다. 만든 문맥의 보관은 호출자(문맥 저장소)가 맡는다.
    """

    _SEOUL = ZoneInfo("Asia/Seoul")
    _HANDLED_SOURCE_FIELDS = frozenset({
        "patient", "bloodPressureList", "bloodSugarList", "oxygenSaturationList",
        "labResultList", "medicineDtailList", "diagnosisList", "calculationContext",
    })

    def __init__(self, reference_lookup: ReferenceLookup | None = None) -> None:
        self._reference_lookup = reference_lookup

    def map(self, patient_id: str, payload: Mapping[str, Any], *, prepared_at: datetime | None = None) -> PreparedRawEmrContext:
        """요청한 환자와 EMR의 환자가 같은지 확인하고 원문 EMR을 비식별 문맥으로 변환한다.

        역할: 봉투를 벗긴 EMR의 ``patient.patientId``를 확인한 뒤 목록별 정규화 결과·계산 결과·계산에 필요한 입력·
        식별자를 뺀 추가 필드·``recorded_none``을 모으고 빈 그룹은 버린다. ``updated_at``은 측정·검사 시각 중 가장
        늦은 값이며, 없으면 ``prepared_at``(없으면 현재 UTC 시각)이다.
        입력:
            patient_id: 요청에서 지정한 환자 ID.
            payload: Bridge가 보낸 EMR 객체 또는 ``{"data": ...}`` 봉투.
            prepared_at: 측정 시각이 하나도 없을 때 쓸 기준 시각. 시간대가 없으면 UTC로 본다. ``None``이면 현재 시각.
        출력: ``data``·``updated_at``·``rule_source``를 채운 ``PreparedRawEmrContext``(``rule_input``은 비어 있음).
        예외:
            PatientContextUnavailableError: ``patient``가 객체가 아니거나 ``patientId``가 없거나 비었을 때.
            PatientMismatchError: ``patientId``가 ``patient_id``와 다를 때.
        """
        # Bridge는 EMR 객체 자체를 보내거나 평소의 {"data": ...} 응답 봉투를 보낼 수 있다.
        source = emr_source(payload)
        patient = source.get("patient")
        if not isinstance(patient, Mapping):
            raise PatientContextUnavailableError("emr_payload.patient:missing_or_not_object")
        raw_patient_id = patient.get("patientId")
        if not isinstance(raw_patient_id, str) or not raw_patient_id.strip():
            raise PatientContextUnavailableError("emr_payload.patient.patientId:missing_or_blank")
        if raw_patient_id != patient_id:
            raise PatientMismatchError()

        timestamps: list[datetime] = []
        calculations, calculation_input_needed = calculate_review_aids(source.get("calculationContext"))
        fallback = prepared_at or datetime.now(UTC)
        if fallback.tzinfo is None:
            fallback = fallback.replace(tzinfo=UTC)
        data: dict[str, object] = {
            # EMR이 보낸 그대로의 나이·성별·체중·키·크레아티닌(아래 계산 결과는 이 값에서 나온다).
            "calculation_context": deidentify(source.get("calculationContext")),
            "blood_pressure": self._blood_pressure(source.get("bloodPressureList"), timestamps),
            "blood_sugar": self._single_value(source.get("bloodSugarList"), "glucoseValue", "mg/dL", timestamps, extra=("timingType", "timing_type")),
            "oxygen_saturation": self._single_value(source.get("oxygenSaturationList"), "spo2Value", "%", timestamps),
            "lab_results": self._labs(source.get("labResultList"), timestamps),
            "prescriptions": self._prescriptions(source.get("medicineDtailList"), self._reference_lookup),
            "diagnoses": self._diagnoses(source.get("diagnosisList"), self._reference_lookup),
            "calculation_results": [result.as_dict() for result in calculations],
            "calculation_input_needed": list(calculation_input_needed),
            "additional_clinical_data": {
                key: value
                for key, raw_value in source.items()
                if key not in self._HANDLED_SOURCE_FIELDS
                and not is_identifier_key(key)
                and (value := deidentify(raw_value)) is not None
            },
            "recorded_none": [
                _LIST_LABELS.get(key, key) for key, value in source.items()
                if isinstance(value, list) and not value and not is_identifier_key(key)
            ],
        }
        return PreparedRawEmrContext(
            # 빈 그룹은 임상 내용 없이 프롬프트 토큰만 늘린다.
            data={key: value for key, value in data.items() if value},
            updated_at=max(timestamps, default=fallback),
            rule_source=rule_source(source),
        )

    def _blood_pressure(self, items: object, timestamps: list[datetime]) -> list[dict[str, object]]:
        """혈압 목록을 측정 시각·수축기·이완기 행으로 정규화한다.

        역할: 측정 시각이 없거나 읽을 수 없는 항목, ``sbp``·``dbp``가 모두 없는 항목은 버린다(한쪽만 있으면 남긴다).
        남긴 행에 단위 ``mmHg``와 정규화하지 않은 추가 필드를 붙이고, 남긴 행의 측정 시각만 ``timestamps``에 더한다.
        값 없는 행이 갱신 시각을 앞당기지 않게 하려는 것으로, 혈당·산소포화도·검사 결과 목록과 같은 기준이다.
        입력:
            items: EMR의 ``bloodPressureList`` 값(목록이 아니면 빈 결과).
            timestamps: 문맥 갱신 시각 계산용으로 측정 시각을 모으는 목록(제자리에서 추가).
        출력: 정규화된 혈압 행 목록.
        """
        result: list[dict[str, object]] = []
        for item in self._items(items):
            measured_at = self._timestamp(item.get("measuredAt"))
            if measured_at is None:
                continue
            row: dict[str, object] = {"measured_at": measured_at.isoformat(), "unit": "mmHg"}
            for key in ("sbp", "dbp"):
                if (value := deidentify(item.get(key))) is not None:
                    row[key] = value
            if len(row) > 2:
                timestamps.append(measured_at)
                result.append({**row, **_extras(item, ("measuredAt", "sbp", "dbp"))})
        return result

    def _single_value(self, items: object, value_key: str, unit: str, timestamps: list[datetime], *, extra: tuple[str, str] | None = None) -> list[dict[str, object]]:
        """혈당·산소포화도처럼 값 하나짜리 측정 목록을 정규화한다.

        역할: 측정 시각과 값이 모두 있는 항목만 ``measured_at``·``value``·``unit`` 행으로 만들고, ``extra``가 있으면 그
        필드를 지정한 이름으로 옮긴다. 나머지 필드는 식별자를 뺀 뒤 붙이고, 측정 시각을 ``timestamps``에 더한다.
        입력:
            items: EMR 측정 목록 값(목록이 아니면 빈 결과).
            value_key: 측정값 필드 이름(예: ``glucoseValue``).
            unit: 행에 붙일 고정 단위.
            timestamps: 측정 시각을 모으는 목록(제자리에서 추가).
            extra: (원본 필드, 출력 이름) 한 쌍(예: ``("timingType", "timing_type")``). ``None``이면 없음.
        출력: 정규화된 측정 행 목록.
        """
        result: list[dict[str, object]] = []
        for item in self._items(items):
            measured_at = self._timestamp(item.get("measuredAt"))
            value = deidentify(item.get(value_key))
            if measured_at is None or value is None:
                continue
            timestamps.append(measured_at)
            row: dict[str, object] = {"measured_at": measured_at.isoformat(), "value": value, "unit": unit}
            if extra is not None and (extra_value := deidentify(item.get(extra[0]))) is not None:
                row[extra[1]] = extra_value
            result.append({**row, **_extras(item, ("measuredAt", value_key, *(extra[:1] if extra else ())))})
        return result

    def _labs(self, items: object, timestamps: list[datetime]) -> list[dict[str, object]]:
        """검사 결과 목록을 검사일·값·코드·항목명·단위 행으로 정규화한다.

        역할: 검사일이 없거나 읽을 수 없는 항목, 값이 없거나 숫자 또는 200자 이하의 비어 있지 않은 문자열이 아닌 항목은
        버린다. 비어 있지 않은 ``examCode``·``itemName``·``unit``을 옮기고 나머지 필드는 식별자를 뺀 뒤 붙인다.
        입력:
            items: EMR의 ``labResultList`` 값(목록이 아니면 빈 결과).
            timestamps: 검사 시각을 모으는 목록(제자리에서 추가).
        출력: 정규화된 검사 행 목록.
        """
        result: list[dict[str, object]] = []
        for item in self._items(items):
            tested_at = self._timestamp(item.get("testDate"))
            value = item.get("value")
            if tested_at is None or value is None or not self._is_short_value(value):
                continue
            timestamps.append(tested_at)
            row: dict[str, object] = {"test_date": tested_at.isoformat(), "value": value}
            self._copy_non_blank(item, row, "examCode", "exam_code")
            self._copy_non_blank(item, row, "itemName", "item_name")
            self._copy_non_blank(item, row, "unit", "unit")
            result.append({**row, **_extras(item, ("testDate", "value", "examCode", "itemName", "unit"))})
        return result

    @staticmethod
    def _prescriptions(
        items: object, reference_lookup: ReferenceLookup | None
    ) -> list[dict[str, object]]:
        """처방 목록을 허용 필드 행으로 정규화하고 참조 DB로 보강한다.

        역할: 허용 필드(``medCode``·``medName``·``dosage`` 등)를 출력 이름으로 옮기고(중첩 값도 ``deidentify`` 통과),
        ``medCode``에 정확히 대응하는 참조 약품이 있으면 없는 ``med_name``을 채우고 효능·주의사항을 600자까지 붙인다.
        나머지 필드는 식별자를 뺀 뒤 붙이고, 빈 행은 버린다.
        입력:
            items: EMR의 ``medicineDtailList`` 값(목록이 아니면 빈 결과).
            reference_lookup: 참조 DB 조회기. ``None``이면 보강하지 않는다.
        출력: 처방 행 목록.
        """
        allowed = (("medCode", "med_code"), ("medName", "med_name"), ("category", "category"), ("dosage", "dosage"), ("unit", "unit"), ("frequency", "frequency"), ("interval", "interval"), ("durationDays", "duration_days"))
        result: list[dict[str, object]] = []
        for item in RawEmrPayloadMapper._items(items):
            # 허용된 값도 중첩 객체일 수 있으므로 알 수 없는 필드와 같은 식별자 경계를 통과시킨다.
            row = {target: value for source, target in allowed if (value := deidentify(item.get(source))) is not None}
            reference = None if reference_lookup is None else reference_lookup.medicine(item.get("medCode"))
            if reference is not None:
                row.setdefault("med_name", reference.name_ko)
                if reference.efficacy is not None:
                    row["reference_efficacy"] = reference.efficacy[:600]
                if reference.precautions is not None:
                    row["reference_precautions"] = reference.precautions[:600]
            # 알려진 필드 밖의 필드(복용 지시, 투여 경로, 시작일 등)도 같은 식별자 경계를 통과한다.
            row.update(_extras(item, tuple(source for source, _target in allowed)))
            if row:
                result.append(row)
        return result

    @staticmethod
    def _diagnoses(
        items: object, reference_lookup: ReferenceLookup | None
    ) -> list[dict[str, object]]:
        """진단 목록을 상병 코드와 한글명 행으로 정규화한다.

        역할: ``diagnosisCode``(키가 없으면 ``kcdCode``)가 비어 있지 않은 문자열인 항목만 남긴다. 한글명은 참조 DB 조회
        결과를 우선하고, 없으면 EMR의 ``kcdName``을 쓴다. 나머지 필드는 식별자를 뺀 뒤 붙인다.
        입력:
            items: EMR의 ``diagnosisList`` 값(목록이 아니면 빈 결과).
            reference_lookup: 참조 DB 조회기. ``None``이면 EMR의 이름만 쓴다.
        출력: ``code``(앞뒤 공백 제거)와 있으면 ``name_ko``를 담은 진단 행 목록.
        """
        result: list[dict[str, object]] = []
        for item in RawEmrPayloadMapper._items(items):
            code = item.get("diagnosisCode", item.get("kcdCode"))
            if not isinstance(code, str) or not code.strip():
                continue
            row: dict[str, object] = {"code": code.strip()}
            name = None if reference_lookup is None else reference_lookup.disease(code)
            if name is not None:
                row["name_ko"] = name
            elif isinstance(item.get("kcdName"), str) and item["kcdName"].strip():
                row["name_ko"] = item["kcdName"].strip()
            result.append({**row, **_extras(item, ("diagnosisCode", "kcdCode", "kcdName"))})
        return result

    @staticmethod
    def _items(value: object) -> list[Mapping[str, Any]]:
        return [item for item in value if isinstance(item, Mapping)] if isinstance(value, list) else []

    def _timestamp(self, value: object) -> datetime | None:
        """ISO 8601 문자열을 시각으로 읽는다. 시간대가 없으면 서울 시간으로 보고, 읽을 수 없으면 ``None``."""
        if not isinstance(value, str) or not value.strip():
            return None
        try:
            parsed = datetime.fromisoformat(value.strip())
        except ValueError:
            return None
        return parsed.replace(tzinfo=self._SEOUL) if parsed.tzinfo is None else parsed

    @staticmethod
    def _is_short_value(value: object) -> bool:
        return (isinstance(value, (int, float)) and not isinstance(value, bool)) or (isinstance(value, str) and bool(value.strip()) and len(value) <= 200)

    @staticmethod
    def _copy_non_blank(source: Mapping[str, Any], target: dict[str, object], key: str, target_key: str) -> None:
        value = source.get(key)
        if isinstance(value, str) and value.strip():
            target[target_key] = value


def _extras(item: Mapping[str, Any], handled: tuple[str, ...]) -> dict[str, object]:
    """매퍼가 정규화하지 않는 알려진 목록 항목의 필드를, 식별자를 뺀 채 EMR 원래 이름으로 남긴다."""
    return {
        key: value for key, raw in item.items()
        if isinstance(key, str) and key not in handled and not is_identifier_key(key) and (value := deidentify(raw)) is not None
    }
