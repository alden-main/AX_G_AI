"""Bridge가 전달한 원문 EMR의 PoC 전용 최소 문맥 변환기.

이 모듈이 Provider로 가는 환자 문맥의 유일한 비식별 경계다. 알려진 목록은 아는 필드를
정규화하고, 그 밖의 필드(알려진 목록 항목 안의 새 필드 포함)는 직접 식별자를 제거한 뒤 원래 이름으로 보존한다.
빈 목록으로 온 항목은 ``recorded_none``에 남겨 '기록 없음'과 '전달되지 않음'을 구분한다.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping
from zoneinfo import ZoneInfo

from ax_g_ai.errors import PatientContextUnavailableError, PatientMismatchError
from ax_g_ai.services.clinical_calculations import calculate_review_aids
from ax_g_ai.services.deidentified_snapshot import deidentify, is_identifier_key
from ax_g_ai.services.patient_context_store import PreparedRawEmrContext
from ax_g_ai.services.reference_lookup import SqliteReferenceLookup


# EMR fields the rule mapping reads (``emr_mapping.map_emr``, ``ClinicalRecommender.prepare``). A whitelist, so
# no identifier passes; unlike ``deidentify`` it keeps empty lists and null values, which the rules read as
# "recorded: none" versus "not recorded".
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
# Prompt names of the lists an EMR may send empty (``recorded_none``: recorded as none, not "not sent").
_LIST_LABELS = {
    "bloodPressureList": "혈압", "bloodSugarList": "혈당", "oxygenSaturationList": "산소포화도", "labResultList": "검사 결과",
    "glycatedHemoglobinList": "당화혈색소", "medicineDtailList": "처방", "diagnosisList": "진단",
}


def rule_source(source: Mapping[str, Any]) -> dict[str, Any]:
    """The part of the EMR object the rule mapping reads, as JSON-ready data."""
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
    """The EMR object itself, whether the Bridge sent it bare or in its ``{"data": ...}`` envelope."""
    raw_data = payload.get("data")
    return raw_data if isinstance(raw_data, Mapping) else payload


class RawEmrPayloadMapper:
    """원문에서 비식별 임상 문맥을 만들어 Provider 전송 전 메모리에 보관한다."""

    _SEOUL = ZoneInfo("Asia/Seoul")
    _HANDLED_SOURCE_FIELDS = frozenset({
        "patient", "bloodPressureList", "bloodSugarList", "oxygenSaturationList",
        "labResultList", "medicineDtailList", "diagnosisList", "calculationContext",
    })

    def __init__(self, reference_lookup: SqliteReferenceLookup | None = None) -> None:
        self._reference_lookup = reference_lookup

    def map(self, patient_id: str, payload: Mapping[str, Any], *, prepared_at: datetime | None = None) -> PreparedRawEmrContext:
        # The Bridge may pass the EMR object itself or its usual {"data": ...}
        # response envelope.
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
        fallback = prepared_at or datetime.now(timezone.utc)
        if fallback.tzinfo is None:
            fallback = fallback.replace(tzinfo=timezone.utc)
        data: dict[str, object] = {
            # Age, sex, weight, height, creatinine as the EMR sent them (the calculations below are derived from them).
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
            # Empty groups add prompt tokens without clinical content.
            data={key: value for key, value in data.items() if value},
            updated_at=max(timestamps, default=fallback),
            rule_source=rule_source(source),
        )

    def _blood_pressure(self, items: object, timestamps: list[datetime]) -> list[dict[str, object]]:
        result: list[dict[str, object]] = []
        for item in self._items(items):
            measured_at = self._timestamp(item.get("measuredAt"))
            if measured_at is None:
                continue
            timestamps.append(measured_at)
            row: dict[str, object] = {"measured_at": measured_at.isoformat(), "unit": "mmHg"}
            for key in ("sbp", "dbp"):
                if (value := deidentify(item.get(key))) is not None:
                    row[key] = value
            if len(row) > 2:
                result.append({**row, **_extras(item, ("measuredAt", "sbp", "dbp"))})
        return result

    def _single_value(self, items: object, value_key: str, unit: str, timestamps: list[datetime], *, extra: tuple[str, str] | None = None) -> list[dict[str, object]]:
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
        items: object, reference_lookup: SqliteReferenceLookup | None
    ) -> list[dict[str, object]]:
        allowed = (("medCode", "med_code"), ("medName", "med_name"), ("category", "category"), ("dosage", "dosage"), ("unit", "unit"), ("frequency", "frequency"), ("interval", "interval"), ("durationDays", "duration_days"))
        result: list[dict[str, object]] = []
        for item in RawEmrPayloadMapper._items(items):
            # Allowed values may still be nested objects, so they pass the same
            # identifier boundary as unknown fields.
            row = {target: value for source, target in allowed if (value := deidentify(item.get(source))) is not None}
            reference = None if reference_lookup is None else reference_lookup.medicine(item.get("medCode"))
            if reference is not None:
                row.setdefault("med_name", reference.name_ko)
                if reference.efficacy is not None:
                    row["reference_efficacy"] = reference.efficacy[:600]
                if reference.precautions is not None:
                    row["reference_precautions"] = reference.precautions[:600]
            # Fields beyond the known ones (instructions, route, start date, …) pass the same identifier boundary.
            row.update(_extras(item, tuple(source for source, _target in allowed)))
            if row:
                result.append(row)
        return result

    @staticmethod
    def _diagnoses(
        items: object, reference_lookup: SqliteReferenceLookup | None
    ) -> list[dict[str, object]]:
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
    """Fields of a known list item the mapper does not normalize, kept under their EMR names without identifiers."""
    return {
        key: value for key, raw in item.items()
        if isinstance(key, str) and key not in handled and not is_identifier_key(key) and (value := deidentify(raw)) is not None
    }
