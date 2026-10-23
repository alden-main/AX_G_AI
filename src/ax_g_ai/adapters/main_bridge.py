"""Bridge가 전달한 원문 EMR의 PoC 전용 최소 문맥 변환기."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping
from zoneinfo import ZoneInfo

from ax_g_ai.adapters.emr import EmrAdapterError
from ax_g_ai.services.clinical_calculations import calculate_review_aids
from ax_g_ai.services.reference_lookup import SqliteReferenceLookup


@dataclass(frozen=True)
class PreparedRawEmrContext:
    """원문에서 허용 목록만 추린, Provider 전송 전의 메모리 문맥이다."""

    data: Mapping[str, object]
    updated_at: datetime


class RawEmrPayloadMapper:
    """원문에서 비식별 임상 문맥을 만들어 Provider 전송 전 메모리에 보관한다."""

    _SEOUL = ZoneInfo("Asia/Seoul")
    _HANDLED_SOURCE_FIELDS = frozenset({
        "patient", "bloodPressureList", "bloodSugarList", "oxygenSaturationList",
        "labResultList", "medicineDtailList", "diagnosisList", "calculationContext",
    })
    _IDENTIFIER_KEYS = frozenset({
        "patient", "patientid", "name", "firstname", "lastname", "fullname", "birthdate",
        "dateofbirth", "phone", "mobile", "email", "address", "zipcode", "ssn", "rrn",
        "nationalid", "medicalrecordnumber", "mrn", "encounterid", "visitid", "recordid",
        "documentid", "providerid", "doctorid", "userid",
    })

    def __init__(self, reference_lookup: SqliteReferenceLookup | None = None) -> None:
        self._reference_lookup = reference_lookup

    def map(self, patient_id: str, payload: Mapping[str, Any], *, prepared_at: datetime | None = None) -> PreparedRawEmrContext:
        # The Bridge may pass the EMR object itself or its usual {"data": ...}
        # response envelope. Known fields are normalized below and newly added
        # clinical fields are retained only after direct identifiers are removed.
        raw_data = payload.get("data")
        source = raw_data if isinstance(raw_data, Mapping) else payload
        patient = source.get("patient")
        if not isinstance(patient, Mapping):
            raise EmrAdapterError("raw EMR payload has no patient object")
        raw_patient_id = patient.get("patientId")
        if not isinstance(raw_patient_id, str) or not raw_patient_id.strip():
            raise EmrAdapterError("raw EMR payload requires patient.patientId")
        if raw_patient_id != patient_id:
            raise EmrAdapterError("raw EMR patient does not match the request")

        timestamps: list[datetime] = []
        blood_pressure = self._blood_pressure(source.get("bloodPressureList"), timestamps)
        blood_sugar = self._single_value(source.get("bloodSugarList"), "glucoseValue", "mg/dL", timestamps, extra=("timingType", "timing_type"))
        oxygen_saturation = self._single_value(source.get("oxygenSaturationList"), "spo2Value", "%", timestamps)
        lab_results = self._labs(source.get("labResultList"), timestamps)
        prescriptions = self._prescriptions(source.get("medicineDtailList"), self._reference_lookup)
        diagnoses = self._diagnoses(source.get("diagnosisList"), self._reference_lookup)
        calculations, calculation_input_needed = calculate_review_aids(source.get("calculationContext"))
        fallback = prepared_at or datetime.now(timezone.utc)
        if fallback.tzinfo is None:
            fallback = fallback.replace(tzinfo=timezone.utc)
        data: dict[str, object] = {
            "blood_pressure": blood_pressure,
            "blood_sugar": blood_sugar,
            "oxygen_saturation": oxygen_saturation,
            "lab_results": lab_results,
            "prescriptions": prescriptions,
            "diagnoses": diagnoses,
            "calculation_results": [result.as_dict() for result in calculations],
            "calculation_input_needed": list(calculation_input_needed),
        }
        additional = self._additional_clinical_data(source)
        if additional:
            data["additional_clinical_data"] = additional
        return PreparedRawEmrContext(
            data=data,
            updated_at=max(timestamps, default=fallback),
        )

    @classmethod
    def _additional_clinical_data(cls, source: Mapping[str, Any]) -> dict[str, object]:
        """Keep newly supplied clinical fields while excluding direct identifiers."""
        return {
            key: value
            for key, raw_value in source.items()
            if key not in cls._HANDLED_SOURCE_FIELDS
            and not cls._is_identifier_key(key)
            and (value := cls._deidentify_value(raw_value)) is not None
        }

    @classmethod
    def _deidentify_value(cls, value: object) -> object | None:
        if value is None:
            return None
        if isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, Mapping):
            result = {
                key: safe_value
                for key, raw_value in value.items()
                if isinstance(key, str)
                and not cls._is_identifier_key(key)
                and (safe_value := cls._deidentify_value(raw_value)) is not None
            }
            return result or None
        if isinstance(value, list):
            result = [safe_value for item in value if (safe_value := cls._deidentify_value(item)) is not None]
            return result or None
        return None

    @classmethod
    def _is_identifier_key(cls, key: str) -> bool:
        normalized = "".join(character for character in key.casefold() if character.isalnum())
        return normalized in cls._IDENTIFIER_KEYS or normalized.endswith("identifier")

    def _blood_pressure(self, items: object, timestamps: list[datetime]) -> list[dict[str, object]]:
        result: list[dict[str, object]] = []
        for item in self._items(items):
            measured_at = self._timestamp(item.get("measuredAt"))
            if measured_at is None:
                continue
            timestamps.append(measured_at)
            row: dict[str, object] = {"measured_at": measured_at.isoformat(), "unit": "mmHg"}
            if item.get("sbp") is not None:
                row["sbp"] = item["sbp"]
            if item.get("dbp") is not None:
                row["dbp"] = item["dbp"]
            if len(row) > 2:
                result.append(row)
        return result

    def _single_value(self, items: object, value_key: str, unit: str, timestamps: list[datetime], *, extra: tuple[str, str] | None = None) -> list[dict[str, object]]:
        result: list[dict[str, object]] = []
        for item in self._items(items):
            measured_at = self._timestamp(item.get("measuredAt"))
            value = item.get(value_key)
            if measured_at is None or value is None:
                continue
            timestamps.append(measured_at)
            row: dict[str, object] = {"measured_at": measured_at.isoformat(), "value": value, "unit": unit}
            if extra is not None and item.get(extra[0]) is not None:
                row[extra[1]] = item[extra[0]]
            result.append(row)
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
            result.append(row)
        return result

    @staticmethod
    def _prescriptions(
        items: object, reference_lookup: SqliteReferenceLookup | None
    ) -> list[dict[str, object]]:
        allowed = (("medCode", "med_code"), ("medName", "med_name"), ("category", "category"), ("dosage", "dosage"), ("unit", "unit"), ("frequency", "frequency"), ("interval", "interval"), ("durationDays", "duration_days"))
        result: list[dict[str, object]] = []
        for item in RawEmrPayloadMapper._items(items):
            row = {target: item[source] for source, target in allowed if item.get(source) is not None}
            reference = None if reference_lookup is None else reference_lookup.medicine(item.get("medCode"))
            if reference is not None:
                row.setdefault("med_name", reference.name_ko)
                if reference.efficacy is not None:
                    row["reference_efficacy"] = reference.efficacy[:600]
                if reference.precautions is not None:
                    row["reference_precautions"] = reference.precautions[:600]
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
            result.append(row)
        return result

    @staticmethod
    def _items(value: object) -> list[Mapping[str, Any]]:
        return [item for item in value if isinstance(item, Mapping)] if isinstance(value, list) else []

    def _timestamp(self, value: object) -> datetime | None:
        if not isinstance(value, str) or not value.strip():
            return None
        try:
            parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00").replace(" ", "T"))
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


# Kept as an import-compatible name while the Bridge migrates to the raw PoC contract.
MainBridgePayloadMapper = RawEmrPayloadMapper
