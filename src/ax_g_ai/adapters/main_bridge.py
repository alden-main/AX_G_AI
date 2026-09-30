"""Bridge가 전달한 원문 EMR의 PoC 전용 최소 문맥 변환기."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping
from zoneinfo import ZoneInfo

from ax_g_ai.adapters.emr import EmrAdapterError


@dataclass(frozen=True)
class PreparedRawEmrContext:
    """원문에서 허용 목록만 추린, Provider 전송 전의 메모리 문맥이다."""

    data: Mapping[str, object]
    updated_at: datetime


class RawEmrPayloadMapper:
    """샘플 원문의 알려진 목록만 PoC 문맥으로 변환한다."""

    _SEOUL = ZoneInfo("Asia/Seoul")

    def map(self, patient_id: str, payload: Mapping[str, Any], *, prepared_at: datetime | None = None) -> PreparedRawEmrContext:
        patient = payload.get("patient")
        if not isinstance(patient, Mapping):
            raise EmrAdapterError("raw EMR payload has no patient object")
        raw_patient_id = patient.get("patientId")
        if not isinstance(raw_patient_id, str) or not raw_patient_id.strip():
            raise EmrAdapterError("raw EMR payload requires patient.patientId")
        if raw_patient_id != patient_id:
            raise EmrAdapterError("raw EMR patient does not match the request")

        timestamps: list[datetime] = []
        blood_pressure = self._blood_pressure(payload.get("bloodPressureList"), timestamps)
        blood_sugar = self._single_value(payload.get("bloodSugarList"), "glucoseValue", "mg/dL", timestamps, extra=("timingType", "timing_type"))
        oxygen_saturation = self._single_value(payload.get("oxygenSaturationList"), "spo2Value", "%", timestamps)
        lab_results = self._labs(payload.get("labResultList"), timestamps)
        prescriptions = self._prescriptions(payload.get("medicineDtailList"))
        fallback = prepared_at or datetime.now(timezone.utc)
        if fallback.tzinfo is None:
            fallback = fallback.replace(tzinfo=timezone.utc)
        return PreparedRawEmrContext(
            data={"blood_pressure": blood_pressure, "blood_sugar": blood_sugar, "oxygen_saturation": oxygen_saturation, "lab_results": lab_results, "prescriptions": prescriptions},
            updated_at=max(timestamps, default=fallback),
        )

    def _blood_pressure(self, items: object, timestamps: list[datetime]) -> list[dict[str, object]]:
        result: list[dict[str, object]] = []
        for item in self._items(items):
            measured_at = self._timestamp(item.get("measuredAt"))
            sbp, dbp = item.get("sbp"), item.get("dbp")
            if measured_at is None or sbp is None or dbp is None:
                continue
            timestamps.append(measured_at)
            result.append({"measured_at": measured_at.isoformat(), "sbp": sbp, "dbp": dbp, "unit": "mmHg"})
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
    def _prescriptions(items: object) -> list[dict[str, object]]:
        allowed = (("medCode", "med_code"), ("medName", "med_name"), ("category", "category"), ("dosage", "dosage"), ("unit", "unit"), ("frequency", "frequency"), ("interval", "interval"), ("durationDays", "duration_days"))
        result: list[dict[str, object]] = []
        for item in RawEmrPayloadMapper._items(items):
            row = {target: item[source] for source, target in allowed if item.get(source) is not None}
            if row:
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
