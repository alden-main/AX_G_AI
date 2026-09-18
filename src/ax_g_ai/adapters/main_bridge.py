"""메인 서버 브릿지의 생체정보 응답을 내부 EMR 계약으로 변환한다."""

from __future__ import annotations

from typing import Any, Mapping

from ax_g_ai.adapters.emr import EmrAdapterError


class MainBridgePayloadMapper:
    """광주AX API 명세 v0.1의 camelCase 생체정보 응답을 정규 입력으로 바꾼다.

    혈압·혈당·SpO2·HbA1c는 원천 ID·단위·결과 상태가 명세에 없으므로, 값을 추정하지
    않고 ``PatientContextBuilder``가 임상 사용 불가로 표시할 필드를 빈 상태로 남긴다.
    ``encounterId``는 명세에 없지만 D-03의 필수 계약이므로 없으면 명시적으로 실패한다.
    """

    def __init__(self, organization_id: str) -> None:
        self._organization_id = organization_id

    def to_patient_context_payload(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        """브릿지 JSON 객체를 D-02 입력 객체로 변환한다.

        Raises:
            EmrAdapterError: 환자·의료진·에피소드 식별자가 없거나 형식이 다를 때
                발생한다. 이 경우 챗봇은 해당 환자 데이터를 사용하면 안 된다.
        """
        patient = payload.get("patient")
        if not isinstance(patient, Mapping):
            raise EmrAdapterError("bridge response has no patient object")
        patient_id = self._string(patient, "patientId")
        encounter_id = self._string(patient, "encounterId")
        admission = self._string(patient, "admissionDate")
        discharge = self._string(patient, "dischargeDate")
        observations = []
        observations.extend(self._blood_pressure(payload.get("bloodPressureList", [])))
        observations.extend(self._single_value(payload.get("bloodSugarList", []), "glucoseValue", "GLUCOSE"))
        observations.extend(self._single_value(payload.get("oxygenSaturationList", []), "spo2Value", "SPO2"))
        observations.extend(self._single_value(payload.get("glycatedHemoglobinList", []), "hba1cValue", "HBA1C", "testDate"))
        return {
            "actor": {"user_id": self._string(patient, "doctorId"), "organization_id": self._organization_id, "role": "physician", "purpose": "treatment"},
            "patient": {"patient_id": patient_id},
            "encounter": {"encounter_id": encounter_id, "type": "unknown", "status": "unknown"},
            "range": {"from": f"{admission}T00:00:00+09:00", "to": f"{discharge}T23:59:59+09:00", "timezone": "Asia/Seoul"},
            "observations": observations,
            "diagnoses": [], "prescriptions": list(payload.get("medicineDtailList", [])), "lab_results": [],
            "missing_data": [], "delayed_data": [],
        }

    @staticmethod
    def _string(data: Mapping[str, Any], key: str) -> str:
        value = data.get(key)
        if not isinstance(value, str) or not value.strip():
            raise EmrAdapterError(f"bridge response requires patient.{key}")
        return value

    @staticmethod
    def _single_value(items: Any, value_key: str, code: str, time_key: str = "measuredAt") -> list[dict[str, Any]]:
        if not isinstance(items, list):
            raise EmrAdapterError("bridge observation list must be an array")
        return [{"source_record_id": None, "code_system": "main-bridge-v0.1", "code": code, "value": item.get(value_key), "unit": None, "observed_at": MainBridgePayloadMapper._seoul_time(item.get(time_key)), "source": None, "status": None} for item in items if isinstance(item, Mapping)]

    @staticmethod
    def _blood_pressure(items: Any) -> list[dict[str, Any]]:
        if not isinstance(items, list):
            raise EmrAdapterError("bridge bloodPressureList must be an array")
        result = []
        for item in items:
            if isinstance(item, Mapping):
                for key, code in (("sbp", "SYSTOLIC_BP"), ("dbp", "DIASTOLIC_BP")):
                    result.append({"source_record_id": None, "code_system": "main-bridge-v0.1", "code": code, "value": item.get(key), "unit": None, "observed_at": MainBridgePayloadMapper._seoul_time(item.get("measuredAt")), "source": None, "status": None})
        return result

    @staticmethod
    def _seoul_time(value: Any) -> Any:
        return f"{value.replace(' ', 'T')}+09:00" if isinstance(value, str) and "T" not in value else value
