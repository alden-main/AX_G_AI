"""EMR 읽기 연동과 내부 환자 문맥 사이의 Adapter 경계.

P-01의 endpoint·인증·Schema 합의 전에는 구체 HTTP/event client를 구현하지 않는다.
이 모듈은 제공사 세부사항이 D-02 도메인 코드로 전파되지 않게 하고, 응답 환자·진료
에피소드·조회 기간이 요청과 다를 때 이전 또는 타 환자 데이터를 반환하지 않게 한다.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from typing import Any, Mapping, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

from ax_g_ai.domain.patient_context import PatientContext, PatientContextBuilder


class EmrAdapterError(RuntimeError):
    """EMR 조회 실패 시 발생하며, 호출자는 이전 환자 데이터로 대체하면 안 된다."""


@dataclass(frozen=True)
class EmrReadRequest:
    """D-01이 제공사 Client에 전달하는 읽기 요청의 내부 계약.

    Attributes:
        patient_id: 조회할 EMR 환자 식별자다.
        encounter_id: 조회할 진료 에피소드 식별자다.
        from_at/to_at: 시간대가 포함된 조회 시작·종료 ``datetime``이다.
        timezone: 제공사 응답과 비교할 IANA 시간대 문자열이다.
        request_id: 제공사 요청 추적과 감사 연계를 위한 호출 식별자다.
    """

    patient_id: str
    encounter_id: str
    from_at: datetime
    to_at: datetime
    timezone: str
    request_id: str
    context_id: str | None = None


@dataclass(frozen=True)
class PatientSnapshot:
    """Bridge Snapshot의 대조된 환자 문맥과 최소 표시 메타데이터다.

    ``context_id``가 있는 요청에서만 생성된다. 기존 EMR provider 계약은 이
    확장 메타데이터를 제공하지 않을 수 있으므로 ``read_patient_context``의 기존
    반환 계약은 유지한다.
    """

    context: PatientContext
    context_id: str
    snapshot_id: str
    snapshot_at: datetime
    schema_version: str
    assembly_status: str
    missing_data: tuple[str, ...]
    delayed_data: tuple[str, ...]


@dataclass(frozen=True)
class PreparedPatientSnapshot:
    """간소화된 Bridge 계약에서 검증한 원본 ``patient-context-v1`` Snapshot."""

    patient_id: str
    snapshot_at: datetime
    data: Mapping[str, Any]


class EmrReadClient(Protocol):
    """P-01 확정 후 제공사별 구현체가 따라야 하는 읽기 인터페이스."""

    def fetch_patient_snapshot(self, request: EmrReadRequest) -> Mapping[str, Any]: ...


class UrllibEmrReadClient:
    """환경 설정으로 지정한 EMR HTTP API를 D-01 내부 계약으로 연결한다.

    제공사 API는 ``EmrReadRequest``의 환자·에피소드·기간·시간대를 POST JSON 본문 또는
    GET query로 받는다. 응답은 설계된 EMR payload 객체여야 하며, 그 정규화와 환자 일치
    검증은 ``EmrAdapter``가 수행한다. 이 Client는 성공 응답을 캐시하거나 이전 환자
    데이터로 대체하지 않는다.
    """

    def __init__(self, settings: object) -> None:
        # Legacy read-client boundary retained for the older, separate EMR flow.
        self._settings = settings

    def fetch_patient_snapshot(self, request: EmrReadRequest) -> Mapping[str, Any]:
        """EMR에 읽기 요청을 보내고 JSON 객체 응답만 반환한다.

        Args:
            request: 환자·에피소드·기간·시간대·추적 ID를 담은 ``EmrReadRequest``다.

        Returns:
            EMR이 제공한 ``Mapping[str, Any]`` 형식의 원천 payload다.

        Raises:
            EmrAdapterError: 네트워크·HTTP·Content-Type·JSON 형식 실패가 발생하면
                안전한 오류로 변환한다. 토큰이나 제공사 오류 원문은 포함하지 않는다.
        """
        request_data = self._request_data(request) if self._settings.send_request_context else {}
        url = self._url_for(request, request_data)
        headers = {"Accept": "application/json", "X-Request-ID": request.request_id}
        if self._settings.access_token:
            headers["Authorization"] = f"Bearer {self._settings.access_token}"
        body = None
        if self._settings.method == "POST":
            headers["Content-Type"] = "application/json"
            body = json.dumps(request_data).encode("utf-8")
        http_request = Request(url, data=body, headers=headers, method=self._settings.method)
        try:
            with urlopen(http_request, timeout=self._settings.timeout_seconds) as response:
                content_type = response.headers.get("Content-Type", "")
                raw_body = response.read()
        except (HTTPError, URLError, TimeoutError, OSError) as error:
            raise EmrAdapterError("EMR HTTP read failed") from error
        if not content_type.lower().startswith("application/json"):
            raise EmrAdapterError("EMR returned an unexpected content type")
        try:
            payload = json.loads(raw_body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise EmrAdapterError("EMR returned invalid JSON") from error
        if not isinstance(payload, Mapping):
            raise EmrAdapterError("EMR response must be a JSON object")
        if request.context_id:
            if payload.get("success") is not True or not isinstance(payload.get("data"), Mapping):
                raise EmrAdapterError("EMR bridge returned no usable patient data")
            return payload["data"]
        if "success" in payload:
            if payload.get("success") is not True or not isinstance(payload.get("data"), Mapping):
                raise EmrAdapterError("EMR bridge returned no usable patient data")
            return payload["data"]
        return payload

    def _url_for(self, request: EmrReadRequest, request_data: Mapping[str, str]) -> str:
        try:
            path = self._settings.snapshot_path.format(
                patient_id=quote(request.patient_id, safe=""),
                encounter_id=quote(request.encounter_id, safe=""),
            )
        except KeyError as error:
            raise EmrAdapterError("EMR context path contains an unsupported placeholder") from error
        url = f"{self._settings.bridge_base_url}/{path.lstrip('/')}"
        return f"{url}?{urlencode(request_data)}" if self._settings.method == "GET" else url

    @staticmethod
    def _request_data(request: EmrReadRequest) -> dict[str, str]:
        # The current Bridge Snapshot API accepts only patient_id.  The legacy
        # request shape remains below for the older provider adapter boundary.
        if not request.encounter_id:
            return {"patient_id": request.patient_id}
        data = {
            "patient_id": request.patient_id,
            "encounter_id": request.encounter_id,
            "from": request.from_at.isoformat(),
            "to": request.to_at.isoformat(),
            "timezone": request.timezone,
        }
        if request.context_id:
            data["context_id"] = request.context_id
        return data


class EmrAdapter:
    """D-01의 EMR 응답 검증과 D-02 환자 문맥 생성 연결을 담당한다.

    ``EmrReadClient``는 제공사 endpoint·인증·재시도 정책을 캡슐화하고,
    ``PatientContextBuilder``는 응답을 내부 DTO로 정규화한다. 이 Adapter는 I/O
    실패와 payload 검증 실패를 ``EmrAdapterError``로 구분 가능한 안전 오류로
    바꾸며, 요청과 다른 환자·에피소드·기간·시간대의 응답은 반드시 폐기한다.
    """

    def __init__(self, client: EmrReadClient, context_builder: PatientContextBuilder) -> None:
        self._client = client
        self._context_builder = context_builder

    def read_patient_context(self, request: EmrReadRequest) -> PatientContext:
        """EMR 스냅샷을 읽고 요청과 일치하는 정규 환자 문맥만 반환한다.

        Args:
            request: 환자 ID, 에피소드 ID, 시간대 포함 조회 기간, 추적 ID를 가진
                ``EmrReadRequest``다. D-03의 권한 검증 뒤에만 호출해야 한다.

        Returns:
            요청의 환자·에피소드·기간·시간대와 모두 일치하고 D-02 품질 판정을
            통과한 ``PatientContext``다. 관찰값의 품질 문제는 문맥에 보존된다.

        Raises:
            EmrAdapterError: provider I/O가 실패하거나, 응답 형식이 잘못되었거나,
                응답 문맥이 요청과 일치하지 않을 때 발생한다. 호출자는 과거 캐시나
                다른 환자의 문맥을 fallback으로 반환하면 안 된다.
        """
        return self._read(request)[0]

    def read_bridge_snapshot(self, request: EmrReadRequest) -> PatientSnapshot:
        """`context_id`가 있는 Bridge Snapshot만 반환한다.

        요청과 응답의 ``context_id``, 환자, 에피소드, 기간·시간대를 모두 대조한다.
        대조할 수 없는 응답은 늦게 도착한 이전 환자 Snapshot일 수 있으므로 폐기한다.
        """
        if not request.context_id:
            raise EmrAdapterError("Bridge snapshot request requires a context_id")
        context, payload = self._read(request)
        context_id = self._required_string(payload, "context_id")
        if context_id != request.context_id:
            raise EmrAdapterError("EMR response context does not match the request")
        snapshot_id = self._required_string(payload, "snapshot_id")
        snapshot_at = self._datetime(payload, "snapshot_at")
        schema_version = self._required_string(payload, "schema_version")
        if schema_version != "patient-context-v1":
            raise EmrAdapterError("EMR response schema version is invalid")
        assembly_status = self._required_string(payload, "assembly_status")
        if assembly_status not in {"complete", "partial"}:
            raise EmrAdapterError("EMR response assembly status is invalid")
        missing_data = payload.get("missing_data", [])
        if not isinstance(missing_data, list) or not all(
            isinstance(item, str) and item.strip() for item in missing_data
        ):
            raise EmrAdapterError("EMR response missing_data is invalid")
        delayed_data = payload.get("delayed_data")
        if not isinstance(delayed_data, list) or not all(
            isinstance(item, str) and item.strip() for item in delayed_data
        ):
            raise EmrAdapterError("EMR response delayed_data is invalid")
        return PatientSnapshot(
            context, context_id, snapshot_id, snapshot_at, schema_version, assembly_status,
            tuple(missing_data), tuple(delayed_data),
        )

    def read_prepared_snapshot(self, request: EmrReadRequest) -> PreparedPatientSnapshot:
        """환자 ID만으로 조회한 Snapshot의 필수 wrapper 내부 구조를 검증한다."""
        try:
            payload = self._client.fetch_patient_snapshot(request)
        except EmrAdapterError:
            raise
        except Exception as error:
            raise EmrAdapterError("EMR patient-data read failed") from error
        if self._required_string(payload, "schema_version") != "patient-context-v1":
            raise EmrAdapterError("EMR response schema version is invalid")
        snapshot_at = self._datetime(payload, "snapshot_at")
        if self._required_string(payload, "assembly_status") not in {"complete", "partial"}:
            raise EmrAdapterError("EMR response assembly status is invalid")
        patient = payload.get("patient")
        if not isinstance(patient, Mapping):
            raise EmrAdapterError("EMR response patient is required")
        patient_id = self._required_string(patient, "patient_id")
        if patient_id != request.patient_id:
            raise EmrAdapterError("EMR response patient does not match the request")
        for key in ("observations", "diagnoses", "prescriptions", "lab_results", "missing_data", "delayed_data"):
            if not isinstance(payload.get(key), list):
                raise EmrAdapterError(f"EMR response {key} must be an array")
        return PreparedPatientSnapshot(patient_id, snapshot_at, payload)

    @staticmethod
    def validate_prepared_snapshot(
        patient_id: str, emr_snapshot: Mapping[str, Any]
    ) -> PreparedPatientSnapshot:
        """Bridge가 전달한 ``success`` wrapper와 Snapshot을 검증한다.

        이 경계는 외부 I/O를 하지 않는다. 호출자는 검증 실패를 이전 메모리
        항목으로 대체하지 않아야 한다.
        """
        if emr_snapshot.get("success") is not True:
            raise EmrAdapterError("EMR snapshot was unsuccessful")
        payload = emr_snapshot.get("data")
        if not isinstance(payload, Mapping):
            raise EmrAdapterError("EMR snapshot data is required")
        if EmrAdapter._required_string(payload, "schema_version") != "patient-context-v1":
            raise EmrAdapterError("EMR response schema version is invalid")
        snapshot_at = EmrAdapter._datetime(payload, "snapshot_at")
        patient = payload.get("patient")
        if not isinstance(patient, Mapping):
            raise EmrAdapterError("EMR response patient is required")
        snapshot_patient_id = EmrAdapter._required_string(patient, "patient_id")
        if snapshot_patient_id != patient_id:
            raise EmrAdapterError("EMR response patient does not match the request")
        for key in (
            "observations", "diagnoses", "prescriptions", "lab_results",
            "missing_data", "delayed_data",
        ):
            if not isinstance(payload.get(key), list):
                raise EmrAdapterError(f"EMR response {key} must be an array")
        return PreparedPatientSnapshot(snapshot_patient_id, snapshot_at, payload)

    def _read(self, request: EmrReadRequest) -> tuple[PatientContext, Mapping[str, Any]]:
        try:
            payload = self._client.fetch_patient_snapshot(request)
        except EmrAdapterError:
            raise
        except Exception as error:
            raise EmrAdapterError("EMR patient-data read failed") from error
        try:
            context = self._context_builder.build(payload)
        except ValueError as error:
            raise EmrAdapterError("EMR returned an invalid patient-data payload") from error
        if context.patient_id != request.patient_id:
            raise EmrAdapterError("EMR response patient does not match the request")
        if context.encounter.encounter_id != request.encounter_id:
            raise EmrAdapterError("EMR response encounter does not match the request")
        if context.query_range.from_at != request.from_at or context.query_range.to_at != request.to_at:
            raise EmrAdapterError("EMR response range does not match the request")
        if context.query_range.timezone != request.timezone:
            raise EmrAdapterError("EMR response timezone does not match the request")
        return context, payload

    @staticmethod
    def _required_string(payload: Mapping[str, Any], key: str) -> str:
        value = payload.get(key)
        if not isinstance(value, str) or not value.strip():
            raise EmrAdapterError(f"EMR response {key} is required")
        return value

    @staticmethod
    def _datetime(payload: Mapping[str, Any], key: str) -> datetime:
        value = payload.get(key)
        if not isinstance(value, str):
            raise EmrAdapterError(f"EMR response {key} must be an ISO-8601 timestamp")
        try:
            result = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as error:
            raise EmrAdapterError(f"EMR response {key} must be an ISO-8601 timestamp") from error
        if result.tzinfo is None:
            raise EmrAdapterError(f"EMR response {key} must include a timezone")
        return result
