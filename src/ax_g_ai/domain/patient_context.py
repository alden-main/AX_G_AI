"""정규 환자 문맥과 EMR 데이터 품질을 결정적으로 처리하는 도메인 계약.

이 모듈은 D-02(Patient Context Builder)를 구현한다. 권한 결정은 D-03의
책임이므로, 호출자는 이미 검증된 의료진·환자·진료 에피소드 문맥을 제공해야
한다. 이 모듈은 원천 데이터를 보존하고 요약·임상 규칙에서 사용 가능한지
품질 상태로 표시할 뿐, 누락 데이터를 추정하거나 보완하지 않는다.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Iterable, Mapping


class ContextBuildError(ValueError):
    """EMR 입력값으로 안전한 환자 문맥을 만들 수 없을 때 발생한다."""


class QualityStatus(str, Enum):
    USABLE = "usable"
    MISSING_REQUIRED_FIELD = "missing_required_field"
    UNMAPPED_CODE = "unmapped_code"
    UNCONFIRMED_STATUS = "unconfirmed_status"


@dataclass(frozen=True)
class CodeMapping:
    """병원 원천 코드를 승인된 표준 코드로 연결하는 매핑 데이터.

    Attributes:
        source_system: EMR이 제공한 원천 코드체계 문자열이다.
        source_code: 원천 코드체계 안의 코드 문자열이다.
        normalized_system: 승인된 표준 코드체계 문자열이다.
        normalized_code: 표준 코드체계 안의 코드 문자열이다.
    """

    source_system: str
    source_code: str
    normalized_system: str
    normalized_code: str


@dataclass(frozen=True)
class Actor:
    user_id: str
    organization_id: str
    role: str
    purpose: str


@dataclass(frozen=True)
class Encounter:
    encounter_id: str
    encounter_type: str
    status: str


@dataclass(frozen=True)
class QueryRange:
    from_at: datetime
    to_at: datetime
    timezone: str


@dataclass(frozen=True)
class Observation:
    """단일 EMR 관찰값의 원천 정보와 정규화·품질 판정 결과.

    Attributes:
        source_record_id: EMR 원천 레코드 식별자이며, 없으면 ``None``이다.
        code_system/code: EMR 원천 코드체계와 코드이며, 미제공 시 ``None``이다.
        normalized_code_system/normalized_code: 승인된 매핑 결과이며, 미매핑이면
            ``None``이다.
        value: EMR이 제공한 측정값 원본이다. 수치·문자 등 원천 타입을 보존한다.
        unit: 측정 단위 문자열이며, 없으면 ``None``이다.
        observed_at: 시간대가 포함된 측정 시각이며, 없으면 ``None``이다.
        source: 검사실·기기·수기 입력 등 원천 구분 문자열이다.
        status: EMR 결과 상태 문자열이다.
        freshness: 제공사가 전달한 최신성 상태 문자열이며, 미제공이면 ``unknown``이다.
        quality: 요약·임상 규칙 사용 가능 여부를 나타내는 ``QualityStatus``이다.
    """

    source_record_id: str | None
    code_system: str | None
    code: str | None
    normalized_code_system: str | None
    normalized_code: str | None
    value: Any
    unit: str | None
    observed_at: datetime | None
    source: str | None
    status: str | None
    freshness: str
    quality: QualityStatus

    @property
    def eligible_for_clinical_use(self) -> bool:
        return self.quality is QualityStatus.USABLE


@dataclass(frozen=True)
class DataQuality:
    """문맥 전체에서 표시해야 하는 데이터 품질 문제 목록.

    ``missing``, ``delayed``, ``unmapped``은 각각 문자열 튜플이며, 원천 데이터군
    또는 원천 레코드 식별자를 담는다. 값이 없다는 사실을 정상값으로 바꾸지 않고
    화면과 후속 정책에서 표시하도록 보존한다.
    """

    missing: tuple[str, ...]
    delayed: tuple[str, ...]
    unmapped: tuple[str, ...]


@dataclass(frozen=True)
class PatientContext:
    """D-02가 반환하는 환자·에피소드 단위의 읽기 전용 정규 문맥.

    Attributes:
        actor: D-03에서 검증되어 전달된 의료진의 사용자·기관·역할·접근 목적이다.
        patient_id: 현재 조회 대상인 EMR 환자 식별자다.
        encounter: 현재 진료 에피소드의 식별자·유형·상태다.
        query_range: 조회 시작·종료 시각과 시간대다.
        observations: 원천 식별자와 품질 상태를 보존한 관찰값 튜플이다.
        diagnoses/prescriptions/lab_results: 아직 제공사 세부 Schema가 미확정인
            원천 레코드 객체 튜플이다. Adapter가 값을 변형하지 않고 보존한다.
        data_quality: 누락·지연·미매핑 상태를 표시하기 위한 집계 정보다.
    """

    actor: Actor
    patient_id: str
    encounter: Encounter
    query_range: QueryRange
    observations: tuple[Observation, ...]
    diagnoses: tuple[Mapping[str, Any], ...]
    prescriptions: tuple[Mapping[str, Any], ...]
    lab_results: tuple[Mapping[str, Any], ...]
    data_quality: DataQuality

    @property
    def clinically_usable_observations(self) -> tuple[Observation, ...]:
        return tuple(
            observation
            for observation in self.observations
            if observation.eligible_for_clinical_use
        )


class PatientContextBuilder:
    """D-02의 EMR 원천 payload 정규화 및 데이터 품질 판정을 담당한다.

    입력은 EMR Adapter가 전달한 ``Mapping[str, Any]`` payload다. payload에는
    ``actor``, ``patient``, ``encounter``, ``range`` 객체와 ``observations`` 배열이
    포함되어야 한다. 출력은 불변 ``PatientContext``이며, 관찰값에 필수 원천 정보,
    최종 결과 상태, 승인 코드 매핑이 모두 있을 때에만 ``USABLE``로 판정한다.

    이 클래스는 권한을 검증하거나 외부 I/O를 수행하지 않는다. 불완전한 관찰값은
    보존하되 임상 사용 대상에서 제외하고, 환자·에피소드·기간 같은 문맥 자체가
    불완전하면 ``ContextBuildError``를 발생시켜 호출을 중단한다.
    """

    _OBSERVATION_FIELDS = (
        "source_record_id",
        "code_system",
        "code",
        "unit",
        "observed_at",
        "source",
        "status",
    )
    _FINAL_STATUSES = frozenset({"final", "corrected"})

    def __init__(self, code_mappings: Iterable[CodeMapping] = ()) -> None:
        self._code_mappings = {
            (item.source_system, item.source_code): item for item in code_mappings
        }

    def build(self, payload: Mapping[str, Any]) -> PatientContext:
        """EMR payload를 검증하여 정규 ``PatientContext``로 변환한다.

        Args:
            payload: ``Mapping[str, Any]`` 타입의 EMR 스냅샷이다. ``actor``(의료진
                문맥), ``patient``(환자 ID), ``encounter``(에피소드), ``range``(시간대
                포함 기간), ``observations``(관찰값 배열)를 포함해야 한다.

        Returns:
            원천 레코드와 품질 상태를 보존한 불변 ``PatientContext``. 코드 미매핑,
            누락 단위, 미확정 결과 상태는 오류로 숨기지 않고 ``data_quality``와
            관찰값 ``quality``에 반영된다.

        Raises:
            ContextBuildError: 문맥 객체·배열 형식, 필수 식별자 또는 시간대가
                포함된 시각 형식이 유효하지 않을 때 발생한다.
        """
        actor_data = self._mapping(payload, "actor")
        patient_data = self._mapping(payload, "patient")
        encounter_data = self._mapping(payload, "encounter")
        range_data = self._mapping(payload, "range")

        observations = tuple(
            self._observation(index, item)
            for index, item in enumerate(self._list(payload, "observations"))
            if isinstance(item, Mapping)
        )
        if len(observations) != len(self._list(payload, "observations")):
            raise ContextBuildError("observations must contain objects")

        missing = list(self._string_list(payload, "missing_data"))
        delayed = list(self._string_list(payload, "delayed_data"))
        unmapped = [
            observation.source_record_id or f"observations[{index}]"
            for index, observation in enumerate(observations)
            if observation.quality is QualityStatus.UNMAPPED_CODE
        ]

        return PatientContext(
            actor=Actor(
                user_id=self._required_string(actor_data, "user_id", "actor"),
                organization_id=self._required_string(
                    actor_data, "organization_id", "actor"
                ),
                role=self._required_string(actor_data, "role", "actor"),
                purpose=self._required_string(actor_data, "purpose", "actor"),
            ),
            patient_id=self._required_string(patient_data, "patient_id", "patient"),
            encounter=Encounter(
                encounter_id=self._required_string(
                    encounter_data, "encounter_id", "encounter"
                ),
                encounter_type=self._required_string(
                    encounter_data, "type", "encounter"
                ),
                status=self._required_string(encounter_data, "status", "encounter"),
            ),
            query_range=QueryRange(
                from_at=self._datetime(range_data, "from", "range"),
                to_at=self._datetime(range_data, "to", "range"),
                timezone=self._required_string(range_data, "timezone", "range"),
            ),
            observations=observations,
            diagnoses=self._records(payload, "diagnoses"),
            prescriptions=self._records(payload, "prescriptions"),
            lab_results=self._records(payload, "lab_results"),
            data_quality=DataQuality(
                missing=tuple(missing), delayed=tuple(delayed), unmapped=tuple(unmapped)
            ),
        )

    def _observation(self, index: int, data: Mapping[str, Any]) -> Observation:
        """관찰값 원천 객체 하나를 보존하고 임상 사용 가능 여부를 판정한다.

        Args:
            index: 오류 메시지와 대체 식별자에 쓰는 ``observations`` 배열의 위치다.
            data: 관찰값 원천 ``Mapping[str, Any]``이다. 원천 ID, 코드, 단위, 측정
                시각, 원천, 결과 상태를 읽고 ``value``는 원본 타입 그대로 보존한다.

        Returns:
            정규 코드와 ``QualityStatus``를 포함한 ``Observation``. 필수 필드 누락,
            최종/정정 전 결과, 승인되지 않은 코드 매핑은 임상 사용 불가로 반환한다.

        Raises:
            ContextBuildError: 제공된 문자열이나 시각이 예상 타입·ISO-8601 시간대
                형식을 따르지 않을 때 발생한다.
        """
        values = {field: self._optional_string(data, field) for field in self._OBSERVATION_FIELDS}
        observed_at = self._optional_datetime(data, "observed_at", f"observations[{index}]")
        missing_required = any(values[field] is None for field in self._OBSERVATION_FIELDS)
        mapping = None
        if values["code_system"] is not None and values["code"] is not None:
            mapping = self._code_mappings.get((values["code_system"], values["code"]))

        if missing_required:
            quality = QualityStatus.MISSING_REQUIRED_FIELD
        elif values["status"].lower() not in self._FINAL_STATUSES:
            quality = QualityStatus.UNCONFIRMED_STATUS
        elif mapping is None:
            quality = QualityStatus.UNMAPPED_CODE
        else:
            quality = QualityStatus.USABLE

        return Observation(
            source_record_id=values["source_record_id"],
            code_system=values["code_system"],
            code=values["code"],
            normalized_code_system=mapping.normalized_system if mapping else None,
            normalized_code=mapping.normalized_code if mapping else None,
            value=data.get("value"),
            unit=values["unit"],
            observed_at=observed_at,
            source=values["source"],
            status=values["status"],
            freshness=self._optional_string(data, "freshness") or "unknown",
            quality=quality,
        )

    @staticmethod
    def _mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
        value = payload.get(key)
        if not isinstance(value, Mapping):
            raise ContextBuildError(f"{key} must be an object")
        return value

    @staticmethod
    def _list(payload: Mapping[str, Any], key: str) -> list[Any]:
        value = payload.get(key, [])
        if not isinstance(value, list):
            raise ContextBuildError(f"{key} must be an array")
        return value

    def _records(self, payload: Mapping[str, Any], key: str) -> tuple[Mapping[str, Any], ...]:
        records = self._list(payload, key)
        if not all(isinstance(item, Mapping) for item in records):
            raise ContextBuildError(f"{key} must contain objects")
        return tuple(records)

    def _string_list(self, payload: Mapping[str, Any], key: str) -> list[str]:
        values = self._list(payload, key)
        if not all(isinstance(item, str) and item.strip() for item in values):
            raise ContextBuildError(f"{key} must contain non-empty strings")
        return values

    @staticmethod
    def _required_string(data: Mapping[str, Any], key: str, parent: str) -> str:
        value = PatientContextBuilder._optional_string(data, key)
        if value is None:
            raise ContextBuildError(f"{parent}.{key} is required")
        return value

    @staticmethod
    def _optional_string(data: Mapping[str, Any], key: str) -> str | None:
        value = data.get(key)
        if value is None:
            return None
        if not isinstance(value, str) or not value.strip():
            raise ContextBuildError(f"{key} must be a non-empty string when supplied")
        return value

    @staticmethod
    def _datetime(data: Mapping[str, Any], key: str, parent: str) -> datetime:
        value = PatientContextBuilder._optional_datetime(data, key, parent)
        if value is None:
            raise ContextBuildError(f"{parent}.{key} is required")
        return value

    @staticmethod
    def _optional_datetime(
        data: Mapping[str, Any], key: str, parent: str
    ) -> datetime | None:
        value = data.get(key)
        if value is None:
            return None
        if not isinstance(value, str):
            raise ContextBuildError(f"{parent}.{key} must be an ISO-8601 string")
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as error:
            raise ContextBuildError(
                f"{parent}.{key} must be an ISO-8601 timestamp"
            ) from error
        if parsed.tzinfo is None:
            raise ContextBuildError(f"{parent}.{key} must include a timezone")
        return parsed
