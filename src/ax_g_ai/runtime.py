"""간소화된 Bridge API의 런타임 조립과 프로세스 메모리 Snapshot 저장소."""

from __future__ import annotations

from datetime import date, datetime
from typing import Mapping

from fastapi import Request

from ax_g_ai.adapters.provider import build_provider_client
from ax_g_ai.adapters.emr import EmrAdapterError
from ax_g_ai.adapters.main_bridge import RawEmrPayloadMapper
from ax_g_ai.api import (
    PatientContextNotReadyError,
    PatientMismatchError,
    PatientContextUnavailableError,
    create_app,
)
from ax_g_ai.config import AISettings, ConfigurationError, reference_data_database_path
from ax_g_ai.services.reference_lookup import SqliteReferenceLookup
from ax_g_ai.services.audit import InMemoryAuditSink
from ax_g_ai.services.clinical_chat import (
    ClinicalChatProviderUnavailableError,
    ClinicalChatService,
)
from ax_g_ai.services.evaluation import ChatVersion, EvaluationGate, EvaluationRecord
from ax_g_ai.services.knowledge import ApprovedKnowledgeStore, KnowledgeDocument
from ax_g_ai.services.prepared_patient_summary import PreparedPatientSummary
from ax_g_ai.services.access_policy import ContextGuard


class PatientContextStore:
    """검증된 Snapshot을 해당 프로세스의 환자 ID별 메모리에만 보관한다."""

    def __init__(self, service: ClinicalChatService, mapper: RawEmrPayloadMapper | None = None) -> None:
        self._service = service
        self._mapper = mapper or RawEmrPayloadMapper()
        self._snapshots: dict[str, tuple[Mapping[str, object], PreparedPatientSummary | None, datetime]] = {}

    def prepare(
        self, patient_id: str, emr_payload: Mapping[str, object]
    ) -> tuple[PreparedPatientSummary, datetime]:
        # A refresh starts by invalidating the old item.  Neither malformed EMR
        # data nor a failed summary may leave stale patient context available.
        self._snapshots.pop(patient_id, None)
        context = self._map(patient_id, emr_payload)
        try:
            summary = self._service.summarize_prepared_snapshot(patient_id=patient_id, snapshot=context.data)
        except ClinicalChatProviderUnavailableError:
            raise
        self._snapshots[patient_id] = (context.data, summary, context.updated_at)
        return summary, context.updated_at

    def prepare_without_summary(
        self, patient_id: str, emr_payload: Mapping[str, object]
    ) -> tuple[Mapping[str, object], datetime]:
        """Store the mapped context at once so a summary can stream afterwards.

        The caller must ``discard`` the item if the streamed summary fails.
        """
        self._snapshots.pop(patient_id, None)
        context = self._map(patient_id, emr_payload)
        self._snapshots[patient_id] = (context.data, None, context.updated_at)
        return context.data, context.updated_at

    def discard(self, patient_id: str) -> None:
        self._snapshots.pop(patient_id, None)

    def summarize_once(self, patient_id: str, emr_payload: Mapping[str, object]) -> tuple[str, datetime]:
        context = self._map(patient_id, emr_payload)
        return self._service.summarize_once(snapshot=context.data), context.updated_at

    def map_once(self, patient_id: str, emr_payload: Mapping[str, object]) -> tuple[Mapping[str, object], datetime]:
        """Map an EMR payload for a one-shot summary without storing it."""
        context = self._map(patient_id, emr_payload)
        return context.data, context.updated_at

    def _map(self, patient_id: str, emr_payload: Mapping[str, object]):
        try:
            return self._mapper.map(patient_id, emr_payload)
        except EmrAdapterError as error:
            if "patient does not match" in str(error):
                raise PatientMismatchError() from error
            raise PatientContextUnavailableError(str(error)) from error

    def get(self, patient_id: str) -> tuple[Mapping[str, object], datetime]:
        try:
            snapshot, _summary, updated_at = self._snapshots[patient_id]
            return snapshot, updated_at
        except KeyError as error:
            raise PatientContextNotReadyError() from error


class _AllowAllAuthorizer:
    def is_allowed(self, request: object) -> bool:
        return True


class _NoopSessionState:
    def clear(self, session_id: str) -> None:
        return None


def _services(settings: AISettings) -> ClinicalChatService:
    """신규 경로는 provider 호출만 사용하지만 기존 서비스 경계를 재사용한다."""
    today = date.today()
    version = ChatVersion(settings.provider, settings.model, "bridge-v2", "not-used")
    audit = InMemoryAuditSink()
    guard = ContextGuard(_AllowAllAuthorizer(), _NoopSessionState(), audit)
    clinical = ClinicalChatService(
        guard,
        ApprovedKnowledgeStore((KnowledgeDocument("placeholder", "placeholder", "v1", today, "", True, today, None),)),
        build_provider_client(settings),
        audit,
        EvaluationGate((EvaluationRecord(version, "runtime", today, "runtime", True, "runtime"),)),
        version,
    )
    return clinical


def create_runtime_app():
    """필수 server-to-server 설정이 없으면 fail-closed 앱을 생성한다."""
    try:
        ai_settings = AISettings.from_environment()
    except ConfigurationError:
        return create_app()
    service = _services(ai_settings)
    try:
        database_path = reference_data_database_path()
    except ConfigurationError:
        return create_app()
    lookup = None if database_path is None else SqliteReferenceLookup(database_path)
    mapper = RawEmrPayloadMapper(lookup)
    return create_app(service, PatientContextStore(service, mapper))
