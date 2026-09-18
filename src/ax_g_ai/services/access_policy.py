"""D-03의 결정적 접근 제어와 환자 문맥 전환 보호 기능."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from secrets import token_urlsafe
from typing import Protocol

from ax_g_ai.adapters.emr import EmrAdapter, EmrReadRequest
from ax_g_ai.domain.patient_context import Actor, Encounter
from ax_g_ai.domain.patient_context import PatientContext
from ax_g_ai.services.audit import AuditEventType, AuditSink, audit_event


class AccessDeniedError(PermissionError):
    """권한이 없거나 현재 세션의 검증 문맥이 유효하지 않을 때 발생한다."""


@dataclass(frozen=True)
class AccessRequest:
    """권한 위임 구현체에 전달하는 최소 접근 판단 입력값.

    ``actor``는 로그인 의료진의 사용자·기관·역할·목적을, ``patient_id``와
    ``encounter``는 현재 EMR 화면의 대상 문맥을 의미한다. ``request_id``는 인증
    provider, EMR Adapter, 감사 이벤트를 연결하는 추적 식별자다.
    """

    actor: Actor
    patient_id: str
    encounter: Encounter
    request_id: str


class AccessAuthorizer(Protocol):
    """P-01 확정 후 EMR/기관의 RBAC 위임 구현체가 따라야 하는 인터페이스."""

    def is_allowed(self, request: AccessRequest) -> bool: ...


class SessionStateStore(Protocol):
    """환자 전환 때 폐기해야 하는 대화·캐시 상태의 최소 인터페이스."""

    def clear(self, session_id: str) -> None: ...


@dataclass(frozen=True)
class VerifiedContext:
    """D-03 통과 후 하위 조회·AI 요청에만 전달할 세션 한정 문맥 토큰.

    Attributes:
        token: 서버가 생성한 예측 불가능한 문자열이다. UI나 감사 로그에 기록하지 않는다.
        session_id: 현재 의료진 브라우저/애플리케이션 세션 식별자다.
        actor: 권한을 검증받은 의료진 문맥이다.
        patient_id: 허용된 환자 식별자다.
        encounter: 허용된 진료 에피소드다.
        issued_at: UTC 시간대가 포함된 발급 시각이다.
    """

    token: str
    session_id: str
    actor: Actor
    patient_id: str
    encounter: Encounter
    issued_at: datetime


class ContextGuard:
    """D-03의 권한 검증, 환자 전환 시 상태 폐기, 토큰 재검증을 담당한다.

    ``AccessAuthorizer``가 외부 RBAC·환자 관계를 결정적으로 판단하고,
    ``SessionStateStore``는 같은 세션에서 이전 환자의 대화·캐시를 지운다.
    이 Guard는 세션마다 단 하나의 ``VerifiedContext``만 유지한다. 새 환자나
    에피소드가 활성화되면 이전 토큰은 즉시 무효가 되어 하위 EMR/AI 호출에 쓸 수 없다.
    """

    def __init__(
        self,
        authorizer: AccessAuthorizer,
        session_state: SessionStateStore,
        audit_sink: AuditSink,
    ) -> None:
        self._authorizer = authorizer
        self._session_state = session_state
        self._audit_sink = audit_sink
        self._active_contexts: dict[str, VerifiedContext] = {}

    def activate(self, session_id: str, request: AccessRequest) -> VerifiedContext:
        """권한 있는 요청만 세션의 현재 환자 문맥으로 활성화한다.

        Args:
            session_id: 대화·캐시를 공유하는 애플리케이션 세션 식별자다.
            request: 의료진, 환자, 에피소드 및 추적 ID를 포함한 ``AccessRequest``다.

        Returns:
            하위 조회와 AI 요청 전에 ``require_current``으로 재검증해야 하는
            ``VerifiedContext``다.

        Raises:
            AccessDeniedError: 권한 위임이 거부되면 세션 상태를 지우고 발생한다.
        """
        if not self._authorizer.is_allowed(request):
            self._clear(session_id, request, AuditEventType.ACCESS_DENIED, "access_denied")
            raise AccessDeniedError("access to the requested patient context was denied")

        previous = self._active_contexts.get(session_id)
        if previous and (
            previous.patient_id != request.patient_id
            or previous.encounter.encounter_id != request.encounter.encounter_id
        ):
            self._clear(session_id, request, AuditEventType.CONTEXT_SWITCHED, "patient_changed")

        context = VerifiedContext(
            token=token_urlsafe(32),
            session_id=session_id,
            actor=request.actor,
            patient_id=request.patient_id,
            encounter=request.encounter,
            issued_at=datetime.now(timezone.utc),
        )
        self._active_contexts[session_id] = context
        self._audit_sink.record(
            audit_event(
                AuditEventType.ACCESS_GRANTED,
                actor_id=request.actor.user_id,
                organization_id=request.actor.organization_id,
                patient_id=request.patient_id,
                encounter_id=request.encounter.encounter_id,
                request_id=request.request_id,
                outcome="granted",
                reason_code="access_verified",
            )
        )
        return context

    def require_current(
        self,
        context: VerifiedContext,
        *,
        session_id: str,
        patient_id: str,
        encounter_id: str,
    ) -> None:
        """하위 요청에 전달된 토큰이 현재 환자 문맥과 정확히 같은지 검증한다.

        Args:
            context: ``activate``가 발급한 ``VerifiedContext``다.
            session_id: 하위 호출을 시작한 현재 세션 식별자다.
            patient_id: 하위 호출이 조회·전송하려는 환자 식별자다.
            encounter_id: 하위 호출이 조회·전송하려는 에피소드 식별자다.

        Raises:
            AccessDeniedError: 세션, 토큰, 환자 또는 에피소드가 현재 활성 문맥과
                하나라도 다르면 발생한다. 이 경우 호출자는 Adapter·AI Client를
                실행하면 안 된다.
        """
        active = self._active_contexts.get(session_id)
        if (
            active is None
            or active != context
            or context.session_id != session_id
            or context.patient_id != patient_id
            or context.encounter.encounter_id != encounter_id
        ):
            raise AccessDeniedError("verified context is not current for this request")

    def _clear(
        self,
        session_id: str,
        request: AccessRequest,
        event_type: AuditEventType,
        reason_code: str,
    ) -> None:
        self._active_contexts.pop(session_id, None)
        self._session_state.clear(session_id)
        self._audit_sink.record(
            audit_event(
                event_type,
                actor_id=request.actor.user_id,
                organization_id=request.actor.organization_id,
                patient_id=request.patient_id,
                encounter_id=request.encounter.encounter_id,
                request_id=request.request_id,
                outcome="cleared" if event_type is AuditEventType.CONTEXT_SWITCHED else "denied",
                reason_code=reason_code,
            )
        )


class AuthorizedEmrReader:
    """검증된 환자 문맥을 가진 요청만 D-01 EMR Adapter로 전달하는 진입점.

    D-03은 권한 실패·환자 전환 뒤에 Adapter가 호출되지 않아야 한다고 요구한다.
    이 클래스는 ``ContextGuard.require_current``을 먼저 실행하고 성공했을 때만
    ``EmrAdapter.read_patient_context``을 호출한다. 향후 웹/API 진입점은 직접
    Adapter를 호출하지 않고 이 클래스를 사용해야 한다.
    """

    def __init__(self, guard: ContextGuard, adapter: EmrAdapter) -> None:
        self._guard = guard
        self._adapter = adapter

    def read(
        self,
        context: VerifiedContext,
        request: EmrReadRequest,
    ) -> PatientContext:
        """현재 세션·환자·에피소드와 일치할 때만 EMR 읽기를 수행한다.

        Args:
            context: D-03이 발급하고 현재 세션에서 아직 유효한 검증 문맥이다.
            request: D-01에 전달할 환자·에피소드·조회 기간 요청이다.

        Returns:
            D-01/D-02가 반환한 ``PatientContext``다.

        Raises:
            AccessDeniedError: 문맥이 현재 세션과 일치하지 않으면 발생하며, 이때
                EMR Adapter는 호출되지 않는다.
            EmrAdapterError: 권한 검증 후 EMR 읽기 또는 payload 검증에 실패하면
                Adapter가 발생시키는 안전 오류가 그대로 전파된다.
        """
        self._guard.require_current(
            context,
            session_id=context.session_id,
            patient_id=request.patient_id,
            encounter_id=request.encounter_id,
        )
        return self._adapter.read_patient_context(request)
