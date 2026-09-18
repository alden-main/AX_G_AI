"""배포 환경 설정으로 FastAPI 의존성을 조립하는 모듈.

실제 병원 인증 연동 전에는 ``AX_G_CHAT_TEST_MODE=true``일 때만 가상 문맥을 써서
AI-Tank 연결을 시험한다. 이 모드는 실제 환자 정보·EMR·운영 권한을 사용하지 않는다.
"""

from __future__ import annotations

from datetime import date, datetime
import os

from fastapi import Request

from ax_g_ai.adapters.ai_tank import AiTankClient, UrllibHttpTransport
from ax_g_ai.api import ChatContextProvider, ResolvedChatContext, create_app
from ax_g_ai.config import AiTankSettings
from ax_g_ai.domain.patient_context import Actor, Encounter, PatientContext, QueryRange, DataQuality
from ax_g_ai.services.access_policy import AccessRequest, ContextGuard
from ax_g_ai.services.audit import InMemoryAuditSink
from ax_g_ai.services.clinical_chat import ClinicalChatService
from ax_g_ai.services.evaluation import ChatVersion, EvaluationGate, EvaluationRecord
from ax_g_ai.services.knowledge import ApprovedKnowledgeStore, KnowledgeDocument


class _TestModeAuthorizer:
    def is_allowed(self, request: AccessRequest) -> bool:
        return True


class _TestModeSessionState:
    def clear(self, session_id: str) -> None:
        pass


class TestModeContextProvider(ChatContextProvider):
    """로컬 연결 시험에만 쓰는 가상 의료진·환자 문맥 공급자."""

    def __init__(self, guard: ContextGuard) -> None:
        self._guard = guard
        self._actor = Actor("local-test-clinician", "local-test-org", "physician", "treatment")
        self._encounter = Encounter("local-test-encounter", "test", "open")
        self._patient_context = PatientContext(
            actor=self._actor,
            patient_id="local-test-patient",
            encounter=self._encounter,
            query_range=QueryRange(
                datetime.fromisoformat("2026-01-01T00:00:00+00:00"),
                datetime.fromisoformat("2026-12-31T23:59:59+00:00"),
                "UTC",
            ),
            observations=(), diagnoses=(), prescriptions=(), lab_results=(),
            data_quality=DataQuality(missing=(), delayed=(), unmapped=()),
        )

    def resolve(self, request: Request, request_id: str) -> ResolvedChatContext:
        verified = self._guard.activate(
            "local-test-session",
            AccessRequest(self._actor, self._patient_context.patient_id, self._encounter, request_id),
        )
        return ResolvedChatContext(verified, self._patient_context)


def create_test_app():
    """AI-Tank 연결 검증용 앱을 생성한다. 운영 서비스 구성에는 사용하지 않는다."""
    settings = AiTankSettings.from_environment()
    today = date.today()
    version = ChatVersion("ai-tank", "configured-endpoint", "local-test", "local-test-guide-v1")
    audit = InMemoryAuditSink()
    guard = ContextGuard(_TestModeAuthorizer(), _TestModeSessionState(), audit)
    service = ClinicalChatService(
        guard,
        ApprovedKnowledgeStore((
            KnowledgeDocument(
                "local-test-guide", "로컬 연결 시험용 근거", "v1", today,
                "연결 시험", True, today, None,
            ),
        )),
        AiTankClient(UrllibHttpTransport(), settings.api_key, settings.endpoint, settings.timeout_seconds),
        audit,
        EvaluationGate((
            EvaluationRecord(version, "local-test-set", today, "local-test-reviewer", True, "local-test-only"),
        )),
        version,
    )
    return create_app(service, TestModeContextProvider(guard))


def create_runtime_app():
    """명시적으로 허용된 로컬 시험 모드만 실제 Provider 호출을 활성화한다."""
    if os.getenv("AX_G_CHAT_TEST_MODE", "false").lower() == "true":
        return create_test_app()
    return create_app()
