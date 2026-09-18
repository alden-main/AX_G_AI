"""의료진 챗봇의 HTTP 진입점.

브라우저 요청 본문에는 의료진 질문과 승인 지식 ID만 들어간다. 인증된 의료진과
현재 환자·에피소드 문맥은 메인 서버/인증 계층이 제공하는 ``ChatContextProvider``가
결정하며, 이 모듈은 그 값을 요청 본문으로 대체하지 않는다.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Literal, Protocol

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ax_g_ai.adapters.ai_tank import ChatHistoryItem
from ax_g_ai.domain.patient_context import PatientContext
from ax_g_ai.services.access_policy import AccessDeniedError, VerifiedContext
from ax_g_ai.services.clinical_chat import (
    ClinicalChatProviderUnavailableError,
    ClinicalChatRequest,
    ClinicalChatService,
    ClinicalChatUnavailableError,
)
from ax_g_ai.services.evaluation import EvaluationNotApprovedError


class ChatContextUnavailableError(RuntimeError):
    """인증·환자 문맥 공급자가 아직 연결되지 않았을 때 발생한다."""


class BridgeAuthenticationError(PermissionError):
    """Bridge 서버 간 인증을 확인할 수 없을 때 발생한다."""


class BridgeContextMismatchError(PermissionError):
    """요청과 Snapshot의 환자·에피소드 문맥이 다를 때 발생한다."""


class BridgeSnapshotUnavailableError(RuntimeError):
    """현재 환자 Snapshot을 안전하게 읽을 수 없을 때 발생한다."""


@dataclass(frozen=True)
class ResolvedChatContext:
    """인증 계층이 제공하고 검증한 현재 세션의 환자 문맥."""

    verified_context: VerifiedContext
    patient_context: PatientContext


@dataclass(frozen=True)
class SnapshotInfo:
    """Bridge가 조립한 Snapshot의 화면 표시용 최소 메타데이터다."""

    snapshot_at: datetime
    assembly_status: Literal["complete", "partial"]
    missing_data: tuple[str, ...]


@dataclass(frozen=True)
class ResolvedBridgeChatContext:
    """내부 Bridge 요청의 검증 문맥과 대조된 Snapshot 메타데이터다."""

    chat_context: ResolvedChatContext
    context_id: str
    context_info: SnapshotInfo


class ChatContextProvider(Protocol):
    """메인 서버의 인증 세션과 EMR 문맥을 API에 제공하는 경계다."""

    def resolve(self, request: Request, request_id: str) -> ResolvedChatContext: ...


class BridgeChatContextProvider(Protocol):
    """서버 간 인증 및 Snapshot 대조를 마친 내부 API 문맥 공급자 경계다.

    구현체는 Bridge 서비스 인증을 먼저 확인하고, 요청 ``context_id``와 환자·에피소드가
    Snapshot 응답과 모두 일치하는지 확인해야 한다. 이 API 모듈은 헤더나 body만으로
    그 검증을 대신하지 않는다.
    """

    def resolve_internal(
        self, request: Request, body: "InternalClinicalChatRequestBody", request_id: str
    ) -> ResolvedBridgeChatContext: ...


class UnconfiguredChatContextProvider:
    """운영 인증 연동 전에는 어떤 챗봇 요청도 처리하지 않는 기본 구현체."""

    def resolve(self, request: Request, request_id: str) -> ResolvedChatContext:
        raise ChatContextUnavailableError("chat context provider is not configured")


class UnconfiguredBridgeChatContextProvider:
    """Bridge 인증·Snapshot 연결 전에는 내부 API를 fail-closed로 유지한다."""

    def resolve_internal(
        self, request: Request, body: "InternalClinicalChatRequestBody", request_id: str
    ) -> ResolvedBridgeChatContext:
        raise ChatContextUnavailableError("bridge chat context provider is not configured")


class HistoryItemBody(BaseModel):
    inputs: str = Field(min_length=1)
    outputs: str = Field(min_length=1)


class ClinicalChatRequestBody(BaseModel):
    """프론트엔드 계약의 HTTP 요청 body. 환자 식별자·EMR 원천값은 포함하지 않는다."""

    question: str = Field(min_length=1)
    language: str = Field(min_length=1)
    knowledge_ids: list[str] = Field(min_length=1)
    request_id: str = Field(min_length=1)
    history: list[HistoryItemBody] = Field(default_factory=list)


class InternalClinicalChatRequestBody(BaseModel):
    """Bridge 전용 요청. 브라우저 공개 API에서 사용하지 않는다."""

    context_id: str = Field(min_length=1)
    patient_id: str = Field(min_length=1)
    encounter_id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    language: str = Field(min_length=1)
    knowledge_ids: list[str] = Field(min_length=1)
    history: list[HistoryItemBody] = Field(default_factory=list)


class EvidenceBody(BaseModel):
    document_id: str
    title: str
    version: str
    published_on: date
    citation_location: str


class ClinicalChatResponseBody(BaseModel):
    answer: str
    evidence: list[EvidenceBody]
    limitations: list[str]


class ContextInfoBody(BaseModel):
    snapshot_at: datetime
    assembly_status: Literal["complete", "partial"]
    missing_data: list[str]


class InternalClinicalChatResponseBody(ClinicalChatResponseBody):
    request_id: str
    context_id: str
    context_info: ContextInfoBody


class InternalErrorBody(BaseModel):
    code: str
    message: str
    request_id: str


def create_app(
    service: ClinicalChatService | None = None,
    context_provider: ChatContextProvider | None = None,
    bridge_context_provider: BridgeChatContextProvider | None = None,
) -> FastAPI:
    """동작 가능한 ASGI 앱을 만든다.

    ``service``와 ``context_provider``는 애플리케이션 조립 계층에서 주입한다.
    기본 앱은 상태 점검에는 사용할 수 있지만, 인증 연동이 없으므로 챗봇 요청을
    503으로 거부한다. 이 기본값은 임의 헤더나 브라우저 body로 권한을 우회하지 않기
    위한 안전 장치다.
    """
    app = FastAPI(title="AX-G AI Clinical Chat API", version="0.1.0")
    provider = context_provider or UnconfiguredChatContextProvider()
    bridge_provider = bridge_context_provider or UnconfiguredBridgeChatContextProvider()

    @app.get("/health", tags=["operations"])
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post(
        "/api/clinical-chat",
        response_model=ClinicalChatResponseBody,
        tags=["clinical-chat"],
    )
    def clinical_chat(body: ClinicalChatRequestBody, request: Request) -> ClinicalChatResponseBody:
        if service is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Clinical chat service is not configured.",
            )
        try:
            context = provider.resolve(request, body.request_id)
            response = service.answer(
                context.verified_context,
                context.patient_context,
                ClinicalChatRequest(
                    question=body.question,
                    language=body.language,
                    knowledge_ids=tuple(body.knowledge_ids),
                    request_id=body.request_id,
                    history=tuple(
                        ChatHistoryItem(inputs=item.inputs, outputs=item.outputs)
                        for item in body.history
                    ),
                ),
                on_date=date.today(),
            )
        except AccessDeniedError:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="The current patient context is not authorized.",
            ) from None
        except ClinicalChatProviderUnavailableError:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="The AI provider is temporarily unavailable.",
            ) from None
        except (EvaluationNotApprovedError, ClinicalChatUnavailableError):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Clinical chat is unavailable without approved evidence and evaluation.",
            ) from None
        except ChatContextUnavailableError:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="The authenticated patient context is not configured.",
            ) from None

        return ClinicalChatResponseBody(
            answer=response.answer,
            evidence=[
                EvidenceBody(
                    document_id=evidence.document_id,
                    title=evidence.title,
                    version=evidence.version,
                    published_on=evidence.published_on,
                    citation_location=evidence.citation_location,
                )
                for evidence in response.evidence
            ],
            limitations=list(response.limitations),
        )

    @app.post(
        "/internal/v1/clinical-chat",
        response_model=InternalClinicalChatResponseBody,
        responses={
            401: {"model": InternalErrorBody}, 403: {"model": InternalErrorBody},
            409: {"model": InternalErrorBody}, 502: {"model": InternalErrorBody},
            503: {"model": InternalErrorBody},
        },
        tags=["internal"],
    )
    def internal_clinical_chat(
        body: InternalClinicalChatRequestBody, request: Request
    ) -> InternalClinicalChatResponseBody | JSONResponse:
        """Bridge 서버만 호출하는 문맥 대조형 챗봇 경로다."""
        request_id = request.headers.get("X-Request-ID", "").strip()
        if not request_id:
            return _internal_error(400, "INVALID_CONTEXT", "Request context is invalid.", "")
        if service is None:
            return _internal_error(503, "CLINICAL_CHAT_UNAVAILABLE", "Clinical chat is unavailable.", request_id)
        try:
            resolved = bridge_provider.resolve_internal(request, body, request_id)
            response = service.answer(
                resolved.chat_context.verified_context,
                resolved.chat_context.patient_context,
                ClinicalChatRequest(
                    question=body.question, language=body.language,
                    knowledge_ids=tuple(body.knowledge_ids), request_id=request_id,
                    history=tuple(ChatHistoryItem(inputs=item.inputs, outputs=item.outputs) for item in body.history),
                ),
                on_date=date.today(),
            )
        except BridgeAuthenticationError:
            return _internal_error(401, "UNAUTHENTICATED", "Bridge authentication failed.", request_id)
        except (BridgeContextMismatchError, AccessDeniedError):
            return _internal_error(403, "CONTEXT_MISMATCH", "The current patient context is not authorized.", request_id)
        except BridgeSnapshotUnavailableError:
            return _internal_error(502, "EMR_UNAVAILABLE", "Patient context is temporarily unavailable.", request_id)
        except ClinicalChatProviderUnavailableError:
            return _internal_error(502, "AI_PROVIDER_UNAVAILABLE", "The AI provider is temporarily unavailable.", request_id)
        except (EvaluationNotApprovedError, ClinicalChatUnavailableError, ChatContextUnavailableError):
            return _internal_error(409, "CLINICAL_CHAT_UNAVAILABLE", "Clinical chat is unavailable.", request_id)

        return InternalClinicalChatResponseBody(
            request_id=request_id,
            context_id=resolved.context_id,
            answer=response.answer,
            evidence=[EvidenceBody(**evidence.__dict__) for evidence in response.evidence],
            limitations=list(response.limitations),
            context_info=ContextInfoBody(
                snapshot_at=resolved.context_info.snapshot_at,
                assembly_status=resolved.context_info.assembly_status,
                missing_data=list(resolved.context_info.missing_data),
            ),
        )

    return app


def _internal_error(status_code: int, code: str, message: str, request_id: str) -> JSONResponse:
    """Bridge 계약의 원문 PHI 없는 오류 envelope를 만든다."""
    return JSONResponse(
        status_code=status_code,
        content=InternalErrorBody(code=code, message=message, request_id=request_id).model_dump(mode="json"),
    )


# ``runtime``은 순환 import를 피하기 위해 모든 API 선언 뒤에 불러온다.
from ax_g_ai.runtime import create_runtime_app

app = create_runtime_app()
