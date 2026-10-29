"""D-05 의료진 전용 챗봇의 근거·권한·Provider 호출 오케스트레이션."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import logging
from collections.abc import Iterator
from typing import Any, Mapping

from ax_g_ai.adapters.openai import OpenAIProviderError, OpenAIRequest, ProviderClient
from ax_g_ai.domain.patient_context import PatientContext
from ax_g_ai.services.access_policy import AccessDeniedError, ContextGuard, VerifiedContext
from ax_g_ai.services.audit import AuditEventType, AuditSink, audit_event
from ax_g_ai.services.evaluation import ChatVersion, EvaluationGate
from ax_g_ai.services.deidentified_snapshot import deidentified_snapshot
from ax_g_ai.services.knowledge import ApprovedKnowledgeStore, KnowledgeDocument
from ax_g_ai.services.patient_context_summary import (
    PatientContextSummaryBuilder,
    compose_provider_question,
)
from ax_g_ai.services.prepared_patient_summary import (
    PATIENT_SUMMARY_OUTPUT_SCHEMA,
    PreparedPatientSummary,
)
from ax_g_ai.prompts.clinical import (
    CLINICAL_INSTRUCTIONS,
    INITIAL_PATIENT_SUMMARY_REQUEST,
    ONE_SHOT_PATIENT_SUMMARY_REQUEST,
    compose_prepared_snapshot_question as _compose_prepared_snapshot_question,
)

logger = logging.getLogger("uvicorn.error")


class ClinicalChatUnavailableError(RuntimeError):
    """근거 또는 검증 문맥이 부족해 의료진 챗봇 답변을 안전하게 만들 수 없을 때 발생한다."""


class ClinicalChatProviderUnavailableError(ClinicalChatUnavailableError):
    """승인 근거는 있으나 AI Provider 호출이 실패했을 때 발생한다."""


@dataclass(frozen=True)
class ClinicalChatRequest:
    """의료진 챗봇의 서버 측 요청 DTO.

    ``question``은 의료진 질문이고, D-05가 비식별 최소 환자 요약·승인 근거 메타데이터와
    함께 Provider 요청으로 조립한다. ``knowledge_ids``는 답변 근거로 화면에 표시할 승인 문서 후보 식별자다. ``request_id``는 질문 원문을 남기지 않고
    AI 호출 감사 이벤트를 연결하는 추적 식별자다.
    """

    question: str
    language: str
    knowledge_ids: tuple[str, ...]
    request_id: str


@dataclass(frozen=True)
class KnowledgeEvidence:
    """응답에 표시하는 승인 지식문서의 식별·버전·인용 위치."""

    document_id: str
    title: str
    version: str
    published_on: date
    citation_location: str


@dataclass(frozen=True)
class ClinicalChatResponse:
    """의료진 챗봇이 반환하는 답변·근거·제한사항 envelope."""

    answer: str
    evidence: tuple[KnowledgeEvidence, ...]
    limitations: tuple[str, ...]


class ClinicalChatService:
    """D-05를 구현하는 의료진 전용 비스트리밍 챗봇 서비스.

    이 서비스는 D-03의 현재 문맥을 먼저 확인하고, D-06에서 승인·유효한 지식문서를
    확보한 후에만 OpenAI Responses API Client를 호출한다. `PatientContextSummaryBuilder`는
    환자 식별자·원천 레코드 ID·원문 진료기록을 제외한 비식별 최소 관찰값만 Provider
    요청에 포함한다. 근거가 없거나 환자 문맥이
    불일치하면 Provider를 호출하지 않고 안전한 오류를 반환한다.
    """

    def __init__(
        self,
        guard: ContextGuard,
        knowledge_store: ApprovedKnowledgeStore,
        openai: ProviderClient,
        audit_sink: AuditSink,
        evaluation_gate: EvaluationGate,
        version: ChatVersion,
        summary_builder: PatientContextSummaryBuilder | None = None,
    ) -> None:
        self._guard = guard
        self._knowledge_store = knowledge_store
        self._openai = openai
        self._audit_sink = audit_sink
        self._evaluation_gate = evaluation_gate
        self._version = version
        self._summary_builder = summary_builder or PatientContextSummaryBuilder()

    def answer(
        self,
        verified_context: VerifiedContext,
        patient_context: PatientContext,
        request: ClinicalChatRequest,
        *,
        on_date: date,
    ) -> ClinicalChatResponse:
        """검증된 환자 문맥과 승인 근거가 있을 때만 의료지식 답변을 생성한다.

        Args:
            verified_context: D-03이 현재 세션에 발급한 검증 문맥 토큰이다.
            patient_context: D-02가 만든 현재 환자·에피소드 문맥이다. D-05는 이 중
                승인된 비식별 최소 요약만 Provider에 전달한다.
            request: 질문, 언어, 근거 문서 ID, 요청 추적 ID를 담은 DTO다.
            on_date: 승인 문서 유효기간을 판정할 기준일이다.

        Returns:
            답변 문자열과 승인 문서별 식별자·버전·인용 위치, 고정 제한사항을 가진
            ``ClinicalChatResponse``다.

        Raises:
            AccessDeniedError: 세션·환자·에피소드가 현재 검증 문맥과 다를 때 발생한다.
            ClinicalChatUnavailableError: 승인되고 유효한 근거 문서가 없을 때 발생한다.
        """
        self._guard.require_current(
            verified_context,
            session_id=verified_context.session_id,
            patient_id=patient_context.patient_id,
            encounter_id=patient_context.encounter.encounter_id,
        )
        self._evaluation_gate.require_approved(self._version)
        evidence_documents = self._knowledge_store.get_approved(request.knowledge_ids, on_date)
        if not evidence_documents:
            raise ClinicalChatUnavailableError("approved knowledge evidence is unavailable")
        self._record_provider_event(
            AuditEventType.AI_REQUEST_STARTED, verified_context, request.request_id, "started"
        )
        try:
            provider_answer = self._openai.respond(
                OpenAIRequest(
                    instructions=CLINICAL_INSTRUCTIONS,
                    input=compose_provider_question(
                        request.question,
                        self._summary_builder.build(patient_context),
                        evidence_documents,
                    ),
                )
            )
        except Exception as error:
            self._log_provider_failure("clinical_chat", error)
            self._record_provider_event(
                AuditEventType.AI_REQUEST_FAILED, verified_context, request.request_id, "failed"
            )
            raise ClinicalChatProviderUnavailableError(
                "medical knowledge response is unavailable"
            ) from error
        self._record_provider_event(
            AuditEventType.AI_REQUEST_SUCCEEDED, verified_context, request.request_id, "succeeded"
        )
        return ClinicalChatResponse(
            answer=provider_answer.answer,
            evidence=tuple(self._evidence(document) for document in evidence_documents),
            limitations=(),
        )

    def answer_prepared_snapshot(
        self, *, patient_id: str, question: str, snapshot: Mapping[str, Any], reference_record: str | None = None
    ) -> str:
        """준비된 동일 환자의 최소 EMR 요약으로 한국어 답변을 생성한다."""
        try:
            response = self._openai.respond(
                OpenAIRequest(
                    instructions=CLINICAL_INSTRUCTIONS,
                    input=compose_prepared_snapshot_question(question, snapshot, reference_record),
                )
            )
        except Exception as error:
            self._log_provider_failure("prepared_snapshot_chat", error)
            raise ClinicalChatProviderUnavailableError("medical response is unavailable") from error
        return response.answer

    def stream_prepared_snapshot(
        self, *, patient_id: str, question: str, snapshot: Mapping[str, Any], reference_record: str | None = None
    ) -> Iterator[str]:
        """Yield a prepared-snapshot answer as provider text becomes available."""
        try:
            yield from self._openai.stream(
                OpenAIRequest(
                    instructions=CLINICAL_INSTRUCTIONS,
                    input=compose_prepared_snapshot_question(question, snapshot, reference_record),
                )
            )
        except Exception as error:
            self._log_provider_failure("prepared_snapshot_chat_stream", error)
            raise ClinicalChatProviderUnavailableError("medical response is unavailable") from error

    def summarize_prepared_snapshot(self, *, patient_id: str, snapshot: Mapping[str, Any]) -> PreparedPatientSummary:
        """준비 시점에 최소 EMR 문맥으로 한국어 환자 상태 요약을 생성한다."""
        try:
            response = self._openai.respond(
                OpenAIRequest(
                    instructions=CLINICAL_INSTRUCTIONS,
                    input=compose_prepared_snapshot_question(
                        INITIAL_PATIENT_SUMMARY_REQUEST,
                        snapshot,
                    ),
                    output_schema=PATIENT_SUMMARY_OUTPUT_SCHEMA,
                )
            )
        except Exception as error:
            self._log_provider_failure("patient_context_summary", error)
            raise ClinicalChatProviderUnavailableError("patient summary is unavailable") from error
        try:
            return PreparedPatientSummary.from_provider_text(response.answer)
        except OpenAIProviderError as error:
            self._log_provider_failure("patient_context_summary_schema", error)
            raise ClinicalChatProviderUnavailableError("patient summary is unavailable") from error

    def summarize_once(self, *, snapshot: Mapping[str, Any]) -> str:
        """비식별 EMR에서 독립적인 간결 요약을 한 번 생성한다."""
        try:
            response = self._openai.respond(OpenAIRequest(
                instructions=CLINICAL_INSTRUCTIONS,
                input=compose_prepared_snapshot_question(ONE_SHOT_PATIENT_SUMMARY_REQUEST, snapshot),
            ))
            if not response.answer.strip():
                raise OpenAIProviderError("OpenAI returned an empty patient summary")
            return response.answer.strip()
        except Exception as error:
            self._log_provider_failure("one_shot_patient_summary", error)
            raise ClinicalChatProviderUnavailableError("patient summary is unavailable") from error

    def stream_summary_once(self, *, snapshot: Mapping[str, Any]) -> Iterator[str]:
        """Yield the one-shot EMR summary as provider text becomes available.

        Unlike ``summarize_prepared_snapshot`` this has no JSON schema, so the
        first sentence can be shown before the whole summary is generated.
        """
        try:
            yield from self._openai.stream(
                OpenAIRequest(
                    instructions=CLINICAL_INSTRUCTIONS,
                    input=compose_prepared_snapshot_question(ONE_SHOT_PATIENT_SUMMARY_REQUEST, snapshot),
                )
            )
        except Exception as error:
            self._log_provider_failure("one_shot_patient_summary_stream", error)
            raise ClinicalChatProviderUnavailableError("patient summary is unavailable") from error

    @staticmethod
    def _evidence(document: KnowledgeDocument) -> KnowledgeEvidence:
        return KnowledgeEvidence(
            document_id=document.document_id,
            title=document.title,
            version=document.version,
            published_on=document.published_on,
            citation_location=document.citation_location,
        )

    def _record_provider_event(
        self,
        event_type: AuditEventType,
        context: VerifiedContext,
        request_id: str,
        outcome: str,
    ) -> None:
        """질문·답변·토큰 원문 없이 AI 호출 생명주기만 감사 기록한다."""
        self._audit_sink.record(
            audit_event(
                event_type,
                actor_id=context.actor.user_id,
                organization_id=context.actor.organization_id,
                patient_id=context.patient_id,
                encounter_id=context.encounter.encounter_id,
                request_id=request_id,
                outcome=outcome,
                reason_code="openai_responses",
            )
        )

    @staticmethod
    def _log_provider_failure(operation: str, error: Exception) -> None:
        """Log a classified provider failure without prompts, EMR, or API keys."""
        reason = str(error) if isinstance(error, OpenAIProviderError) else "unexpected_provider_exception"
        logger.warning(
            "ai_provider_failed operation=%s error_type=%s reason=%s",
            operation,
            type(error).__name__,
            reason,
        )


def compose_prepared_snapshot_question(
    question: str, snapshot: Mapping[str, Any], reference_record: str | None = None
) -> str:
    """비식별 경계 후 별도 템플릿으로 준비 문맥 프롬프트를 조립한다."""
    return _compose_prepared_snapshot_question(question, deidentified_snapshot(snapshot), reference_record)
