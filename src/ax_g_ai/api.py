"""간소화된 Bridge 전용 환자 EMR 준비·임상 채팅 API.

모든 엔드포인트는 ``async``다. Provider 응답을 기다리는 동안 스레드를 점유하지 않으므로
동시 요청 수가 스레드풀 크기(기본 40)에 묶이지 않고, ``/health``도 지연되지 않는다.
도메인 예외는 아래 ``_ERROR_RESPONSES``의 예외 처리기에서 안전한 오류 코드로 바뀐다.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Sequence
from dataclasses import replace
from datetime import datetime
import json
import logging
import re
from time import perf_counter
from typing import Annotated, Mapping
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.concurrency import run_in_threadpool
from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from ax_g_ai.adapters.main_bridge import PreparedRawEmrContext, RawEmrPayloadMapper, emr_source
from ax_g_ai.errors import (
    ClinicalChatProviderUnavailableError,
    PatientContextNotReadyError,
    PatientContextUnavailableError,
    PatientMismatchError,
    RecommendationUnavailableError,
)
from ax_g_ai.services.clinical_chat import ClinicalChatService, evidence_payload
from ax_g_ai.services.clinical_evidence import EvidenceChunk
from ax_g_ai.services.clinical_recommendation import Assessment, ClinicalRecommender
from ax_g_ai.services.patient_context_store import PatientContextStore
from ax_g_ai.services.prepared_patient_summary import PreparedPatientSummary


# Use Uvicorn's error logger so application warnings are emitted by the
# container's standard logging configuration as well as Uvicorn's own logs.
logger = logging.getLogger("uvicorn.error")

_SOURCE = "EMR 데이터베이스"
_ERROR_RESPONSES: dict[type[Exception], tuple[int, str, str]] = {
    PatientContextNotReadyError: (404, "PATIENT_CONTEXT_NOT_READY", "환자 EMR 준비가 필요합니다."),
    PatientMismatchError: (403, "PATIENT_MISMATCH", "환자 정보가 일치하지 않습니다."),
    PatientContextUnavailableError: (502, "EMR_UNAVAILABLE", "환자 EMR 정보를 준비할 수 없습니다."),
    ClinicalChatProviderUnavailableError: (502, "AI_PROVIDER_UNAVAILABLE", "AI 답변을 준비할 수 없습니다."),
    RecommendationUnavailableError: (503, "RECOMMENDATION_UNAVAILABLE", "추천 진단·검사 규칙을 사용할 수 없습니다."),
}
# Chat questions that ask for diagnoses or tests also get the stage-2 rule results.
_RECOMMENDATION_INTENT = re.compile(
    r"추천|감별|의심|진단명|어떤\s*검사|검사가?\s*필요|필요한\s*검사|추가\s*검사|워크업|workup|work-up|협진|의뢰"
    r"|병용|금기|상호\s*작용|DUR|처방\s*(점검|검토)|약물\s*(점검|검토)|복용\s*약"
    r"|용량|투여량|감량|증량|mg\s*/\s*kg|신기능\s*(조정|보정)|dose|dosing", re.I,
)
_RULE_EVIDENCE_IN_CHAT = 4

NonBlankText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class PatientIdBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    patient_id: NonBlankText


class ClinicalChatRequestBody(PatientIdBody):
    question: NonBlankText
    reference_record: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=12000)] | None = None


class PatientContextRequestBody(PatientIdBody):
    emr_payload: dict[str, object]


class PatientContextReadyBody(BaseModel):
    patient_id: str
    status: str = "ready"
    patient_summary: PreparedPatientSummary
    emr_updated_at: datetime


class ClinicalChatResponseBody(BaseModel):
    answer: str
    source: str = _SOURCE
    emr_updated_at: datetime
    # 답변에 붙인 최신화 근거([근거 n] 번호와 같음). 근거가 없으면 필드를 생략한다. 검수 상태는 review_status(데이터)로만 둔다.
    evidence: list[dict[str, object]] | None = None


class ClinicalRecommendationRequestBody(PatientIdBody):
    # 의료진이 입력한 증상·소견 메모. 소견 키로 추출해 규칙에 반영하고, 추천 문장 생성에도 쓴다.
    clinical_note: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=4000)] | None = None
    # 소견 키 → 있음(true)/없음(false). 메모 추출 결과보다 우선한다. 키 목록은 GET /internal/v1/clinical-recommendations/findings.
    findings: dict[str, bool] | None = Field(default=None, max_length=200)
    question: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2000)] | None = None
    # 용량을 계산할 약(성분 ID 또는 한글·영문 성분명). 현재 처방 약은 자동 포함. 목록은 GET .../findings 의 dose_drugs.
    dose_drugs: list[Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=100)]] | None = Field(
        default=None, max_length=20,
    )


class ClinicalRecommendationResponseBody(BaseModel):
    answer: str
    source: str = _SOURCE
    emr_updated_at: datetime
    # 규칙 평가 결과(진단·검사·협진·추적 항목, 검사값 범주, 적용한 소견). 항목별 sources 포함.
    assessment: dict[str, object]
    evidence: list[dict[str, object]] | None = None


class ErrorBody(BaseModel):
    code: str
    message: str


def create_app(
    service: ClinicalChatService | None = None,
    mapper: RawEmrPayloadMapper | None = None,
    store: PatientContextStore | None = None,
    recommender: ClinicalRecommender | None = None,
) -> FastAPI:
    """새 내부 계약만 노출하는 ASGI 앱을 조립한다.

    ``service``가 없으면 AI 설정이 없는 fail-closed 앱이며, AI 호출이 필요한 요청은
    ``AI_PROVIDER_UNAVAILABLE``로 거부된다.
    """
    app = FastAPI(title="AX-G AI Clinical Chat API", version="1")
    mapper = mapper or RawEmrPayloadMapper()
    store = store or PatientContextStore()

    def require_service() -> ClinicalChatService:
        if service is None:
            raise ClinicalChatProviderUnavailableError("AI provider is not configured")
        return service

    def require_recommender() -> ClinicalRecommender:
        if recommender is None or len(recommender) == 0:
            raise RecommendationUnavailableError()
        return recommender

    def prepare(patient_id: str, payload: Mapping[str, object]) -> PreparedRawEmrContext:
        context = mapper.map(patient_id, payload)
        if recommender is None:
            return context
        try:
            rule_input = recommender.prepare(emr_source(payload))
        except Exception as error:  # rule facts are optional; never block EMR preparation
            logger.warning("clinical_rule_mapping_failed error_type=%s", type(error).__name__)
            return context
        return replace(context, rule_input=rule_input)

    def chat_grounding(question: str, context: PreparedRawEmrContext) -> tuple[tuple[EvidenceChunk, ...], dict[str, object] | None]:
        """Evidence (and, for diagnosis/test questions, rule results) for one chat question."""
        chat = require_service()
        searched = chat.find_evidence(question)
        if not searched:
            # A generic question ("추천 검사는?") has no topic words: add the recorded diagnoses.
            hint = _diagnosis_hint(context)
            if hint:
                searched = chat.find_evidence(f"{question} {hint}")
        if recommender is None or context.rule_input is None or not _RECOMMENDATION_INTENT.search(question):
            return searched, None
        assessment = recommender.assess(context.rule_input)
        alerts = (assessment.medication_alerts or {}).get("alerts")
        if not assessment.items and not assessment.lab_interpretations and not alerts and not assessment.dose_assessments:
            return searched, None
        rule_chunks = assessment.evidence[:_RULE_EVIDENCE_IN_CHAT]
        payload = _renumber(assessment, len(rule_chunks))
        return _merge(rule_chunks, searched), payload

    @app.middleware("http")
    async def request_logging(request: Request, call_next):
        """Attach a safe correlation ID and log only failed HTTP requests.

        Request bodies can contain EMR data, questions, and credentials, so
        they must never be sent to container logs.  The request ID instead
        lets Bridge and Docker logs be correlated without retaining that data.
        """
        request_id = request.headers.get("X-Request-ID", "").strip() or str(uuid4())
        request.state.request_id = request_id
        started_at = perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            logger.exception(
                "api_unhandled_exception request_id=%s method=%s path=%s",
                request_id,
                request.method,
                request.url.path,
            )
            response = _error(500, "INTERNAL_ERROR", "서버 내부 오류가 발생했습니다.")

        response.headers["X-Request-ID"] = request_id
        if response.status_code >= 400:
            logger.warning(
                "api_request_failed request_id=%s status=%s method=%s path=%s duration_ms=%d",
                request_id,
                response.status_code,
                request.method,
                request.url.path,
                (perf_counter() - started_at) * 1000,
            )
        return response

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        if request.url.path.startswith("/internal/v1/"):
            # Do not log Pydantic's ``input`` field: it can contain an entire
            # EMR payload.  Field locations and validation categories are
            # enough to identify a malformed integration request.
            failures = ",".join(
                f"{'.'.join(str(item) for item in error['loc'])}:{error['type']}"
                for error in exc.errors()
            )
            logger.warning(
                "api_validation_failed request_id=%s method=%s path=%s failures=%s",
                _request_id(request),
                request.method,
                request.url.path,
                failures,
            )
            return _error(400, "INVALID_REQUEST", "요청 형식이 올바르지 않습니다.")
        return JSONResponse(status_code=422, content={"detail": exc.errors()})

    for error_type, (status_code, code, message) in _ERROR_RESPONSES.items():
        app.add_exception_handler(error_type, _domain_error_handler(status_code, code, message))

    @app.get("/health", tags=["operations"])
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    _responses = {status: {"model": ErrorBody} for status in (400, 403, 404, 502, 503)}

    @app.post("/internal/v1/patient-context", response_model=PatientContextReadyBody, responses=_responses, tags=["internal"])
    async def prepare_patient_context(body: PatientContextRequestBody) -> PatientContextReadyBody:
        # A refresh starts by invalidating the old item.  Neither malformed EMR
        # data nor a failed summary may leave stale patient context available.
        store.discard(body.patient_id)
        context = prepare(body.patient_id, body.emr_payload)
        summary = await require_service().summarize_prepared_snapshot(snapshot=context.data)
        store.put(body.patient_id, context)
        return PatientContextReadyBody(patient_id=body.patient_id, patient_summary=summary, emr_updated_at=context.updated_at)

    @app.post("/internal/v1/emr-summary", response_model=ClinicalChatResponseBody, responses=_responses, tags=["internal"])
    async def emr_summary(body: PatientContextRequestBody) -> ClinicalChatResponseBody:
        """Bridge가 전달한 EMR을 저장 없이 한 번 요약한다."""
        context = mapper.map(body.patient_id, body.emr_payload)
        answer = await require_service().summarize_once(snapshot=context.data)
        return ClinicalChatResponseBody(answer=answer, emr_updated_at=context.updated_at)

    @app.post("/internal/v1/clinical-chat", response_model=ClinicalChatResponseBody, response_model_exclude_none=True, responses=_responses, tags=["internal"])
    async def clinical_chat(body: ClinicalChatRequestBody) -> ClinicalChatResponseBody:
        context = store.get(body.patient_id)
        service = require_service()
        # Retrieval may call the embedding server and database; keep it off the event loop.
        evidence, assessment = await run_in_threadpool(chat_grounding, body.question, context)
        answer = await service.answer_prepared_snapshot(
            question=body.question,
            reference_record=body.reference_record,
            snapshot=context.data,
            evidence=evidence,
            **({"assessment": assessment} if assessment else {}),
        )
        return ClinicalChatResponseBody(answer=answer, emr_updated_at=context.updated_at, evidence=evidence_payload(evidence) or None)

    @app.post("/internal/v1/clinical-chat/stream", response_model=None, responses=_responses, tags=["internal"])
    async def clinical_chat_stream(body: ClinicalChatRequestBody, request: Request) -> StreamingResponse:
        """Return the same clinical answer as SSE without exposing provider events."""
        context = store.get(body.patient_id)
        service = require_service()
        evidence, assessment = await run_in_threadpool(chat_grounding, body.question, context)
        return await _stream(
            request,
            lambda: service.stream_prepared_snapshot(
                question=body.question,
                reference_record=body.reference_record,
                snapshot=context.data,
                evidence=evidence,
                **({"assessment": assessment} if assessment else {}),
            ),
            {"source": _SOURCE, "emr_updated_at": context.updated_at.isoformat(),
             **({"evidence": evidence_payload(evidence)} if evidence else {})},
        )

    async def recommendation_inputs(
        body: ClinicalRecommendationRequestBody,
    ) -> tuple[PreparedRawEmrContext, Assessment, tuple[EvidenceChunk, ...], dict[str, object]]:
        context = store.get(body.patient_id)
        rules = require_recommender()
        service = require_service()
        if context.rule_input is None:
            # Prepared before rules were available, or EMR mapping failed: evaluate with no EMR facts.
            context = replace(context, rule_input=rules.prepare({}))
        findings: dict[str, bool] = {}
        if body.clinical_note:
            findings.update(await service.extract_findings(note=body.clinical_note, vocabulary=rules.finding_vocabulary))
        findings.update(body.findings or {})  # explicit clinician input wins over extraction
        assessment = rules.assess(context.rule_input, findings, body.dose_drugs or ())  # type: ignore[arg-type]
        query = " ".join(part for part in (body.question, body.clinical_note, _diagnosis_hint(context)) if part)
        searched = await run_in_threadpool(service.find_evidence, query) if query else ()
        evidence = _merge(assessment.evidence, searched)
        logger.info(
            "clinical_recommendation_assessed items=%d met=%d findings=%d unknown_findings=%d evidence=%d unreviewed=%s doses=%d",
            len(assessment.items), sum(1 for item in assessment.items if item.get("status") == "met"),
            len(assessment.findings_used), len(assessment.unknown_findings), len(evidence), assessment.unreviewed,
            len(assessment.dose_assessments or ()),
        )
        return context, assessment, evidence, assessment.as_dict()

    @app.get("/internal/v1/clinical-recommendations/findings", tags=["internal"])
    async def recommendation_findings() -> dict[str, object]:
        """규칙이 사용하는 소견·사건 키 목록(``findings`` 입력용)과 용량 규칙이 있는 약(``dose_drugs`` 입력용)."""
        rules = require_recommender()
        return {"findings": list(rules.finding_vocabulary), "rules": len(rules), "unreviewed_rules": rules.unreviewed_count,
                "dose_drugs": list(rules.dose_drugs)}

    @app.post("/internal/v1/clinical-recommendations", response_model=ClinicalRecommendationResponseBody, response_model_exclude_none=True, responses=_responses, tags=["internal"])
    async def clinical_recommendations(body: ClinicalRecommendationRequestBody) -> ClinicalRecommendationResponseBody:
        """말뭉치 규칙 평가 + 근거 인용으로 의심 진단·추천 검사·협진 의견을 만든다."""
        context, assessment, evidence, assessment_payload = await recommendation_inputs(body)
        answer = await require_service().recommend(
            snapshot=context.data, assessment=assessment_payload, evidence=evidence,
            clinical_note=body.clinical_note, question=body.question,
        )
        return ClinicalRecommendationResponseBody(
            answer=answer, emr_updated_at=context.updated_at, assessment=assessment_payload,
            evidence=evidence_payload(evidence) or None,
        )

    @app.post("/internal/v1/clinical-recommendations/stream", response_model=None, responses=_responses, tags=["internal"])
    async def clinical_recommendations_stream(body: ClinicalRecommendationRequestBody, request: Request) -> StreamingResponse:
        """같은 추천 의견을 SSE로 보낸다. 규칙 평가 결과·근거는 ``meta``에 먼저 담는다."""
        context, assessment, evidence, assessment_payload = await recommendation_inputs(body)
        return await _stream(
            request,
            lambda: require_service().stream_recommendation(
                snapshot=context.data, assessment=assessment_payload, evidence=evidence,
                clinical_note=body.clinical_note, question=body.question,
            ),
            {"source": _SOURCE, "emr_updated_at": context.updated_at.isoformat(), "assessment": assessment_payload,
             **({"evidence": evidence_payload(evidence)} if evidence else {})},
        )

    @app.post("/internal/v1/patient-context/stream", response_model=None, responses=_responses, tags=["internal"])
    async def prepare_patient_context_stream(body: PatientContextRequestBody, request: Request) -> StreamingResponse:
        """Prepare the EMR context at once and stream the initial summary as SSE.

        Chat can use the context as soon as ``meta`` is sent.  A summary that
        fails before ``done`` discards the context, as the JSON endpoint does.
        """
        store.discard(body.patient_id)
        context = prepare(body.patient_id, body.emr_payload)
        store.put(body.patient_id, context)
        return await _stream(
            request,
            lambda: require_service().stream_summary_once(snapshot=context.data),
            {"patient_id": body.patient_id, "status": "ready", "source": _SOURCE, "emr_updated_at": context.updated_at.isoformat()},
            on_failure=lambda: store.discard(body.patient_id),
        )

    @app.post("/internal/v1/emr-summary/stream", response_model=None, responses=_responses, tags=["internal"])
    async def emr_summary_stream(body: PatientContextRequestBody, request: Request) -> StreamingResponse:
        """Stream a one-shot EMR summary without storing the chat context."""
        context = mapper.map(body.patient_id, body.emr_payload)
        return await _stream(
            request,
            lambda: require_service().stream_summary_once(snapshot=context.data),
            {"source": _SOURCE, "emr_updated_at": context.updated_at.isoformat()},
        )

    return app


def _diagnosis_hint(context: PreparedRawEmrContext) -> str:
    diagnoses = context.data.get("diagnoses")
    names = [item.get("name_ko") for item in diagnoses if isinstance(item, Mapping)] if isinstance(diagnoses, list) else []
    return " ".join(name for name in names if isinstance(name, str))[:300]


def _merge(first: Sequence[EvidenceChunk], second: Sequence[EvidenceChunk]) -> tuple[EvidenceChunk, ...]:
    """Concatenate keeping ``first``'s numbering (rule evidence refs point into it)."""
    seen = {chunk.chunk_id for chunk in first}
    return (*first, *(chunk for chunk in second if chunk.chunk_id not in seen))


def _renumber(assessment: Assessment, kept: int) -> dict[str, object]:
    """Assessment payload whose evidence refs only point at the first ``kept`` passages."""
    payload = assessment.as_dict()
    for group in ("items", "lab_interpretations", "dose_assessments"):
        for item in payload.get(group) or ():  # type: ignore[union-attr]
            item["evidence_refs"] = [ref for ref in item.get("evidence_refs", ()) if ref <= kept]
    return payload


async def _stream(
    request: Request,
    start: Callable[[], AsyncIterator[str]],
    meta: Mapping[str, object],
    *,
    on_failure: Callable[[], None] | None = None,
) -> StreamingResponse:
    """Serve provider deltas as SSE ``meta``/``delta``/``done`` events.

    The provider is primed before committing HTTP 200, so a connection or
    first-token failure still returns the normal JSON 502.
    """
    try:
        deltas = start()
        first_delta = await anext(deltas)
    except (ClinicalChatProviderUnavailableError, StopAsyncIteration) as error:
        if on_failure is not None:
            on_failure()
        raise ClinicalChatProviderUnavailableError("stream did not start") from error

    async def events() -> AsyncIterator[str]:
        yield _sse("meta", meta)
        yield _sse("delta", {"text": first_delta})
        try:
            async for delta in deltas:
                yield _sse("delta", {"text": delta})
        except ClinicalChatProviderUnavailableError:
            # HTTP status is already 200 after the first event.  Do not
            # disclose provider details, prompts, or EMR in an SSE error.
            if on_failure is not None:
                on_failure()
            logger.warning(
                "api_stream_failed request_id=%s path=%s code=AI_PROVIDER_UNAVAILABLE stage=streaming",
                _request_id(request),
                request.url.path,
            )
            yield _sse("error", {"code": "AI_PROVIDER_UNAVAILABLE", "message": "AI 답변을 준비하는 중 연결이 종료되었습니다."})
        else:
            yield _sse("done", {})
        finally:
            await deltas.aclose()

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _domain_error_handler(status_code: int, code: str, message: str):
    async def handle(request: Request, exc: Exception) -> JSONResponse:
        failure = f" failure={exc.failure}" if isinstance(exc, PatientContextUnavailableError) else ""
        logger.warning(
            "api_request_rejected request_id=%s path=%s code=%s%s",
            _request_id(request),
            request.url.path,
            code,
            failure,
        )
        return _error(status_code, code, message)

    return handle


def _request_id(request: Request) -> str:
    return getattr(request.state, "request_id", "unavailable")


def _error(status_code: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=status_code, content=ErrorBody(code=code, message=message).model_dump())


def _sse(event: str, data: Mapping[str, object]) -> str:
    """Serialize a single trusted SSE event without leaking provider payloads."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False, separators=(',', ':'))}\n\n"


def __getattr__(name: str) -> FastAPI:
    """Build the ``ax_g_ai.api:app`` entry point on first access.

    ``runtime`` imports ``create_app`` from this module, so importing it at
    module load would be circular; tests also avoid a configuration read.
    """
    if name != "app":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from ax_g_ai.runtime import create_runtime_app

    application = create_runtime_app()
    globals()["app"] = application
    return application
