"""간소화된 Bridge 전용 환자 EMR 준비·임상 채팅 API.

모든 엔드포인트는 ``async``다. Provider 응답을 기다리는 동안 스레드를 점유하지 않으므로
동시 요청 수가 스레드풀 크기(기본 40)에 묶이지 않고, ``/health``도 지연되지 않는다.
도메인 예외는 아래 ``_ERROR_RESPONSES``의 예외 처리기에서 안전한 오류 코드로 바뀐다.
"""

from __future__ import annotations

import inspect
import json
import logging
from collections.abc import AsyncGenerator, AsyncIterator, Callable, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime
from time import perf_counter
from typing import Annotated, Any
from uuid import uuid4

from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, SerializerFunctionWrapHandler, StringConstraints, model_serializer
from starlette.concurrency import run_in_threadpool

from ax_g_ai.adapters.main_bridge import RawEmrPayloadMapper
from ax_g_ai.errors import (
    ClinicalChatProviderUnavailableError,
    PatientContextNotReadyError,
    PatientContextUnavailableError,
    PatientMismatchError,
    RecommendationUnavailableError,
)
from ax_g_ai.services.clinical_chat import ClinicalChatService, evidence_payload
from ax_g_ai.services.clinical_grounding import chat_grounding, recommendation_grounding, require_recommender, with_rule_input
from ax_g_ai.services.clinical_recommendation import ClinicalRecommender
from ax_g_ai.services.clinical_recommendation.schema import AssessmentPayload, Evidence
from ax_g_ai.services.corpus_snapshot import CorpusSnapshot, CorpusSource, StaticCorpus
from ax_g_ai.services.patient_context_store import ContextStore, PatientContextStore, PreparedRawEmrContext
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


class _OmitsMissingEvidence(BaseModel):
    """Leaves out ``evidence`` when there is none. Only that key: nulls elsewhere (inside ``assessment`` or an
    evidence passage) are part of the contract, which ``response_model_exclude_none`` would drop."""

    @model_serializer(mode="wrap")
    def _omit_missing_evidence(self, handler: SerializerFunctionWrapHandler) -> dict[str, object]:
        data = handler(self)
        if data.get("evidence") is None:
            data.pop("evidence", None)
        return data


class ClinicalChatResponseBody(_OmitsMissingEvidence):
    answer: str
    source: str = _SOURCE
    emr_updated_at: datetime
    # 답변에 붙인 최신화 근거([근거 n] 번호와 같음). 근거가 없으면 필드를 생략한다. 검수 상태는 review_status(데이터)로만 둔다.
    evidence: list[Evidence] | None = None


class EmrSummaryResponseBody(BaseModel):
    """One-shot summary: never cites evidence, and keeps ``evidence: null`` (the published contract)."""

    answer: str
    source: str = _SOURCE
    emr_updated_at: datetime
    evidence: None = None


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


class ClinicalRecommendationResponseBody(_OmitsMissingEvidence):
    answer: str
    source: str = _SOURCE
    emr_updated_at: datetime
    # 규칙 평가 결과(진단·검사·협진·추적 항목, 검사값 범주, DUR, 용량). 구조는 docs/assessment_schema.md.
    # 해당 없는 키는 생략하고, 값이 없는 키는 null로 둔다(SSE meta.assessment와 같은 JSON).
    assessment: AssessmentPayload
    evidence: list[Evidence] | None = None


class ErrorBody(BaseModel):
    code: str
    message: str


@dataclass(frozen=True)
class AppComponents:
    """Everything the routes use, held on ``app.state.components`` and injected with ``Depends``."""

    service: ClinicalChatService | None
    mapper: RawEmrPayloadMapper
    store: ContextStore
    corpus: CorpusSource


def app_components(request: Request) -> AppComponents:
    return request.app.state.components


def chat_service(components: Annotated[AppComponents, Depends(app_components)]) -> ClinicalChatService:
    if components.service is None:
        raise ClinicalChatProviderUnavailableError("AI provider is not configured")
    return components.service


def corpus_snapshot(components: Annotated[AppComponents, Depends(app_components)]) -> CorpusSnapshot:
    """Read once per request: evidence, rules and DUR of one release for the whole request."""
    return components.corpus.current


Components = Annotated[AppComponents, Depends(app_components)]
Service = Annotated[ClinicalChatService, Depends(chat_service)]
Snapshot = Annotated[CorpusSnapshot, Depends(corpus_snapshot)]


def create_app(
    service: ClinicalChatService | None = None,
    mapper: RawEmrPayloadMapper | None = None,
    store: ContextStore | None = None,
    recommender: ClinicalRecommender | None = None,
    *,
    corpus: CorpusSource | None = None,
    on_shutdown: Sequence[Callable[[], object]] = (),
) -> FastAPI:
    """새 내부 계약만 노출하는 ASGI 앱을 조립한다.

    ``service``가 없으면 AI 설정이 없는 fail-closed 앱이며, AI 호출이 필요한 요청은
    ``AI_PROVIDER_UNAVAILABLE``로 거부된다. ``corpus``는 요청마다 읽는 말뭉치 스냅샷(근거·규칙·DUR)이며,
    없으면 ``recommender``와 ``service``의 근거 검색기로 고정 스냅샷을 만든다. ``on_shutdown``은 종료 시
    닫을 자원(DB 풀 등)이다.
    """

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        yield
        for close in (*on_shutdown, components.store.close):
            try:
                result = close()
                if inspect.isawaitable(result):
                    await result
            except Exception as error:  # shutdown must continue past one failing resource
                logger.warning("resource_close_failed error_type=%s", type(error).__name__)

    # One schema per model in /docs: the payload models customise serialization (set keys only), which would
    # otherwise turn their response schemas into a bare object.
    app = FastAPI(title="AX-G AI Clinical Chat API", version="1", lifespan=lifespan, separate_input_output_schemas=False)
    components = app.state.components = AppComponents(
        service, mapper or RawEmrPayloadMapper(), store or PatientContextStore(),
        corpus or StaticCorpus(CorpusSnapshot(recommender=recommender)),
    )

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

    _responses: dict[int | str, dict[str, Any]] = {status: {"model": ErrorBody} for status in (400, 403, 404, 502, 503)}

    # Each use case has a JSON endpoint and an SSE ``/stream`` twin. Both prepare the same ``_Call`` (provider
    # arguments and the response fields besides ``answer``); the SSE ``meta`` event carries those same fields.

    @app.post("/internal/v1/patient-context", response_model=PatientContextReadyBody, responses=_responses, tags=["internal"])
    async def prepare_patient_context(
        body: PatientContextRequestBody, components: Components, service: Service, snapshot: Snapshot,
    ) -> PatientContextReadyBody:
        # A refresh starts by invalidating the old item.  Neither malformed EMR
        # data nor a failed summary may leave stale patient context available.
        context = await _prepare_context(body, components, snapshot)
        summary = await service.summarize_prepared_snapshot(snapshot=context.data)
        await components.store.put(body.patient_id, context)
        return PatientContextReadyBody(patient_id=body.patient_id, patient_summary=summary, emr_updated_at=context.updated_at)

    @app.post("/internal/v1/patient-context/stream", response_model=None, responses=_responses, tags=["internal"])
    async def prepare_patient_context_stream(
        body: PatientContextRequestBody, request: Request, components: Components, service: Service, snapshot: Snapshot,
    ) -> StreamingResponse:
        """Prepare the EMR context at once and stream the initial summary as SSE.

        Chat can use the context as soon as ``meta`` is sent.  A summary that
        fails before ``done`` discards the context, as the JSON endpoint does.
        """
        store = components.store
        context = await _prepare_context(body, components, snapshot)
        await store.put(body.patient_id, context)
        return await _stream(
            request,
            lambda: service.stream_summary_once(snapshot=context.data),
            {"patient_id": body.patient_id, "status": "ready", "source": _SOURCE, "emr_updated_at": context.updated_at.isoformat()},
            on_failure=lambda: store.discard(body.patient_id),
        )

    @app.post("/internal/v1/emr-summary", response_model=EmrSummaryResponseBody, responses=_responses, tags=["internal"])
    async def emr_summary(body: PatientContextRequestBody, components: Components, service: Service) -> EmrSummaryResponseBody:
        """Bridge가 전달한 EMR을 저장 없이 한 번 요약한다."""
        call = _summary_call(body, components)
        return EmrSummaryResponseBody(answer=await service.summarize_once(**call.arguments), **call.fields)

    @app.post("/internal/v1/emr-summary/stream", response_model=None, responses=_responses, tags=["internal"])
    async def emr_summary_stream(
        body: PatientContextRequestBody, request: Request, components: Components, service: Service,
    ) -> StreamingResponse:
        """Stream a one-shot EMR summary without storing the chat context."""
        call = _summary_call(body, components)
        return await _stream(request, lambda: service.stream_summary_once(**call.arguments), call.meta)

    @app.post("/internal/v1/clinical-chat", response_model=ClinicalChatResponseBody, responses=_responses, tags=["internal"])
    async def clinical_chat(
        body: ClinicalChatRequestBody, components: Components, service: Service, snapshot: Snapshot,
    ) -> ClinicalChatResponseBody:
        call = await _chat_call(body, components, service, snapshot)
        return ClinicalChatResponseBody(answer=await service.answer_prepared_snapshot(**call.arguments), **call.fields)

    @app.post("/internal/v1/clinical-chat/stream", response_model=None, responses=_responses, tags=["internal"])
    async def clinical_chat_stream(
        body: ClinicalChatRequestBody, request: Request, components: Components, service: Service, snapshot: Snapshot,
    ) -> StreamingResponse:
        """Return the same clinical answer as SSE without exposing provider events."""
        call = await _chat_call(body, components, service, snapshot)
        return await _stream(request, lambda: service.stream_prepared_snapshot(**call.arguments), call.meta)

    @app.get("/internal/v1/clinical-recommendations/findings", tags=["internal"])
    async def recommendation_findings(snapshot: Snapshot) -> dict[str, object]:
        """규칙이 사용하는 소견·사건 키 목록(``findings`` 입력용)과 용량 규칙이 있는 약(``dose_drugs`` 입력용)."""
        rules = require_recommender(snapshot)
        return {"findings": list(rules.finding_vocabulary), "rules": len(rules), "unreviewed_rules": rules.unreviewed_count,
                "dose_drugs": list(rules.dose_drugs)}

    @app.post("/internal/v1/clinical-recommendations", response_model=ClinicalRecommendationResponseBody, responses=_responses, tags=["internal"])
    async def clinical_recommendations(
        body: ClinicalRecommendationRequestBody, components: Components, service: Service, snapshot: Snapshot,
    ) -> ClinicalRecommendationResponseBody:
        """말뭉치 규칙 평가 + 근거 인용으로 의심 진단·추천 검사·협진 의견을 만든다."""
        call = await _recommendation_call(body, components, service, snapshot)
        return ClinicalRecommendationResponseBody(answer=await service.recommend(**call.arguments), **call.fields)

    @app.post("/internal/v1/clinical-recommendations/stream", response_model=None, responses=_responses, tags=["internal"])
    async def clinical_recommendations_stream(
        body: ClinicalRecommendationRequestBody, request: Request, components: Components, service: Service, snapshot: Snapshot,
    ) -> StreamingResponse:
        """같은 추천 의견을 SSE로 보낸다. 규칙 평가 결과·근거는 ``meta``에 먼저 담는다."""
        call = await _recommendation_call(body, components, service, snapshot)
        return await _stream(request, lambda: service.stream_recommendation(**call.arguments), call.meta)

    return app


@dataclass(frozen=True)
class _Call:
    """One prepared use case: provider ``arguments`` and the response ``fields`` besides ``answer``."""

    arguments: dict[str, Any]
    fields: dict[str, Any]

    @property
    def meta(self) -> dict[str, object]:
        """The SSE ``meta`` event: the same fields, unset ones left out and times in ISO format."""
        return {key: value.isoformat() if isinstance(value, datetime) else value
                for key, value in self.fields.items() if value is not None}


async def _prepare_context(
    body: PatientContextRequestBody, components: AppComponents, snapshot: CorpusSnapshot,
) -> PreparedRawEmrContext:
    """Discard the stored context first, then map the EMR (with this snapshot's rule facts)."""
    await components.store.discard(body.patient_id)
    return with_rule_input(components.mapper.map(body.patient_id, body.emr_payload), snapshot)


def _summary_call(body: PatientContextRequestBody, components: AppComponents) -> _Call:
    context = components.mapper.map(body.patient_id, body.emr_payload)
    return _Call({"snapshot": context.data}, {"source": _SOURCE, "emr_updated_at": context.updated_at})


async def _chat_call(
    body: ClinicalChatRequestBody, components: AppComponents, service: ClinicalChatService, snapshot: CorpusSnapshot,
) -> _Call:
    context = with_rule_input(await components.store.get(body.patient_id), snapshot)
    # Retrieval may call the embedding server and database; keep it off the event loop.
    grounding = await run_in_threadpool(chat_grounding, body.question, context, service, snapshot)
    return _Call(
        {"question": body.question, "reference_record": body.reference_record, "snapshot": context.data,
         "evidence": grounding.evidence, "assessment": grounding.assessment},
        {"source": _SOURCE, "emr_updated_at": context.updated_at, "evidence": evidence_payload(grounding.evidence) or None},
    )


async def _recommendation_call(
    body: ClinicalRecommendationRequestBody, components: AppComponents, service: ClinicalChatService, snapshot: CorpusSnapshot,
) -> _Call:
    context, grounding = await recommendation_grounding(
        await components.store.get(body.patient_id), service, snapshot,
        clinical_note=body.clinical_note, question=body.question, findings=body.findings, dose_drugs=body.dose_drugs,
    )
    return _Call(
        {"snapshot": context.data, "assessment": grounding.assessment, "evidence": grounding.evidence,
         "clinical_note": body.clinical_note, "question": body.question},
        {"source": _SOURCE, "emr_updated_at": context.updated_at, "assessment": grounding.assessment,
         "evidence": evidence_payload(grounding.evidence) or None},
    )


async def _stream(
    request: Request,
    start: Callable[[], AsyncIterator[str]],
    meta: Mapping[str, object],
    *,
    on_failure: Callable[[], object] | None = None,
) -> StreamingResponse:
    """Serve provider deltas as SSE ``meta``/``delta``/``done`` events.

    The provider is primed before committing HTTP 200, so a connection or
    first-token failure still returns the normal JSON 502.
    """
    try:
        deltas = start()
        first_delta = await anext(deltas)
    except (ClinicalChatProviderUnavailableError, StopAsyncIteration) as error:
        await _call(on_failure)
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
            await _call(on_failure)
            logger.warning(
                "api_stream_failed request_id=%s path=%s code=AI_PROVIDER_UNAVAILABLE stage=streaming",
                _request_id(request),
                request.url.path,
            )
            yield _sse("error", {"code": "AI_PROVIDER_UNAVAILABLE", "message": "AI 답변을 준비하는 중 연결이 종료되었습니다."})
        else:
            yield _sse("done", {})
        finally:
            if isinstance(deltas, AsyncGenerator):  # the service methods are async generators
                await deltas.aclose()

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


async def _call(callback: Callable[[], object] | None) -> None:
    """Run a sync or async callback (e.g. discarding the context in a remote store)."""
    if callback is not None:
        result = callback()
        if inspect.isawaitable(result):
            await result


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
