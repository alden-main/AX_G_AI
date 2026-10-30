"""간소화된 Bridge 전용 환자 EMR 준비·임상 채팅 API.

모든 엔드포인트는 ``async``다. Provider 응답을 기다리는 동안 스레드를 점유하지 않으므로
동시 요청 수가 스레드풀 크기(기본 40)에 묶이지 않고, ``/health``도 지연되지 않는다.
도메인 예외는 아래 ``_ERROR_RESPONSES``의 예외 처리기에서 안전한 오류 코드로 바뀐다.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from datetime import datetime
import json
import logging
from time import perf_counter
from typing import Annotated, Mapping
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, StringConstraints

from ax_g_ai.adapters.main_bridge import RawEmrPayloadMapper
from ax_g_ai.errors import (
    ClinicalChatProviderUnavailableError,
    PatientContextNotReadyError,
    PatientContextUnavailableError,
    PatientMismatchError,
)
from ax_g_ai.services.clinical_chat import ClinicalChatService
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


class ClinicalChatResponseBody(BaseModel):
    answer: str
    source: str = _SOURCE
    emr_updated_at: datetime


class ErrorBody(BaseModel):
    code: str
    message: str


def create_app(
    service: ClinicalChatService | None = None,
    mapper: RawEmrPayloadMapper | None = None,
    store: PatientContextStore | None = None,
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

    _responses = {status: {"model": ErrorBody} for status in (400, 403, 404, 502)}

    @app.post("/internal/v1/patient-context", response_model=PatientContextReadyBody, responses=_responses, tags=["internal"])
    async def prepare_patient_context(body: PatientContextRequestBody) -> PatientContextReadyBody:
        # A refresh starts by invalidating the old item.  Neither malformed EMR
        # data nor a failed summary may leave stale patient context available.
        store.discard(body.patient_id)
        context = mapper.map(body.patient_id, body.emr_payload)
        summary = await require_service().summarize_prepared_snapshot(snapshot=context.data)
        store.put(body.patient_id, context)
        return PatientContextReadyBody(patient_id=body.patient_id, patient_summary=summary, emr_updated_at=context.updated_at)

    @app.post("/internal/v1/emr-summary", response_model=ClinicalChatResponseBody, responses=_responses, tags=["internal"])
    async def emr_summary(body: PatientContextRequestBody) -> ClinicalChatResponseBody:
        """Bridge가 전달한 EMR을 저장 없이 한 번 요약한다."""
        context = mapper.map(body.patient_id, body.emr_payload)
        answer = await require_service().summarize_once(snapshot=context.data)
        return ClinicalChatResponseBody(answer=answer, emr_updated_at=context.updated_at)

    @app.post("/internal/v1/clinical-chat", response_model=ClinicalChatResponseBody, responses=_responses, tags=["internal"])
    async def clinical_chat(body: ClinicalChatRequestBody) -> ClinicalChatResponseBody:
        context = store.get(body.patient_id)
        answer = await require_service().answer_prepared_snapshot(
            question=body.question,
            reference_record=body.reference_record,
            snapshot=context.data,
        )
        return ClinicalChatResponseBody(answer=answer, emr_updated_at=context.updated_at)

    @app.post("/internal/v1/clinical-chat/stream", response_model=None, responses=_responses, tags=["internal"])
    async def clinical_chat_stream(body: ClinicalChatRequestBody, request: Request) -> StreamingResponse:
        """Return the same clinical answer as SSE without exposing provider events."""
        context = store.get(body.patient_id)
        return await _stream(
            request,
            lambda: require_service().stream_prepared_snapshot(
                question=body.question,
                reference_record=body.reference_record,
                snapshot=context.data,
            ),
            {"source": _SOURCE, "emr_updated_at": context.updated_at.isoformat()},
        )

    @app.post("/internal/v1/patient-context/stream", response_model=None, responses=_responses, tags=["internal"])
    async def prepare_patient_context_stream(body: PatientContextRequestBody, request: Request) -> StreamingResponse:
        """Prepare the EMR context at once and stream the initial summary as SSE.

        Chat can use the context as soon as ``meta`` is sent.  A summary that
        fails before ``done`` discards the context, as the JSON endpoint does.
        """
        store.discard(body.patient_id)
        context = mapper.map(body.patient_id, body.emr_payload)
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
