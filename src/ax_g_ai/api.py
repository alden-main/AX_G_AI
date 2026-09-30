"""간소화된 Bridge 전용 환자 EMR 준비·임상 채팅 API."""

from __future__ import annotations

from datetime import datetime
import logging
from time import perf_counter
from typing import Mapping, Protocol
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

from ax_g_ai.services.clinical_chat import ClinicalChatProviderUnavailableError, ClinicalChatService
from ax_g_ai.services.prepared_patient_summary import PreparedPatientSummary


# Use Uvicorn's error logger so application warnings are emitted by the
# container's standard logging configuration as well as Uvicorn's own logs.
logger = logging.getLogger("uvicorn.error")


class PatientContextNotReadyError(LookupError):
    """채팅 전에 환자 Snapshot이 준비되지 않았을 때 발생한다."""


class PatientContextUnavailableError(RuntimeError):
    """EMR Snapshot 조회 또는 검증이 실패했을 때 발생한다."""


class PatientMismatchError(PermissionError):
    """요청 환자와 전달된 Snapshot의 환자가 다를 때 발생한다."""


class PatientContextStore(Protocol):
    def prepare(
        self, patient_id: str, emr_payload: Mapping[str, object]
    ) -> tuple[PreparedPatientSummary, datetime]: ...
    def get(self, patient_id: str) -> tuple[Mapping[str, object], datetime]: ...


class UnconfiguredPatientContextStore:
    def prepare(
        self, patient_id: str, emr_payload: Mapping[str, object]
    ) -> tuple[PreparedPatientSummary, datetime]:
        raise PatientContextUnavailableError()

    def get(self, patient_id: str) -> tuple[Mapping[str, object], datetime]:
        raise PatientContextNotReadyError()


class PatientIdBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    patient_id: str = Field(min_length=1)

    @field_validator("patient_id")
    @classmethod
    def non_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value


class ClinicalChatRequestBody(PatientIdBody):
    question: str = Field(min_length=1)

    @field_validator("question")
    @classmethod
    def non_blank_question(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value


class PatientContextRequestBody(PatientIdBody):
    emr_payload: dict[str, object]


class PatientSummarySectionBody(BaseModel):
    label: str
    content: str


class PatientSummaryBody(BaseModel):
    message: str
    sections: list[PatientSummarySectionBody]
    guidance: str


class PatientContextReadyBody(BaseModel):
    patient_id: str
    status: str = "ready"
    patient_summary: PatientSummaryBody
    emr_updated_at: datetime


class ClinicalChatResponseBody(BaseModel):
    answer: str
    source: str = "EMR 데이터베이스"
    emr_updated_at: datetime


class ErrorBody(BaseModel):
    code: str
    message: str


def create_app(service: ClinicalChatService | None = None, patient_context_store: PatientContextStore | None = None) -> FastAPI:
    """새 내부 계약만 노출하는 ASGI 앱을 조립한다."""
    app = FastAPI(title="AX-G AI Clinical Chat API", version="1")
    store = patient_context_store or UnconfiguredPatientContextStore()

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
            response = JSONResponse(
                status_code=500,
                content=ErrorBody(code="INTERNAL_ERROR", message="서버 내부 오류가 발생했습니다.").model_dump(),
            )

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
                getattr(request.state, "request_id", "unavailable"),
                request.method,
                request.url.path,
                failures,
            )
            return _error(400, "INVALID_REQUEST", "요청 형식이 올바르지 않습니다.")
        return JSONResponse(status_code=422, content={"detail": exc.errors()})

    @app.get("/health", tags=["operations"])
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/internal/v1/patient-context", response_model=PatientContextReadyBody, responses={400: {"model": ErrorBody}, 403: {"model": ErrorBody}, 502: {"model": ErrorBody}}, tags=["internal"])
    def prepare_patient_context(body: PatientContextRequestBody, request: Request) -> PatientContextReadyBody | JSONResponse:
        try:
            summary, updated_at = store.prepare(body.patient_id, body.emr_payload)
        except PatientMismatchError:
            logger.warning("api_patient_context_failed request_id=%s code=PATIENT_MISMATCH", request.state.request_id)
            return _error(403, "PATIENT_MISMATCH", "환자 정보가 일치하지 않습니다.")
        except PatientContextUnavailableError as error:
            logger.warning(
                "api_patient_context_failed request_id=%s code=EMR_UNAVAILABLE failure=%s",
                request.state.request_id,
                _safe_context_failure(error),
            )
            return _error(502, "EMR_UNAVAILABLE", "환자 EMR 정보를 준비할 수 없습니다.")
        except ClinicalChatProviderUnavailableError:
            logger.warning("api_patient_context_failed request_id=%s code=AI_PROVIDER_UNAVAILABLE", request.state.request_id)
            return _error(502, "AI_PROVIDER_UNAVAILABLE", "AI 답변을 준비할 수 없습니다.")
        return PatientContextReadyBody(patient_id=body.patient_id, patient_summary=summary.as_dict(), emr_updated_at=updated_at)

    @app.post("/internal/v1/clinical-chat", response_model=ClinicalChatResponseBody, responses={400: {"model": ErrorBody}, 404: {"model": ErrorBody}, 502: {"model": ErrorBody}}, tags=["internal"])
    def clinical_chat(body: ClinicalChatRequestBody, request: Request) -> ClinicalChatResponseBody | JSONResponse:
        try:
            snapshot, updated_at = store.get(body.patient_id)
            if service is None:
                raise ClinicalChatProviderUnavailableError()
            answer = service.answer_prepared_snapshot(patient_id=body.patient_id, question=body.question, snapshot=snapshot)
        except PatientContextNotReadyError:
            logger.warning("api_clinical_chat_failed request_id=%s code=PATIENT_CONTEXT_NOT_READY", request.state.request_id)
            return _error(404, "PATIENT_CONTEXT_NOT_READY", "환자 EMR 준비가 필요합니다.")
        except ClinicalChatProviderUnavailableError:
            logger.warning("api_clinical_chat_failed request_id=%s code=AI_PROVIDER_UNAVAILABLE", request.state.request_id)
            return _error(502, "AI_PROVIDER_UNAVAILABLE", "AI 답변을 준비할 수 없습니다.")
        return ClinicalChatResponseBody(answer=answer, emr_updated_at=updated_at)

    return app


def _error(status_code: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=status_code, content=ErrorBody(code=code, message=message).model_dump())


def _safe_context_failure(error: PatientContextUnavailableError) -> str:
    """Return only mapper-produced schema diagnostics, never request content."""
    safe_failures = {
        "raw EMR payload has no patient object": "emr_payload.patient:missing_or_not_object",
        "raw EMR payload requires patient.patientId": "emr_payload.patient.patientId:missing_or_blank",
    }
    return safe_failures.get(str(error), "emr_payload:invalid")


from ax_g_ai.runtime import create_runtime_app

app = create_runtime_app()
