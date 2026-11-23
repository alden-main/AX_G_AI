"""간소화된 Bridge 전용 환자 EMR 준비·임상 채팅 API.

설계상 위치: HTTP 계층. ``runtime.create_runtime_app``(운영)과 테스트가 ``create_app``으로 앱을 만들고, Bridge 서버가
``/internal/v1/...`` 엔드포인트를 호출한다. 요청·응답 모델과 예외 → 오류 코드 변환도 이 파일에 있다.

역할: 환자 문맥 준비, EMR 1회 요약, 임상 채팅, 추천 의견(각각 JSON과 SSE ``/stream``)과 ``/health``·``/ready``를
제공한다. 모든 엔드포인트는 ``async``다. Provider 응답을 기다리는 동안 스레드를 점유하지 않으므로
동시 요청 수가 스레드풀 크기(기본 40)에 묶이지 않고, ``/health``도 지연되지 않는다.
도메인 예외는 아래 ``_ERROR_RESPONSES``의 예외 처리기에서 안전한 오류 코드로 바뀐다.
"""

from __future__ import annotations

import inspect
import json
import logging
from collections.abc import AsyncGenerator, AsyncIterator, Callable, Mapping, Sequence, Sized
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime
from time import perf_counter
from typing import Annotated, Any, Literal
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
from ax_g_ai.services.clinical_recommendation.schema import AssessmentPayload, Evidence
from ax_g_ai.services.corpus_snapshot import CorpusSnapshot, CorpusSource, StaticCorpus
from ax_g_ai.services.patient_context_store import ContextStore, PatientContextStore, PreparedRawEmrContext
from ax_g_ai.services.prepared_patient_summary import PreparedPatientSummary

# Uvicorn의 error 로거를 써서 애플리케이션 경고도 Uvicorn 로그와 같은 컨테이너 표준 로깅 설정으로 나가게 한다.
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
    """환자 ID만 가진 요청 본문의 공통 기반.

    역할: 빈 값이 아닌 ``patient_id``를 요구하고 정의되지 않은 필드를 거부한다(``extra="forbid"``).
    """

    model_config = ConfigDict(extra="forbid")
    patient_id: NonBlankText


class ClinicalChatRequestBody(PatientIdBody):
    """임상 채팅 요청 본문.

    역할: 준비된 환자 문맥에 대한 질문과 선택적 참고 기록(최대 12000자)을 받는다.
    """

    question: NonBlankText
    reference_record: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=12000)] | None = None


class PatientContextRequestBody(PatientIdBody):
    """환자 문맥 준비·EMR 요약 요청 본문.

    역할: Bridge가 전달한 EMR 원본 payload를 받는다(검증·변환은 ``RawEmrPayloadMapper``가 한다).
    """

    emr_payload: dict[str, object]


class PatientContextReadyBody(BaseModel):
    """환자 문맥 준비 완료 응답.

    역할: 준비 상태, 초기 환자 요약, EMR 갱신 시각을 돌려준다.
    """

    patient_id: str
    status: str = "ready"
    patient_summary: PreparedPatientSummary
    emr_updated_at: datetime


class _OmitsMissingEvidence(BaseModel):
    """근거가 없으면 ``evidence`` 키를 응답에서 뺀다.

    역할: 그 키만 뺀다. 다른 곳(``assessment`` 안이나 근거 문단)의 null은 계약의 일부인데,
    ``response_model_exclude_none``을 쓰면 그것까지 빠지기 때문이다.
    """

    @model_serializer(mode="wrap")
    def _omit_missing_evidence(self, handler: SerializerFunctionWrapHandler) -> dict[str, object]:
        """기본 직렬화 결과에서 값이 ``None``인 ``evidence`` 키만 제거한다."""
        data = handler(self)
        if data.get("evidence") is None:
            data.pop("evidence", None)
        return data


class ClinicalChatResponseBody(_OmitsMissingEvidence):
    """임상 채팅 응답.

    역할: 답변, 출처, EMR 갱신 시각과 답변에 인용한 근거(없으면 생략)를 돌려준다.
    """

    answer: str
    source: str = _SOURCE
    emr_updated_at: datetime
    # 답변에 붙인 최신화 근거([근거 n] 번호와 같음). 근거가 없으면 필드를 생략한다. 검수 상태는 review_status(데이터)로만 둔다.
    evidence: list[Evidence] | None = None


class EmrSummaryResponseBody(BaseModel):
    """EMR 1회 요약 응답.

    역할: 근거를 인용하지 않으며, 공개된 계약대로 ``evidence: null``을 유지한다.
    """

    answer: str
    source: str = _SOURCE
    emr_updated_at: datetime
    evidence: None = None


class ClinicalRecommendationRequestBody(PatientIdBody):
    """추천 의견(의심 진단·추천 검사·협진) 요청 본문.

    역할: 의료진 메모, 소견 키 지정, 질문, 용량을 계산할 약을 선택적으로 받는다.
    """

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
    """추천 의견 응답.

    역할: 추천 문장, 출처, EMR 갱신 시각, 규칙 평가 결과(``assessment``)와 인용 근거(없으면 생략)를 돌려준다.
    """

    answer: str
    source: str = _SOURCE
    emr_updated_at: datetime
    # 규칙 평가 결과(진단·검사·협진·추적 항목, 검사값 범주, DUR, 용량). 구조는 docs/assessment_schema.md.
    # 해당 없는 키는 생략하고, 값이 없는 키는 null로 둔다(SSE meta.assessment와 같은 JSON).
    assessment: AssessmentPayload
    evidence: list[Evidence] | None = None


class ErrorBody(BaseModel):
    """오류 응답 본문.

    역할: 안전한 오류 코드와 사용자 안내 문구만 담는다(요청 내용·내부 정보는 담지 않는다).
    """

    code: str
    message: str


class StoreReadiness(BaseModel):
    """``/ready``의 환자 문맥 저장소 상태.

    역할: 저장소 종류와 응답 여부를 보고한다.
    """

    backend: str = Field(description="memory(이 프로세스) · redis(워커·인스턴스 공유)")
    reachable: bool


class CorpusReadiness(BaseModel):
    """``/ready``의 말뭉치 상태.

    역할: 현재 스냅샷의 원천, DB 응답 여부, 릴리스·DUR 데이터셋 ID, 근거 문단·규칙 수를 보고한다(준비 판정에는 쓰지 않는다).
    """

    backend: Literal["postgres", "files", "none"] = Field(
        description="postgres: DB 활성 릴리스 · files: 파일 말뭉치(DB 미설정 또는 기동 시 연결 실패로 대체) · none: 근거·규칙 없음")
    database_reachable: bool | None = Field(
        description="말뭉치 DB 응답 여부(null: DB 미사용). false여도 메모리 스냅샷으로 계속 답하며, 의미 검색과 릴리스 갱신만 멈춘다")
    release_id: int | None
    evidence_chunks: int
    rules: int
    dur_dataset_id: int | None


class ReadinessBody(BaseModel):
    """``/ready`` 응답 본문.

    역할: ``ready``는 AI 호출 요청을 처리할 수 있다는 뜻이다(AI 설정 있음, 문맥 저장소 응답). Provider 연결 자체는
    확인하지 않는다.
    """

    status: Literal["ready", "not_ready"]
    ai_provider: Literal["configured", "not_configured"]
    patient_context_store: StoreReadiness
    corpus: CorpusReadiness


@dataclass(frozen=True)
class AppComponents:
    """라우트가 쓰는 구성 요소 묶음.

    역할: ``app.state.components``에 두고 ``Depends``로 각 엔드포인트에 주입한다.
    """

    service: ClinicalChatService | None
    mapper: RawEmrPayloadMapper
    store: ContextStore
    corpus: CorpusSource


def app_components(request: Request) -> AppComponents:
    """앱에 보관된 ``AppComponents``를 꺼낸다(의존성 주입용)."""
    return request.app.state.components


def chat_service(components: Annotated[AppComponents, Depends(app_components)]) -> ClinicalChatService:
    """AI 서비스를 주입한다. 설정되지 않았으면 ``ClinicalChatProviderUnavailableError``를 낸다."""
    if components.service is None:
        raise ClinicalChatProviderUnavailableError("AI provider is not configured")
    return components.service


def corpus_snapshot(components: Annotated[AppComponents, Depends(app_components)]) -> CorpusSnapshot:
    """요청마다 한 번 읽는다. 요청 전체가 한 릴리스의 근거·규칙·DUR을 쓴다."""
    return components.corpus.current


Components = Annotated[AppComponents, Depends(app_components)]
Service = Annotated[ClinicalChatService, Depends(chat_service)]
Snapshot = Annotated[CorpusSnapshot, Depends(corpus_snapshot)]


def create_app(
    service: ClinicalChatService | None = None,
    mapper: RawEmrPayloadMapper | None = None,
    store: ContextStore | None = None,
    *,
    corpus: CorpusSource | None = None,
    on_shutdown: Sequence[Callable[[], object]] = (),
) -> FastAPI:
    """새 내부 계약만 노출하는 ASGI 앱을 조립한다.

    역할: 구성 요소를 ``app.state``에 두고, 요청 로깅 미들웨어·검증 오류 처리기·도메인 예외 처리기와 모든
    엔드포인트를 등록한다. 종료 시 ``on_shutdown`` 자원과 문맥 저장소를 닫는다.
    ``service``가 없으면 AI 설정이 없는 fail-closed 앱이며, AI 호출이 필요한 요청은
    ``AI_PROVIDER_UNAVAILABLE``로 거부된다. ``corpus``는 요청마다 읽는 말뭉치 스냅샷(근거 검색·규칙·DUR)의
    유일한 출처이며, 없으면 근거·규칙이 없는 빈 스냅샷이다. ``on_shutdown``은 종료 시 닫을 자원(DB 풀 등)이다.
    입력:
        service: 임상 채팅 서비스. ``None``이면 fail-closed.
        mapper: EMR payload 변환기. ``None``이면 참조 DB 없는 기본 변환기.
        store: 환자 문맥 저장소. ``None``이면 메모리 저장소.
        corpus: 말뭉치 원천. ``None``이면 빈 ``StaticCorpus``.
        on_shutdown: 종료 시 호출할 동기·비동기 정리 함수들.
    출력: FastAPI 앱.
    """

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        yield
        for close in (*on_shutdown, components.store.close):
            try:
                await _call(close)
            except Exception as error:  # 자원 하나가 실패해도 종료 절차는 계속한다
                logger.warning("resource_close_failed error_type=%s", type(error).__name__)

    # /docs에서 모델마다 스키마 하나를 쓴다. payload 모델은 직렬화를 바꾸므로(값이 있는 키만) 이렇게 하지 않으면
    # 응답 스키마가 속성 없는 object가 된다.
    app = FastAPI(title="AX-G AI Clinical Chat API", version="1", lifespan=lifespan, separate_input_output_schemas=False)
    components = app.state.components = AppComponents(
        service, mapper or RawEmrPayloadMapper(), store or PatientContextStore(),
        corpus or StaticCorpus(),
    )

    @app.middleware("http")
    async def request_logging(request: Request, call_next):
        """안전한 상관관계 ID를 붙이고 실패한 HTTP 요청만 로그에 남긴다.

        요청 본문에는 EMR 데이터·질문·인증 정보가 있을 수 있으므로 절대 컨테이너 로그로 보내지 않는다.
        대신 요청 ID로 그런 데이터를 남기지 않고도 Bridge 로그와 Docker 로그를 맞춰 볼 수 있다.
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
        if response.status_code >= 400 and request.url.path != "/ready":  # /ready는 상태 변화를 직접 로그에 남긴다
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
        """``/internal/v1/`` 요청의 검증 실패는 위치·유형만 로그에 남기고 400 ``INVALID_REQUEST``로 바꾼다."""
        if request.url.path.startswith("/internal/v1/"):
            # Pydantic의 ``input`` 필드는 EMR payload 전체를 담을 수 있으므로 로그에 남기지 않는다.
            # 잘못된 연동 요청을 찾는 데는 필드 위치와 검증 유형이면 충분하다.
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
        """프로세스 생존 확인. 의존성은 확인하지 않는다."""
        return {"status": "ok"}

    last_readiness: list[str] = []  # 바뀔 때만 로그를 남긴다. 몇 초마다 오는 probe가 로그를 채우면 안 된다

    @app.get("/ready", response_model=ReadinessBody, responses={503: {"model": ReadinessBody}}, tags=["operations"])
    async def ready(components: Components) -> JSONResponse:
        """준비 상태: 실패 조건은 AI 설정 없음·문맥 저장소 무응답(503). 말뭉치 상태는 함께 보고만 한다."""
        body = await _readiness(components)
        if last_readiness != [body.status]:
            log = logger.info if body.status == "ready" else logger.warning
            log("readiness status=%s ai_provider=%s store=%s store_reachable=%s corpus=%s database_reachable=%s",
                body.status, body.ai_provider, body.patient_context_store.backend, body.patient_context_store.reachable,
                body.corpus.backend, body.corpus.database_reachable)
            last_readiness[:] = [body.status]
        return JSONResponse(body.model_dump(mode="json"), status_code=200 if body.status == "ready" else 503)

    _responses: dict[int | str, dict[str, Any]] = {status: {"model": ErrorBody} for status in (400, 403, 404, 502, 503)}

    # 사용 사례마다 JSON 엔드포인트와 SSE ``/stream`` 짝이 있다. 둘 다 같은 ``_Call``(Provider 인자와 ``answer``
    # 밖의 응답 필드)을 준비하며, SSE ``meta`` 이벤트가 그 필드를 그대로 담는다.

    @app.post("/internal/v1/patient-context", response_model=PatientContextReadyBody, responses=_responses, tags=["internal"])
    async def prepare_patient_context(
        body: PatientContextRequestBody, components: Components, service: Service, snapshot: Snapshot,
    ) -> PatientContextReadyBody:
        """EMR로 환자 문맥을 준비하고 초기 요약을 만든 뒤 저장한다."""
        # 갱신은 이전 항목을 무효화하는 데서 시작한다. 잘못된 EMR 데이터나 실패한 요약 때문에
        # 오래된 환자 문맥이 남아 있으면 안 된다.
        context = await _prepare_context(body, components, snapshot)
        summary = await service.summarize_prepared_snapshot(snapshot=context.data)
        await components.store.put(body.patient_id, context)
        return PatientContextReadyBody(patient_id=body.patient_id, patient_summary=summary, emr_updated_at=context.updated_at)

    @app.post("/internal/v1/patient-context/stream", response_model=None, responses=_responses, tags=["internal"])
    async def prepare_patient_context_stream(
        body: PatientContextRequestBody, request: Request, components: Components, service: Service, snapshot: Snapshot,
    ) -> StreamingResponse:
        """EMR 문맥을 바로 준비·저장하고 초기 요약을 SSE로 보낸다.

        ``meta``가 나가는 즉시 채팅이 문맥을 쓸 수 있다. ``done`` 전에 요약이 실패하면 JSON 엔드포인트처럼
        문맥을 버린다.
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
        """채팅 문맥을 저장하지 않고 EMR 1회 요약을 SSE로 보낸다."""
        call = _summary_call(body, components)
        return await _stream(request, lambda: service.stream_summary_once(**call.arguments), call.meta)

    @app.post("/internal/v1/clinical-chat", response_model=ClinicalChatResponseBody, responses=_responses, tags=["internal"])
    async def clinical_chat(
        body: ClinicalChatRequestBody, components: Components, service: Service, snapshot: Snapshot,
    ) -> ClinicalChatResponseBody:
        """준비된 환자 문맥과 말뭉치 근거로 임상 질문에 답한다."""
        call = await _chat_call(body, components, service, snapshot)
        return ClinicalChatResponseBody(answer=await service.answer_prepared_snapshot(**call.arguments), **call.fields)

    @app.post("/internal/v1/clinical-chat/stream", response_model=None, responses=_responses, tags=["internal"])
    async def clinical_chat_stream(
        body: ClinicalChatRequestBody, request: Request, components: Components, service: Service, snapshot: Snapshot,
    ) -> StreamingResponse:
        """같은 임상 답변을 Provider 이벤트를 노출하지 않고 SSE로 보낸다."""
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
    """준비된 사용 사례 하나.

    역할: Provider 메서드에 넘길 ``arguments``와 ``answer`` 밖의 응답 ``fields``를 함께 들고, JSON 응답과 SSE ``meta``가
    같은 값을 쓰게 한다.
    """

    arguments: dict[str, Any]
    fields: dict[str, Any]

    @property
    def meta(self) -> dict[str, object]:
        """SSE ``meta`` 이벤트 내용. 같은 필드에서 ``None``은 빼고 시각은 ISO 형식으로 바꾼다."""
        return {key: value.isoformat() if isinstance(value, datetime) else value
                for key, value in self.fields.items() if value is not None}


async def _readiness(components: AppComponents) -> ReadinessBody:
    """``/ready`` 응답을 만든다.

    역할: 문맥 저장소 ping과 말뭉치 DB 확인(스레드풀)을 한 뒤, 현재 스냅샷에서 원천(postgres/files/none)과 근거·규칙 수를 구한다.
    준비 판정은 AI 설정 여부와 저장소 응답만으로 한다.
    입력:
        components: 앱 구성 요소.
    출력: ``ReadinessBody``.
    """
    store_reachable = await components.store.ping()
    # 풀 확인은 연결을 기다릴 수 있으므로 이벤트 루프 밖에서 실행한다.
    database_reachable = await run_in_threadpool(components.corpus.database_reachable)
    snapshot = components.corpus.current
    if snapshot.release_id is not None:
        backend: Literal["postgres", "files", "none"] = "postgres"
    elif snapshot.evidence is not None or snapshot.recommender is not None:
        backend = "files"
    else:
        backend = "none"
    evidence = snapshot.evidence
    configured = components.service is not None
    return ReadinessBody(
        status="ready" if configured and store_reachable else "not_ready",
        ai_provider="configured" if configured else "not_configured",
        patient_context_store=StoreReadiness(backend=components.store.backend, reachable=store_reachable),
        corpus=CorpusReadiness(
            backend=backend, database_reachable=database_reachable, release_id=snapshot.release_id,
            evidence_chunks=len(evidence) if isinstance(evidence, Sized) else 0,
            rules=len(snapshot.recommender) if snapshot.recommender is not None else 0,
            dur_dataset_id=snapshot.dur_dataset_id,
        ),
    )


async def _prepare_context(
    body: PatientContextRequestBody, components: AppComponents, snapshot: CorpusSnapshot,
) -> PreparedRawEmrContext:
    """저장된 문맥을 먼저 버린 뒤 EMR을 변환한다(이 스냅샷의 규칙 입력 사실을 붙인다)."""
    await components.store.discard(body.patient_id)
    return with_rule_input(components.mapper.map(body.patient_id, body.emr_payload), snapshot)


def _summary_call(body: PatientContextRequestBody, components: AppComponents) -> _Call:
    """저장 없이 EMR을 변환해 1회 요약용 ``_Call``을 만든다."""
    context = components.mapper.map(body.patient_id, body.emr_payload)
    return _Call({"snapshot": context.data}, {"source": _SOURCE, "emr_updated_at": context.updated_at})


async def _chat_call(
    body: ClinicalChatRequestBody, components: AppComponents, service: ClinicalChatService, snapshot: CorpusSnapshot,
) -> _Call:
    """임상 채팅용 ``_Call``을 준비한다.

    역할: 저장된 환자 문맥에 규칙 입력을 붙이고, 질문에 맞는 근거와 규칙 평가(``chat_grounding``)를 스레드풀에서 구한다.
    입력:
        body: 채팅 요청 본문.
        components: 앱 구성 요소(문맥 저장소).
        service: 임상 채팅 서비스.
        snapshot: 이 요청의 말뭉치 스냅샷.
    출력: Provider 인자(질문·참고 기록·문맥·근거·평가)와 응답 필드(출처·EMR 갱신 시각·근거)를 담은 ``_Call``.
    예외: 문맥이 없으면 저장소의 ``PatientContextNotReadyError``가 전파된다.
    """
    context = with_rule_input(await components.store.get(body.patient_id), snapshot)
    # 검색은 임베딩 서버와 DB를 호출할 수 있으므로 이벤트 루프 밖에서 실행한다.
    grounding = await run_in_threadpool(chat_grounding, body.question, context, service, snapshot)
    return _Call(
        {"question": body.question, "reference_record": body.reference_record, "snapshot": context.data,
         "evidence": grounding.evidence, "assessment": grounding.assessment},
        {"source": _SOURCE, "emr_updated_at": context.updated_at, "evidence": evidence_payload(grounding.evidence) or None},
    )


async def _recommendation_call(
    body: ClinicalRecommendationRequestBody, components: AppComponents, service: ClinicalChatService, snapshot: CorpusSnapshot,
) -> _Call:
    """추천 의견용 ``_Call``을 준비한다.

    역할: 저장된 환자 문맥에 메모·소견·질문·용량 약을 반영해 규칙 평가와 근거(``recommendation_grounding``)를 구한다.
    입력:
        body: 추천 요청 본문.
        components: 앱 구성 요소(문맥 저장소).
        service: 임상 채팅 서비스.
        snapshot: 이 요청의 말뭉치 스냅샷.
    출력: Provider 인자(문맥·평가·근거·메모·질문)와 응답 필드(출처·EMR 갱신 시각·평가·근거)를 담은 ``_Call``.
    예외: 문맥이 없으면 ``PatientContextNotReadyError``, 규칙이 없으면 ``RecommendationUnavailableError``가 전파된다.
    """
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
    """Provider의 텍스트 조각을 SSE ``meta``/``delta``/``done`` 이벤트로 보낸다.

    역할: HTTP 200을 확정하기 전에 첫 조각을 먼저 받아 두므로, 연결 실패나 첫 토큰 실패는 일반 JSON 502로 응답한다.
    스트리밍 도중 실패하면 ``on_failure``를 호출하고 안전한 SSE ``error`` 이벤트로 끝낸다.
    입력:
        request: 현재 요청(로그용 요청 ID·경로).
        start: Provider 스트림(텍스트 조각 비동기 반복자)을 시작하는 함수.
        meta: 첫 ``meta`` 이벤트에 담을 필드.
        on_failure: 실패 시 호출할 동기·비동기 정리 함수(예: 문맥 폐기). 없으면 ``None``.
    출력: ``text/event-stream`` ``StreamingResponse``.
    예외: 스트림이 시작되지 않으면 ``ClinicalChatProviderUnavailableError``.
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
            # 첫 이벤트 이후 HTTP 상태는 이미 200이다. SSE 오류에 Provider 세부 정보·프롬프트·EMR을 드러내지 않는다.
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
            if isinstance(deltas, AsyncGenerator):  # 서비스 메서드는 비동기 제너레이터다
                await deltas.aclose()

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


async def _call(callback: Callable[[], object] | None) -> None:
    """동기·비동기 콜백을 실행한다(예: 원격 저장소의 문맥 폐기). ``None``이면 아무것도 하지 않는다."""
    if callback is not None:
        result = callback()
        if inspect.isawaitable(result):
            await result


def _domain_error_handler(status_code: int, code: str, message: str):
    """도메인 예외 하나를 고정된 오류 응답으로 바꾸는 예외 처리기를 만든다.

    역할: 처리기는 요청 ID·경로·오류 코드(``PatientContextUnavailableError``이면 진단 코드 ``failure``도)만 로그에 남기고
    ``ErrorBody`` 응답을 돌려준다.
    입력:
        status_code: HTTP 상태 코드.
        code: 오류 코드.
        message: 사용자 안내 문구.
    출력: FastAPI 예외 처리기(비동기 함수).
    """
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
    """미들웨어가 붙인 요청 ID. 없으면 ``unavailable``."""
    return getattr(request.state, "request_id", "unavailable")


def _error(status_code: int, code: str, message: str) -> JSONResponse:
    """``ErrorBody`` JSON 응답을 만든다."""
    return JSONResponse(status_code=status_code, content=ErrorBody(code=code, message=message).model_dump())


def _sse(event: str, data: Mapping[str, object]) -> str:
    """신뢰된 SSE 이벤트 하나를 Provider payload를 흘리지 않고 직렬화한다."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False, separators=(',', ':'))}\n\n"
