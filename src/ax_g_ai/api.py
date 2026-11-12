"""간소화된 Bridge 전용 환자 EMR 준비·임상 채팅 API.

모든 엔드포인트는 ``async``다. Provider 응답을 기다리는 동안 스레드를 점유하지 않으므로
동시 요청 수가 스레드풀 크기(기본 40)에 묶이지 않고, ``/health``도 지연되지 않는다.
도메인 예외는 아래 ``_ERROR_RESPONSES``의 예외 처리기에서 안전한 오류 코드로 바뀐다.
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, replace
from datetime import datetime
import json
import logging
import re
from time import perf_counter
from typing import Annotated, Mapping
from uuid import uuid4

from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.concurrency import run_in_threadpool
from pydantic import BaseModel, ConfigDict, Field, SerializerFunctionWrapHandler, StringConstraints, model_serializer

from ax_g_ai.adapters.main_bridge import RawEmrPayloadMapper
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
from ax_g_ai.services.clinical_recommendation.schema import AssessmentPayload
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
# Chat questions that ask for diagnoses, tests, medications or doses also get the stage-2 rule results (with DUR).
# A question that names a drug, or states pregnancy, gets them as well (``ClinicalRecommender.question_inputs``).
_RECOMMENDATION_INTENT = re.compile(
    r"추천|감별|의심|진단명|어떤\s*검사|검사가?\s*필요|필요한\s*검사|추가\s*검사|워크업|workup|work-up|협진|의뢰"
    r"|병용|금기|상호\s*작용|DUR|처방|복용|투약|약물|약제|(?<![요절계])약(?:을|이|은|들|과|도|\s|$|\?)"
    r"|먹어도|먹여도|(?:같이|함께)\s*(?:먹|복용|써|쓰|사용|투여)|임신|임부|임산부"
    r"|용량|투여량|감량|증량|mg\s*/\s*kg|신기능\s*(조정|보정)|dose|dosing|medication|drug", re.I,
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
    # 규칙 평가 결과(진단·검사·협진·추적 항목, 검사값 범주, DUR, 용량). 구조는 docs/assessment_schema.md.
    # 해당 없는 키는 생략하고, 값이 없는 키는 null로 둔다(SSE meta.assessment와 같은 JSON).
    assessment: AssessmentPayload
    evidence: list[dict[str, object]] | None = None

    @model_serializer(mode="wrap")
    def _omit_missing_evidence(self, handler: SerializerFunctionWrapHandler) -> dict[str, object]:
        data = handler(self)
        if data.get("evidence") is None:
            data.pop("evidence", None)
        return data


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

    _responses = {status: {"model": ErrorBody} for status in (400, 403, 404, 502, 503)}

    @app.post("/internal/v1/patient-context", response_model=PatientContextReadyBody, responses=_responses, tags=["internal"])
    async def prepare_patient_context(
        body: PatientContextRequestBody, components: Components, service: Service, snapshot: Snapshot,
    ) -> PatientContextReadyBody:
        # A refresh starts by invalidating the old item.  Neither malformed EMR
        # data nor a failed summary may leave stale patient context available.
        await components.store.discard(body.patient_id)
        context = _with_rule_input(components.mapper.map(body.patient_id, body.emr_payload), snapshot)
        summary = await service.summarize_prepared_snapshot(snapshot=context.data)
        await components.store.put(body.patient_id, context)
        return PatientContextReadyBody(patient_id=body.patient_id, patient_summary=summary, emr_updated_at=context.updated_at)

    @app.post("/internal/v1/emr-summary", response_model=ClinicalChatResponseBody, responses=_responses, tags=["internal"])
    async def emr_summary(body: PatientContextRequestBody, components: Components, service: Service) -> ClinicalChatResponseBody:
        """Bridge가 전달한 EMR을 저장 없이 한 번 요약한다."""
        context = components.mapper.map(body.patient_id, body.emr_payload)
        answer = await service.summarize_once(snapshot=context.data)
        return ClinicalChatResponseBody(answer=answer, emr_updated_at=context.updated_at)

    @app.post("/internal/v1/clinical-chat", response_model=ClinicalChatResponseBody, response_model_exclude_none=True, responses=_responses, tags=["internal"])
    async def clinical_chat(
        body: ClinicalChatRequestBody, components: Components, service: Service, snapshot: Snapshot,
    ) -> ClinicalChatResponseBody:
        context, evidence, assessment = await _chat_inputs(body, components, service, snapshot)
        answer = await service.answer_prepared_snapshot(
            question=body.question, reference_record=body.reference_record, snapshot=context.data, evidence=evidence,
            assessment=assessment,
        )
        return ClinicalChatResponseBody(answer=answer, emr_updated_at=context.updated_at, evidence=evidence_payload(evidence) or None)

    @app.post("/internal/v1/clinical-chat/stream", response_model=None, responses=_responses, tags=["internal"])
    async def clinical_chat_stream(
        body: ClinicalChatRequestBody, request: Request, components: Components, service: Service, snapshot: Snapshot,
    ) -> StreamingResponse:
        """Return the same clinical answer as SSE without exposing provider events."""
        context, evidence, assessment = await _chat_inputs(body, components, service, snapshot)
        return await _stream(
            request,
            lambda: service.stream_prepared_snapshot(
                question=body.question, reference_record=body.reference_record, snapshot=context.data, evidence=evidence,
                assessment=assessment,
            ),
            {"source": _SOURCE, "emr_updated_at": context.updated_at.isoformat(),
             **({"evidence": evidence_payload(evidence)} if evidence else {})},
        )

    @app.get("/internal/v1/clinical-recommendations/findings", tags=["internal"])
    async def recommendation_findings(snapshot: Snapshot) -> dict[str, object]:
        """규칙이 사용하는 소견·사건 키 목록(``findings`` 입력용)과 용량 규칙이 있는 약(``dose_drugs`` 입력용)."""
        rules = _require_recommender(snapshot)
        return {"findings": list(rules.finding_vocabulary), "rules": len(rules), "unreviewed_rules": rules.unreviewed_count,
                "dose_drugs": list(rules.dose_drugs)}

    # No response_model_exclude_none: it would also drop the nulls inside ``assessment``.
    @app.post("/internal/v1/clinical-recommendations", response_model=ClinicalRecommendationResponseBody, responses=_responses, tags=["internal"])
    async def clinical_recommendations(
        body: ClinicalRecommendationRequestBody, components: Components, service: Service, snapshot: Snapshot,
    ) -> ClinicalRecommendationResponseBody:
        """말뭉치 규칙 평가 + 근거 인용으로 의심 진단·추천 검사·협진 의견을 만든다."""
        context, evidence, assessment_payload = await _recommendation_inputs(body, components, service, snapshot)
        answer = await service.recommend(
            snapshot=context.data, assessment=assessment_payload, evidence=evidence,
            clinical_note=body.clinical_note, question=body.question,
        )
        return ClinicalRecommendationResponseBody(
            answer=answer, emr_updated_at=context.updated_at, assessment=assessment_payload,
            evidence=evidence_payload(evidence) or None,
        )

    @app.post("/internal/v1/clinical-recommendations/stream", response_model=None, responses=_responses, tags=["internal"])
    async def clinical_recommendations_stream(
        body: ClinicalRecommendationRequestBody, request: Request, components: Components, service: Service, snapshot: Snapshot,
    ) -> StreamingResponse:
        """같은 추천 의견을 SSE로 보낸다. 규칙 평가 결과·근거는 ``meta``에 먼저 담는다."""
        context, evidence, assessment_payload = await _recommendation_inputs(body, components, service, snapshot)
        return await _stream(
            request,
            lambda: service.stream_recommendation(
                snapshot=context.data, assessment=assessment_payload, evidence=evidence,
                clinical_note=body.clinical_note, question=body.question,
            ),
            {"source": _SOURCE, "emr_updated_at": context.updated_at.isoformat(), "assessment": assessment_payload,
             **({"evidence": evidence_payload(evidence)} if evidence else {})},
        )

    @app.post("/internal/v1/patient-context/stream", response_model=None, responses=_responses, tags=["internal"])
    async def prepare_patient_context_stream(
        body: PatientContextRequestBody, request: Request, components: Components, service: Service, snapshot: Snapshot,
    ) -> StreamingResponse:
        """Prepare the EMR context at once and stream the initial summary as SSE.

        Chat can use the context as soon as ``meta`` is sent.  A summary that
        fails before ``done`` discards the context, as the JSON endpoint does.
        """
        store = components.store
        await store.discard(body.patient_id)
        context = _with_rule_input(components.mapper.map(body.patient_id, body.emr_payload), snapshot)
        await store.put(body.patient_id, context)
        return await _stream(
            request,
            lambda: service.stream_summary_once(snapshot=context.data),
            {"patient_id": body.patient_id, "status": "ready", "source": _SOURCE, "emr_updated_at": context.updated_at.isoformat()},
            on_failure=lambda: store.discard(body.patient_id),
        )

    @app.post("/internal/v1/emr-summary/stream", response_model=None, responses=_responses, tags=["internal"])
    async def emr_summary_stream(
        body: PatientContextRequestBody, request: Request, components: Components, service: Service,
    ) -> StreamingResponse:
        """Stream a one-shot EMR summary without storing the chat context."""
        context = components.mapper.map(body.patient_id, body.emr_payload)
        return await _stream(
            request,
            lambda: service.stream_summary_once(snapshot=context.data),
            {"source": _SOURCE, "emr_updated_at": context.updated_at.isoformat()},
        )

    return app


def _require_recommender(snapshot: CorpusSnapshot) -> ClinicalRecommender:
    if snapshot.recommender is None or len(snapshot.recommender) == 0:
        raise RecommendationUnavailableError()
    return snapshot.recommender


def _with_rule_input(context: PreparedRawEmrContext, snapshot: CorpusSnapshot) -> PreparedRawEmrContext:
    """Rule facts mapped with this snapshot's rules: kept when the release matches, else mapped again."""
    recommender = snapshot.recommender
    if recommender is None or (context.rule_input is not None and context.rule_release == snapshot.release_id):
        return context
    try:
        rule_input = recommender.prepare(context.rule_source)
    except Exception as error:  # rule facts are optional; never block EMR preparation or chat
        logger.warning("clinical_rule_mapping_failed error_type=%s", type(error).__name__)
        return replace(context, rule_input=None)
    return replace(context, rule_input=rule_input, rule_release=snapshot.release_id)


async def _chat_inputs(
    body: ClinicalChatRequestBody, components: AppComponents, service: ClinicalChatService, snapshot: CorpusSnapshot,
) -> tuple[PreparedRawEmrContext, tuple[EvidenceChunk, ...], dict[str, object] | None]:
    context = _with_rule_input(await components.store.get(body.patient_id), snapshot)
    # Retrieval may call the embedding server and database; keep it off the event loop.
    evidence, assessment = await run_in_threadpool(_chat_grounding, body.question, context, service, snapshot)
    return context, evidence, assessment


def _chat_grounding(
    question: str, context: PreparedRawEmrContext, service: ClinicalChatService, snapshot: CorpusSnapshot,
) -> tuple[tuple[EvidenceChunk, ...], dict[str, object] | None]:
    """Evidence (and, for diagnosis/test questions, rule results) for one chat question."""
    searched = service.find_evidence(question, snapshot.evidence)
    if not searched:
        # A generic question ("추천 검사는?") has no topic words: add the recorded diagnoses.
        hint = _diagnosis_hint(context)
        if hint:
            searched = service.find_evidence(f"{question} {hint}", snapshot.evidence)
    recommender = snapshot.recommender
    if recommender is None or context.rule_input is None:
        return searched, None
    stated = recommender.question_inputs(question)
    if not stated and not _RECOMMENDATION_INTENT.search(question):
        return searched, None
    assessment = recommender.assess(context.rule_input, stated.findings, stated.dose_drugs, stated.proposed_drugs)
    alerts = (assessment.medication_alerts or {}).get("alerts")
    if not assessment.items and not assessment.lab_interpretations and not alerts and not assessment.dose_assessments:
        return searched, None
    rule_chunks = assessment.evidence[:_RULE_EVIDENCE_IN_CHAT]
    return _merge(rule_chunks, searched), _renumber(assessment, len(rule_chunks))


async def _recommendation_inputs(
    body: ClinicalRecommendationRequestBody, components: AppComponents, service: ClinicalChatService, snapshot: CorpusSnapshot,
) -> tuple[PreparedRawEmrContext, tuple[EvidenceChunk, ...], dict[str, object]]:
    context = await components.store.get(body.patient_id)
    rules = _require_recommender(snapshot)
    context = _with_rule_input(context, snapshot)
    if context.rule_input is None:
        # EMR mapping failed: evaluate with no EMR facts.
        context = replace(context, rule_input=rules.prepare({}))
    query = " ".join(part for part in (body.question, body.clinical_note, _diagnosis_hint(context)) if part)
    # The note extraction (an LLM call) and the evidence search do not depend on each other: run them together.
    extracted, searched = await asyncio.gather(
        service.extract_findings(note=body.clinical_note, vocabulary=rules.finding_vocabulary) if body.clinical_note else _value({}),
        run_in_threadpool(service.find_evidence, query, snapshot.evidence) if query else _value(()),
    )
    stated = rules.question_inputs(body.question, body.clinical_note)
    # Explicit clinician input wins over the memo extraction, which wins over facts stated in the question text.
    findings: dict[str, bool] = {**stated.findings, **extracted, **(body.findings or {})}
    dose_drugs = tuple(dict.fromkeys((*(body.dose_drugs or ()), *stated.dose_drugs)))
    assessment = rules.assess(context.rule_input, findings, dose_drugs, stated.proposed_drugs)  # type: ignore[arg-type]
    evidence = _merge(assessment.evidence, searched)
    logger.info(
        "clinical_recommendation_assessed items=%d met=%d findings=%d unknown_findings=%d evidence=%d unreviewed=%s doses=%d",
        len(assessment.items), sum(1 for item in assessment.items if item.get("status") == "met"),
        len(assessment.findings_used), len(assessment.unknown_findings), len(evidence), assessment.unreviewed,
        len(assessment.dose_assessments or ()),
    )
    return context, evidence, assessment.as_dict()


async def _value(value):
    return value


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
