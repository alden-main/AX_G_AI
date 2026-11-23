"""준비된 비식별 EMR 문맥으로 Provider를 호출하는 의료진 챗봇 서비스.

설계상 위치: 서비스 계층의 Provider 호출 지점. API 라우트와 ``clinical_grounding``이 채팅 답변·스트리밍·환자 요약·
추천·메모 소견 추출에 이 서비스를 쓰고, 이 서비스는 ``prompts.clinical``로 입력을 조립해 ``ProviderClient``를 부른다.

역할: 요청 종류별 Provider 요청(``OpenAIRequest``)을 만들고, 말뭉치 스냅샷의 검색기로 인용 근거를 찾으며,
Provider 실패를 분류해 로그로 남긴 뒤 ``ClinicalChatProviderUnavailableError``로 바꾼다.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import Any

from ax_g_ai.errors import ClinicalChatProviderUnavailableError
from ax_g_ai.prompts.clinical import (
    CLINICAL_INSTRUCTIONS,
    FINDING_EXTRACTION_INSTRUCTIONS,
    INITIAL_PATIENT_SUMMARY_REQUEST,
    ONE_SHOT_PATIENT_SUMMARY_REQUEST,
    PROMPT_VERSION,
    RECOMMENDATION_REQUEST,
    compose_finding_extraction,
    compose_prepared_snapshot_question,
    compose_recommendation_input,
    finding_extraction_output,
)
from ax_g_ai.services.clinical_evidence import EvidenceChunk, EvidenceRetriever
from ax_g_ai.services.clinical_recommendation import prompt_view
from ax_g_ai.services.ports import OpenAIProviderError, OpenAIRequest, ProviderClient, ProviderOutputError, ReferenceLookup
from ax_g_ai.services.prepared_patient_summary import PreparedPatientSummary

logger = logging.getLogger("uvicorn.error")


class ClinicalChatService:
    """Provider 호출과 실패 분류를 담당한다.

    역할: 채팅·요약·추천·소견 추출 요청을 조립해 Provider에 보내고 결과(문자열, 스트림, 구조화 모델)를 돌려준다.

    ``snapshot``은 ``RawEmrPayloadMapper``가 비식별 처리한 문맥이어야 한다. Provider
    실패는 원인과 무관하게 ``ClinicalChatProviderUnavailableError``로 바뀌고, 로그에는
    프롬프트·EMR·API Key 없이 분류된 사유만 남는다.

    근거 검색기는 갖지 않는다. 근거는 요청의 말뭉치 스냅샷(``CorpusSnapshot.evidence``)에서 찾아
    (``find_evidence``) 답변 메서드에 넘기므로, 한 요청의 근거·규칙·DUR이 같은 릴리스에서 온다.
    """

    def __init__(
        self,
        provider: ProviderClient,
        reference_lookup: ReferenceLookup | None = None,
        *, evidence_limit: int = 3, allow_general_knowledge: bool = False,
    ) -> None:
        """서비스를 만든다.

        역할: Provider, 참조 DB 조회기, 근거 개수 한도, 일반 지식 허용 여부를 보관한다.
        입력:
            provider: Provider 호출 클라이언트.
            reference_lookup: 질문으로 로컬 참조 DB를 찾는 조회기. ``None``이면 참조를 싣지 않는다.
            evidence_limit: 질문 하나에 인용할 근거 문단 최대 개수.
            allow_general_knowledge: 추천 답변에 말뭉치 출처 없는 일반 지식 제안을 허용할지 여부.
        출력: 없음.
        """
        self._provider = provider
        # False(출처 표기 전용): 추천은 말뭉치 출처 없는 제안을 덧붙일 수 없다.
        self._allow_general_knowledge = allow_general_knowledge
        self._reference_lookup = reference_lookup
        self._evidence_limit = evidence_limit

    def find_evidence(self, question: str, retriever: EvidenceRetriever | None) -> tuple[EvidenceChunk, ...]:
        """``question``에 인용할 근거를 ``retriever``(요청의 말뭉치 스냅샷)에서 찾는다.

        역할: ``evidence_limit``개까지 검색하고, 검색 실패는 경고 로그만 남기고 빈 결과로 넘긴다(근거는 선택 사항).
            로그에는 문단 id와 미검수 개수만 남긴다.
        입력:
            question: 검색할 질문 텍스트.
            retriever: 스냅샷의 근거 검색기. ``None``이면 검색하지 않는다.
        출력: 인용할 ``EvidenceChunk`` 튜플(없으면 빈 튜플).
        """
        if retriever is None:  # ``not retriever``가 아님: 빈 색인은 거짓(``__len__``)이지만 여전히 검색할 수 있다
            return ()
        try:
            chunks = retriever.search(question, self._evidence_limit)
        except Exception as error:  # 근거는 선택 사항이므로 답변을 절대 막지 않는다
            logger.warning("clinical_evidence_failed error_type=%s", type(error).__name__)
            return ()
        if chunks:
            logger.info(
                "clinical_evidence_cited chunk_ids=%s unreviewed=%d",
                ",".join(chunk.chunk_id for chunk in chunks),
                sum(1 for chunk in chunks if chunk.review_status != "approved"),
            )
        return chunks

    async def answer_prepared_snapshot(
        self, *, question: str, snapshot: Mapping[str, Any], reference_record: str | None = None,
        evidence: Sequence[EvidenceChunk] = (), assessment: Mapping[str, Any] | None = None,
    ) -> str:
        """준비된 동일 환자의 EMR 문맥으로 한국어 답변을 생성한다."""
        request = self._chat_request("prepared_snapshot_chat", question, snapshot, reference_record, evidence, assessment)
        with _provider_failure(request.operation):
            response = await self._provider.respond(request)
        return response.answer

    async def stream_prepared_snapshot(
        self, *, question: str, snapshot: Mapping[str, Any], reference_record: str | None = None,
        evidence: Sequence[EvidenceChunk] = (), assessment: Mapping[str, Any] | None = None,
    ) -> AsyncIterator[str]:
        """준비된 EMR 문맥 답변을 Provider 텍스트가 도착하는 대로 조각 단위로 내보낸다."""
        request = self._chat_request("prepared_snapshot_chat_stream", question, snapshot, reference_record, evidence, assessment)
        with _provider_failure(request.operation):
            async for delta in self._provider.stream(request):
                yield delta

    async def extract_findings(self, *, note: str, vocabulary: Sequence[str]) -> dict[str, bool]:
        """의료진 메모를 규칙 소견 키로 옮긴다(명시됨 → True, 부정됨 → False).

        역할: 어휘로 제한한 구조화 출력으로 Provider에 소견 추출을 요청한다. 형식이 깨지거나 잘린 Provider 답변은
            요청을 실패시키지 않고 소견 없음으로 처리한다. 호출자가 ``findings_used``를 돌려주므로 의료진은 무엇이
            적용되었는지 볼 수 있다. 같은 키가 명시와 부정 양쪽에 있으면 부정으로 본다.
        입력:
            note: 의료진 메모. 공백뿐이면 호출하지 않는다.
            vocabulary: 허용할 규칙 소견 키 목록. 비어 있으면 호출하지 않는다.
        출력: 소견 키 → True/False 딕셔너리(없으면 빈 딕셔너리).
        예외: Provider 호출 실패 시 ``ClinicalChatProviderUnavailableError``.
        """
        if not note.strip() or not vocabulary:
            return {}
        request = _request(
            "finding_extraction", compose_finding_extraction(note, list(vocabulary)), instructions=FINDING_EXTRACTION_INSTRUCTIONS,
        )
        with _provider_failure(request.operation):
            try:
                extracted = await self._provider.respond_structured(request, finding_extraction_output(tuple(vocabulary)))
            except ProviderOutputError as error:
                logger.warning("finding_extraction_invalid reason=%s", error)
                return {}
        # 명시와 부정이 함께 있는 키는 부정으로 센다.
        return {**dict.fromkeys(extracted.present, True), **dict.fromkeys(extracted.absent, False)}

    async def recommend(
        self, *, snapshot: Mapping[str, Any], assessment: Mapping[str, Any], evidence: Sequence[EvidenceChunk],
        clinical_note: str | None = None, question: str | None = None,
    ) -> str:
        """규칙 평가 결과와 근거로 추천 답변을 한 번에 생성한다."""
        request = self._recommendation_request("clinical_recommendation", snapshot, assessment, evidence, clinical_note, question)
        with _provider_failure(request.operation):
            response = await self._provider.respond(request)
        return response.answer

    async def stream_recommendation(
        self, *, snapshot: Mapping[str, Any], assessment: Mapping[str, Any], evidence: Sequence[EvidenceChunk],
        clinical_note: str | None = None, question: str | None = None,
    ) -> AsyncIterator[str]:
        """추천 답변을 Provider 텍스트가 도착하는 대로 조각 단위로 내보낸다."""
        request = self._recommendation_request("clinical_recommendation_stream", snapshot, assessment, evidence, clinical_note, question)
        with _provider_failure(request.operation):
            async for delta in self._provider.stream(request):
                yield delta

    def _recommendation_request(
        self, operation: str, snapshot: Mapping[str, Any], assessment: Mapping[str, Any], evidence: Sequence[EvidenceChunk],
        clinical_note: str | None, question: str | None,
    ) -> OpenAIRequest:
        """추천 요청을 만든다(질문이 없으면 ``RECOMMENDATION_REQUEST``)."""
        return _request(operation, compose_recommendation_input(
            snapshot, prompt_view(assessment), evidence_payload(evidence), clinical_note=clinical_note,
            question=question or RECOMMENDATION_REQUEST, allow_general_knowledge=self._allow_general_knowledge,
        ))

    async def summarize_prepared_snapshot(self, *, snapshot: Mapping[str, Any]) -> PreparedPatientSummary:
        """준비 시점에 EMR 문맥으로 화면용 구조화 환자 상태 요약을 생성한다."""
        request = _request("patient_context_summary", compose_prepared_snapshot_question(INITIAL_PATIENT_SUMMARY_REQUEST, snapshot))
        with _provider_failure(request.operation):
            return await self._provider.respond_structured(request, PreparedPatientSummary)

    async def summarize_once(self, *, snapshot: Mapping[str, Any]) -> str:
        """EMR에서 독립적인 간결 요약을 한 번 생성한다."""
        request = _summary_request("one_shot_patient_summary", snapshot)
        with _provider_failure(request.operation):
            response = await self._provider.respond(request)
        return response.answer

    async def stream_summary_once(self, *, snapshot: Mapping[str, Any]) -> AsyncIterator[str]:
        """일회성 EMR 요약을 Provider 텍스트가 도착하는 대로 조각 단위로 내보낸다.

        ``summarize_prepared_snapshot``과 달리 JSON 스키마가 없으므로, 전체 요약이 생성되기 전에 첫 문장을 보여 줄 수 있다.
        """
        request = _summary_request("one_shot_patient_summary_stream", snapshot)
        with _provider_failure(request.operation):
            async for delta in self._provider.stream(request):
                yield delta

    def _chat_request(
        self, operation: str, question: str, snapshot: Mapping[str, Any], reference_record: str | None,
        evidence: Sequence[EvidenceChunk], assessment: Mapping[str, Any] | None = None,
    ) -> OpenAIRequest:
        """채팅 답변용 Provider 요청을 만든다.

        역할: 조회기가 있으면 질문으로 로컬 참조 DB를 찾아 함께 싣고, 근거는 번호를 붙이고 규칙 평가 결과는
            프롬프트용 형태(``prompt_view``)로 바꿔 ``compose_prepared_snapshot_question``에 넘긴다.
        입력:
            operation: trace·로그에 남길 작업 이름.
            question: 의료진 질문.
            snapshot: 비식별 EMR 문맥.
            reference_record: 참고 기록 텍스트(없으면 ``None``).
            evidence: 인용할 근거 문단.
            assessment: 규칙 평가 결과(없으면 ``None``, 싣지 않음).
        출력: ``OpenAIRequest``.
        """
        references =self._reference_lookup.search_question(question) if self._reference_lookup else None
        return _request(operation, compose_prepared_snapshot_question(
            question, snapshot, reference_record, references, evidence=evidence_payload(evidence),
            assessment=prompt_view(assessment) if assessment else None,
        ))


def _request(operation: str, input: str, *, instructions: str = CLINICAL_INSTRUCTIONS) -> OpenAIRequest:
    """작업 이름과 프롬프트 버전을 붙인 Provider 요청(둘 다 trace에 남으며 환자 데이터는 없다)."""
    return OpenAIRequest(instructions, input, operation=operation, prompt_version=PROMPT_VERSION)


def _summary_request(operation: str, snapshot: Mapping[str, Any]) -> OpenAIRequest:
    """일회성 환자 요약 요청을 만든다."""
    return _request(operation, compose_prepared_snapshot_question(ONE_SHOT_PATIENT_SUMMARY_REQUEST, snapshot, one_shot_summary=True))


def evidence_payload(chunks: Sequence[EvidenceChunk]) -> list[dict[str, object]]:
    """프롬프트와 API 응답에 함께 쓰는 번호 붙은 근거(같은 번호 체계, 1부터).

    ``priority: reference``는 같은 주제의 국내 기준에 밀린 해외 문단을 표시한다.
    """
    return [{"n": number, **chunk.as_dict()} for number, chunk in enumerate(chunks, start=1)]


@contextmanager
def _provider_failure(operation: str) -> Iterator[None]:
    """감싼 블록의 Provider 실패를 분류해 로그로 남기고 서비스 오류로 바꾼다.

    역할: 프롬프트·EMR·API Key 없이 작업 이름, 오류 타입, 사유(``OpenAIProviderError``면 그 메시지, 아니면
        ``unexpected_provider_exception``)만 기록한다.
    입력:
        operation: 로그와 오류 메시지에 쓸 작업 이름.
    출력: 없음(컨텍스트 매니저로 한 번 ``yield``).
    예외: 블록에서 어떤 예외가 나도 ``ClinicalChatProviderUnavailableError``로 바꿔 던진다.
    """
    try:
        yield
    except Exception as error:
        reason = str(error) if isinstance(error, OpenAIProviderError) else "unexpected_provider_exception"
        logger.warning(
            "ai_provider_failed operation=%s error_type=%s reason=%s",
            operation,
            type(error).__name__,
            reason,
        )
        raise ClinicalChatProviderUnavailableError(f"{operation} is unavailable") from error
