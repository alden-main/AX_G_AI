"""준비된 비식별 EMR 문맥으로 Provider를 호출하는 의료진 챗봇 서비스.

설계상 위치: 서비스 계층의 Provider 호출 지점. API 라우트와 ``clinical_grounding``이 채팅 답변·스트리밍·환자 요약·
추천·메모 소견 추출에 이 서비스를 쓰고, 이 서비스는 ``prompts.clinical``로 입력을 조립해 ``ProviderClient``를 부른다.

역할: 요청 종류별 Provider 요청(``OpenAIRequest``)을 만들고, 말뭉치 스냅샷의 검색기로 인용 근거를 찾으며,
Provider 실패를 분류해 로그로 남긴 뒤 ``ClinicalChatProviderUnavailableError``로 바꾼다.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator, AsyncIterator, Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import Any

from ax_g_ai.errors import ClinicalChatProviderUnavailableError
from ax_g_ai.prompts.clinical import (
    FINDING_EXTRACTION_INSTRUCTIONS,
    INITIAL_PATIENT_SUMMARY_REQUEST,
    ONE_SHOT_PATIENT_SUMMARY_REQUEST,
    PROMPT_VERSION,
    RECOMMENDATION_REQUEST,
    ModelProfile,
    compose_finding_extraction,
    compose_prepared_snapshot_question,
    compose_recommendation_input,
    finding_extraction_output,
    prompt_texts,
)
from ax_g_ai.prompts.data_labels import PromptLabels
from ax_g_ai.services.answer_review import DUR_ONLY_ANSWER, UNUSABLE, check_opening, dur_notice, fallback_answer, review_answer
from ax_g_ai.services.clinical_evidence import EvidenceChunk, EvidenceRetriever
from ax_g_ai.services.clinical_recommendation import prompt_view
from ax_g_ai.services.ports import OpenAIProviderError, OpenAIRequest, ProviderClient, ProviderOutputError, ReferenceLookup
from ax_g_ai.services.prepared_patient_summary import PreparedPatientSummary
from ax_g_ai.services.question_facts import asks_only_for_dur

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
        *, evidence_limit: int = 3, allow_general_knowledge: bool = False, prompt_labels: PromptLabels = "korean",
        model_profile: ModelProfile = "compact",
    ) -> None:
        """서비스를 만든다.

        역할: Provider, 참조 DB 조회기, 근거 개수 한도, 일반 지식 허용 여부를 보관한다.
        입력:
            provider: Provider 호출 클라이언트.
            reference_lookup: 질문으로 로컬 참조 DB를 찾는 조회기. ``None``이면 참조를 싣지 않는다.
            evidence_limit: 질문 하나에 인용할 근거 문단 최대 개수.
            allow_general_knowledge: 추천 답변에 말뭉치 출처 없는 일반 지식 제안을 허용할지 여부.
            prompt_labels: 프롬프트 데이터·규칙 문구의 항목 이름 표기 방식(``korean`` 기본, ``english``는 비교 실험용).
            model_profile: 모델 프로필. ``compact``면 DUR만 묻는 질문을 모델 없이 답하고, ``capable``이면 모델이 답한다.
        출력: 없음.
        """
        self._provider = provider
        # False(출처 표기 전용): 추천은 말뭉치 출처 없는 제안을 덧붙일 수 없다.
        self._allow_general_knowledge = allow_general_knowledge
        self._reference_lookup = reference_lookup
        self._evidence_limit = evidence_limit
        self._labels = prompt_labels
        self._profile = model_profile

    @property
    def answers_dur_only_itself(self) -> bool:
        """DUR만 묻는 질문을 모델 없이 DUR 점검 블록으로 답하는지(``compact`` 프로필). 그러면 근거 검색도 필요 없다."""
        return self._profile == "compact"

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
        """준비된 동일 환자의 EMR 문맥으로 한국어 답변을 생성한다(DUR 점검 결과가 있으면 끝에 고정 문장으로 붙인다).

        ``compact`` 프로필에서는 DUR 점검만 묻는 질문을 모델을 부르지 않고 DUR 점검 블록으로 답한다(``_server_answer``).
        """
        notice = dur_notice(assessment, prescribed=bool(snapshot.get("prescriptions")))
        if self.answers_dur_only_itself and (answer := _server_answer("prepared_snapshot_chat", question, notice)) is not None:
            return answer
        request = self._chat_request("prepared_snapshot_chat", question, snapshot, reference_record, evidence, assessment)
        with _provider_failure(request.operation):
            answer = await self._respond_checked(request, question, has_notice=notice is not None)
        return answer + (notice or "")

    async def stream_prepared_snapshot(
        self, *, question: str, snapshot: Mapping[str, Any], reference_record: str | None = None,
        evidence: Sequence[EvidenceChunk] = (), assessment: Mapping[str, Any] | None = None,
    ) -> AsyncIterator[str]:
        """준비된 EMR 문맥 답변을 Provider 텍스트가 도착하는 대로 조각 단위로 내보낸다(DUR 경고 문장은 마지막 조각)."""
        notice = dur_notice(assessment, prescribed=bool(snapshot.get("prescriptions")))
        if self.answers_dur_only_itself and (answer := _server_answer("prepared_snapshot_chat_stream", question, notice)) is not None:
            yield answer
            return
        request = self._chat_request("prepared_snapshot_chat_stream", question, snapshot, reference_record, evidence, assessment)
        with _provider_failure(request.operation):
            async for delta in self._stream_checked(request, question, has_notice=notice is not None):
                yield delta
        if notice:
            yield notice

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
        request = self._request(
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
        """규칙 평가 결과와 근거로 추천 답변을 한 번에 생성한다(DUR 점검 결과가 있으면 끝에 고정 문장으로 붙인다)."""
        notice = dur_notice(assessment, prescribed=bool(snapshot.get("prescriptions")))
        request = self._recommendation_request("clinical_recommendation", snapshot, assessment, evidence, clinical_note, question)
        with _provider_failure(request.operation):
            answer = await self._respond_checked(request, question or RECOMMENDATION_REQUEST, has_notice=notice is not None)
        return answer + (notice or "")

    async def stream_recommendation(
        self, *, snapshot: Mapping[str, Any], assessment: Mapping[str, Any], evidence: Sequence[EvidenceChunk],
        clinical_note: str | None = None, question: str | None = None,
    ) -> AsyncIterator[str]:
        """추천 답변을 Provider 텍스트가 도착하는 대로 조각 단위로 내보낸다."""
        notice = dur_notice(assessment, prescribed=bool(snapshot.get("prescriptions")))
        request = self._recommendation_request("clinical_recommendation_stream", snapshot, assessment, evidence, clinical_note, question)
        with _provider_failure(request.operation):
            async for delta in self._stream_checked(request, question or RECOMMENDATION_REQUEST, has_notice=notice is not None):
                yield delta
        if notice:
            yield notice

    def _recommendation_request(
        self, operation: str, snapshot: Mapping[str, Any], assessment: Mapping[str, Any], evidence: Sequence[EvidenceChunk],
        clinical_note: str | None, question: str | None,
    ) -> OpenAIRequest:
        """추천 요청을 만든다(질문이 없으면 ``RECOMMENDATION_REQUEST``)."""
        return self._request(operation, compose_recommendation_input(
            snapshot, prompt_view(assessment), evidence_payload(evidence), clinical_note=clinical_note,
            question=question or RECOMMENDATION_REQUEST, allow_general_knowledge=self._allow_general_knowledge, labels=self._labels,
            profile=self._profile,
        ))

    async def summarize_prepared_snapshot(self, *, snapshot: Mapping[str, Any]) -> PreparedPatientSummary:
        """준비 시점에 EMR 문맥으로 화면용 구조화 환자 상태 요약을 생성한다."""
        request = self._request(
            "patient_context_summary", compose_prepared_snapshot_question(INITIAL_PATIENT_SUMMARY_REQUEST, snapshot, labels=self._labels),
        )
        with _provider_failure(request.operation):
            return await self._provider.respond_structured(request, PreparedPatientSummary)

    async def summarize_once(self, *, snapshot: Mapping[str, Any]) -> str:
        """EMR에서 독립적인 간결 요약을 한 번 생성한다."""
        request = self._summary_request("one_shot_patient_summary", snapshot)
        with _provider_failure(request.operation):
            return await self._respond_checked(request, ONE_SHOT_PATIENT_SUMMARY_REQUEST)

    async def stream_summary_once(self, *, snapshot: Mapping[str, Any]) -> AsyncIterator[str]:
        """일회성 EMR 요약을 Provider 텍스트가 도착하는 대로 조각 단위로 내보낸다.

        ``summarize_prepared_snapshot``과 달리 JSON 스키마가 없으므로, 전체 요약이 생성되기 전에 첫 문장을 보여 줄 수 있다.
        """
        request = self._summary_request("one_shot_patient_summary_stream", snapshot)
        with _provider_failure(request.operation):
            async for delta in self._stream_checked(request, ONE_SHOT_PATIENT_SUMMARY_REQUEST):
                yield delta

    async def _respond_checked(self, request: OpenAIRequest, question: str, *, has_notice: bool = False) -> str:
        """Provider 답변을 점검하고, 쓸 수 없거나 내부 키가 보이면 한 번 다시 요청한다.

        역할: 질문 되풀이는 잘라 낸다. 다시 받은 답변에도 내부 키만 보이면 그 답변을 쓰고, 여전히 쓸 수 없으면(질문 되풀이뿐·JSON)
            오류 대신 고정 안내 문장(``fallback_answer``)을 돌려준다. Provider 호출 자체의 실패는 그대로 전파된다.
        입력:
            request: Provider 요청.
            question: 되풀이 비교에 쓸 질문(프롬프트의 ``[의료진 질문]``).
            has_notice: 답변 끝에 DUR 점검 블록이 붙는지(안내 문장을 고른다).
        출력: 점검한 답변 또는 안내 문장.
        """
        text, problem = "", None
        for attempt in (1, 2):
            text, problem = review_answer((await self._provider.respond(request)).answer, question)
            if problem is None:
                return text
            _log_rejected(request.operation, problem, attempt)
        if problem not in UNUSABLE:
            return text
        _log_fallback(request.operation, problem)
        return fallback_answer(has_notice=has_notice)

    async def _stream_checked(self, request: OpenAIRequest, question: str, *, has_notice: bool = False) -> AsyncIterator[str]:
        """Provider 스트림 앞부분을 점검해 질문 되풀이를 잘라 내고, 쓸 수 없으면 한 번 다시 요청한다.

        역할: 앞부분이 질문과 갈라질 때까지만 모아 두고(되풀이가 아니면 첫 조각을 바로 내보낸다), 아직 아무것도 내보내지
            않았을 때만 다시 요청한다. 내보내기 시작한 뒤의 내용(내부 키 등)은 고칠 수 없다. 두 번 모두 쓸 수 없으면
            오류 대신 고정 안내 문장(``fallback_answer``)을 한 조각으로 내보낸다.
        입력:
            request: Provider 요청.
            question: 되풀이 비교에 쓸 질문.
            has_notice: 답변 끝에 DUR 점검 블록이 붙는지(안내 문장을 고른다).
        출력: 텍스트 조각 비동기 반복자.
        """
        problem = None
        for attempt in (1, 2):
            buffer, started = "", False
            deltas = self._provider.stream(request)
            try:
                async for delta in deltas:
                    if started:
                        yield delta
                        continue
                    buffer += delta
                    opening = check_opening(buffer, question)
                    if opening.state == "bad":
                        break
                    if opening.state == "ok":
                        started = True
                        yield opening.text
                else:
                    if not started:
                        opening = check_opening(buffer, question, final=True)
                        if opening.state == "ok":
                            started = True
                            yield opening.text
            finally:
                if isinstance(deltas, AsyncGenerator):  # 버린 스트림의 연결을 닫는다
                    await deltas.aclose()
            if started:
                return
            problem = opening.problem
            _log_rejected(request.operation, problem, attempt)
        _log_fallback(request.operation, problem)
        yield fallback_answer(has_notice=has_notice)

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
        return self._request(operation, compose_prepared_snapshot_question(
            question, snapshot, reference_record, references, evidence=evidence_payload(evidence),
            assessment=prompt_view(assessment) if assessment else None, labels=self._labels, profile=self._profile,
        ))

    def _request(self, operation: str, input: str, *, instructions: str | None = None) -> OpenAIRequest:
        """작업 이름과 프롬프트 버전을 붙인 Provider 요청(둘 다 trace에 남으며 환자 데이터는 없다).

        역할: 지시문 기본값은 표기 방식에 맞춘 임상 지시문이다. 영문 키 표기면 프롬프트 버전에 ``-en``을, ``capable`` 프로필이면
            ``-capable``을 붙여 추적에서 구분한다.
        """
        version = PROMPT_VERSION + ("-en" if self._labels == "english" else "") + ("-capable" if self._profile == "capable" else "")
        return OpenAIRequest(
            prompt_texts(self._labels, self._profile).instructions if instructions is None else instructions, input,
            operation=operation, prompt_version=version,
        )

    def _summary_request(self, operation: str, snapshot: Mapping[str, Any]) -> OpenAIRequest:
        """일회성 환자 요약 요청을 만든다."""
        return self._request(operation, compose_prepared_snapshot_question(
            ONE_SHOT_PATIENT_SUMMARY_REQUEST, snapshot, one_shot_summary=True, labels=self._labels,
        ))


def _log_rejected(operation: str, problem: object, attempt: int) -> None:
    """점검에서 걸린 답변을 기록한다(답변 본문·프롬프트는 남기지 않는다)."""
    logger.warning("provider_answer_rejected operation=%s reason=%s attempt=%d", operation, problem, attempt)


def _log_fallback(operation: str, problem: object) -> None:
    """두 번 모두 쓸 수 없어 안내 문장으로 답한 요청을 기록한다(API 응답은 200이므로 이 로그로 집계한다)."""
    logger.warning("provider_answer_fallback operation=%s reason=%s", operation, problem)


def _server_answer(operation: str, question: str, notice: str | None) -> str | None:
    """DUR 점검만 묻는 질문이고 DUR 점검 블록이 있으면 모델 없이 만든 답변을, 아니면 ``None``을 돌려준다.

    역할: 작은 자체 모델이 이 질문에 DUR 결론을 뒤집거나 프롬프트 JSON을 그대로 내는 일이 있어, 결정적으로 답한다.
    """
    if notice is None or not asks_only_for_dur(question):
        return None
    logger.info("server_answer operation=%s kind=dur_only", operation)
    return DUR_ONLY_ANSWER + notice


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
