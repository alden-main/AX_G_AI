"""준비된 비식별 EMR 문맥으로 Provider를 호출하는 의료진 챗봇 서비스."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator, Sequence
from contextlib import contextmanager
import json
import logging
from typing import Any, Mapping

from ax_g_ai.services.ports import OpenAIProviderError, OpenAIRequest, ProviderClient
from ax_g_ai.errors import ClinicalChatProviderUnavailableError
from ax_g_ai.prompts.clinical import (
    CLINICAL_INSTRUCTIONS,
    FINDING_EXTRACTION_INSTRUCTIONS,
    INITIAL_PATIENT_SUMMARY_REQUEST,
    ONE_SHOT_PATIENT_SUMMARY_REQUEST,
    RECOMMENDATION_REQUEST,
    compose_finding_extraction,
    compose_prepared_snapshot_question,
    compose_recommendation_input,
    finding_extraction_schema,
)
from ax_g_ai.services.prepared_patient_summary import (
    PATIENT_SUMMARY_OUTPUT_SCHEMA,
    PreparedPatientSummary,
)
from ax_g_ai.services.clinical_evidence import EvidenceChunk, EvidenceRetriever
from ax_g_ai.services.clinical_recommendation import prompt_view
from ax_g_ai.services.reference_lookup import SqliteReferenceLookup

logger = logging.getLogger("uvicorn.error")


class ClinicalChatService:
    """Provider 호출과 실패 분류를 담당한다.

    ``snapshot``은 ``RawEmrPayloadMapper``가 비식별 처리한 문맥이어야 한다. Provider
    실패는 원인과 무관하게 ``ClinicalChatProviderUnavailableError``로 바뀌고, 로그에는
    프롬프트·EMR·API Key 없이 분류된 사유만 남는다.
    """

    def __init__(
        self,
        provider: ProviderClient,
        reference_lookup: SqliteReferenceLookup | None = None,
        evidence: EvidenceRetriever | None = None,
        evidence_limit: int = 3,
        *, allow_general_knowledge: bool = False,
    ) -> None:
        self._provider = provider
        # False (attribution-only): recommendations may not add suggestions without a corpus source.
        self._allow_general_knowledge = allow_general_knowledge
        self._reference_lookup = reference_lookup
        self._evidence = evidence
        self._evidence_limit = evidence_limit

    def find_evidence(self, question: str, retriever: EvidenceRetriever | None = None) -> tuple[EvidenceChunk, ...]:
        """Corpus evidence to cite for ``question``; logs chunk ids only (no question text).

        ``retriever`` is the request's corpus snapshot; without it the service's own retriever is used.
        """
        if retriever is None:  # not ``or``: an empty index is falsy (``__len__``) but still the snapshot's index
            retriever = self._evidence
        if retriever is None:
            return ()
        try:
            chunks = retriever.search(question, self._evidence_limit)
        except Exception as error:  # evidence is optional; never block the answer
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
        evidence: Sequence[EvidenceChunk] | None = None, assessment: Mapping[str, Any] | None = None,
    ) -> str:
        """준비된 동일 환자의 EMR 문맥으로 한국어 답변을 생성한다."""
        with _provider_failure("prepared_snapshot_chat"):
            response = await self._provider.respond(self._chat_request(question, snapshot, reference_record, evidence, assessment))
        return response.answer

    async def stream_prepared_snapshot(
        self, *, question: str, snapshot: Mapping[str, Any], reference_record: str | None = None,
        evidence: Sequence[EvidenceChunk] | None = None, assessment: Mapping[str, Any] | None = None,
    ) -> AsyncIterator[str]:
        """Yield a prepared-snapshot answer as provider text becomes available."""
        with _provider_failure("prepared_snapshot_chat_stream"):
            async for delta in self._provider.stream(self._chat_request(question, snapshot, reference_record, evidence, assessment)):
                yield delta

    async def extract_findings(self, *, note: str, vocabulary: Sequence[str]) -> dict[str, bool]:
        """Map a clinician note onto rule finding keys (present → True, denied → False).

        A malformed provider answer yields no findings rather than failing the request;
        the caller returns ``findings_used`` so the clinician sees what was applied.
        """
        if not note.strip() or not vocabulary:
            return {}
        with _provider_failure("finding_extraction"):
            response = await self._provider.respond(OpenAIRequest(
                instructions=FINDING_EXTRACTION_INSTRUCTIONS,
                input=compose_finding_extraction(note, list(vocabulary)),
                output_schema=finding_extraction_schema(list(vocabulary)),
                output_schema_name="clinical_findings",
            ))
        try:
            parsed = json.loads(response.answer)
        except json.JSONDecodeError:
            logger.warning("finding_extraction_invalid reason=not_json")
            return {}
        allowed = set(vocabulary)
        findings: dict[str, bool] = {}
        if isinstance(parsed, Mapping):
            for key, value in (("present", True), ("absent", False)):
                for item in parsed.get(key) or ():
                    if isinstance(item, str) and item in allowed:
                        findings[item] = value
        return findings

    async def recommend(
        self, *, snapshot: Mapping[str, Any], assessment: Mapping[str, Any], evidence: Sequence[EvidenceChunk],
        clinical_note: str | None = None, question: str | None = None,
    ) -> str:
        with _provider_failure("clinical_recommendation"):
            response = await self._provider.respond(self._recommendation_request(snapshot, assessment, evidence, clinical_note, question))
        return response.answer

    async def stream_recommendation(
        self, *, snapshot: Mapping[str, Any], assessment: Mapping[str, Any], evidence: Sequence[EvidenceChunk],
        clinical_note: str | None = None, question: str | None = None,
    ) -> AsyncIterator[str]:
        with _provider_failure("clinical_recommendation_stream"):
            async for delta in self._provider.stream(self._recommendation_request(snapshot, assessment, evidence, clinical_note, question)):
                yield delta

    def _recommendation_request(
        self, snapshot: Mapping[str, Any], assessment: Mapping[str, Any], evidence: Sequence[EvidenceChunk],
        clinical_note: str | None, question: str | None,
    ) -> OpenAIRequest:
        return OpenAIRequest(
            instructions=CLINICAL_INSTRUCTIONS,
            input=compose_recommendation_input(
                snapshot, prompt_view(assessment), evidence_payload(evidence), clinical_note=clinical_note,
                question=question or RECOMMENDATION_REQUEST, allow_general_knowledge=self._allow_general_knowledge,
            ),
        )

    async def summarize_prepared_snapshot(self, *, snapshot: Mapping[str, Any]) -> PreparedPatientSummary:
        """준비 시점에 EMR 문맥으로 화면용 구조화 환자 상태 요약을 생성한다."""
        with _provider_failure("patient_context_summary"):
            response = await self._provider.respond(OpenAIRequest(
                instructions=CLINICAL_INSTRUCTIONS,
                input=compose_prepared_snapshot_question(INITIAL_PATIENT_SUMMARY_REQUEST, snapshot),
                output_schema=PATIENT_SUMMARY_OUTPUT_SCHEMA,
            ))
            return PreparedPatientSummary.from_provider_text(response.answer)

    async def summarize_once(self, *, snapshot: Mapping[str, Any]) -> str:
        """EMR에서 독립적인 간결 요약을 한 번 생성한다."""
        with _provider_failure("one_shot_patient_summary"):
            response = await self._provider.respond(self._summary_request(snapshot))
        return response.answer

    async def stream_summary_once(self, *, snapshot: Mapping[str, Any]) -> AsyncIterator[str]:
        """Yield the one-shot EMR summary as provider text becomes available.

        Unlike ``summarize_prepared_snapshot`` this has no JSON schema, so the
        first sentence can be shown before the whole summary is generated.
        """
        with _provider_failure("one_shot_patient_summary_stream"):
            async for delta in self._provider.stream(self._summary_request(snapshot)):
                yield delta

    def _chat_request(
        self, question: str, snapshot: Mapping[str, Any], reference_record: str | None,
        evidence: Sequence[EvidenceChunk] | None, assessment: Mapping[str, Any] | None = None,
    ) -> OpenAIRequest:
        references = self._reference_lookup.search_question(question) if self._reference_lookup else None
        chunks = self.find_evidence(question) if evidence is None else evidence
        return OpenAIRequest(
            instructions=CLINICAL_INSTRUCTIONS,
            input=compose_prepared_snapshot_question(
                question, snapshot, reference_record, references, evidence=evidence_payload(chunks),
                assessment=prompt_view(assessment) if assessment else None,
            ),
        )

    @staticmethod
    def _summary_request(snapshot: Mapping[str, Any]) -> OpenAIRequest:
        return OpenAIRequest(
            instructions=CLINICAL_INSTRUCTIONS,
            input=compose_prepared_snapshot_question(ONE_SHOT_PATIENT_SUMMARY_REQUEST, snapshot, one_shot_summary=True),
        )


def evidence_payload(chunks: Sequence[EvidenceChunk]) -> list[dict[str, object]]:
    """Numbered evidence for the prompt and the API response (same numbering).

    ``priority: reference`` marks a foreign passage superseded by a domestic standard on the same topic.
    """
    return [{"n": number, **chunk.as_dict()} for number, chunk in enumerate(chunks, start=1)]


@contextmanager
def _provider_failure(operation: str) -> Iterator[None]:
    """Log a classified provider failure without prompts, EMR, or API keys."""
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
