"""준비된 비식별 EMR 문맥으로 Provider를 호출하는 의료진 챗봇 서비스."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from contextlib import contextmanager
import logging
from typing import Any, Mapping

from ax_g_ai.adapters.openai import OpenAIProviderError, OpenAIRequest, ProviderClient
from ax_g_ai.errors import ClinicalChatProviderUnavailableError
from ax_g_ai.prompts.clinical import (
    CLINICAL_INSTRUCTIONS,
    INITIAL_PATIENT_SUMMARY_REQUEST,
    ONE_SHOT_PATIENT_SUMMARY_REQUEST,
    compose_prepared_snapshot_question,
)
from ax_g_ai.services.prepared_patient_summary import (
    PATIENT_SUMMARY_OUTPUT_SCHEMA,
    PreparedPatientSummary,
)
from ax_g_ai.services.reference_lookup import SqliteReferenceLookup

logger = logging.getLogger("uvicorn.error")


class ClinicalChatService:
    """Provider 호출과 실패 분류를 담당한다.

    ``snapshot``은 ``RawEmrPayloadMapper``가 비식별 처리한 문맥이어야 한다. Provider
    실패는 원인과 무관하게 ``ClinicalChatProviderUnavailableError``로 바뀌고, 로그에는
    프롬프트·EMR·API Key 없이 분류된 사유만 남는다.
    """

    def __init__(self, provider: ProviderClient, reference_lookup: SqliteReferenceLookup | None = None) -> None:
        self._provider = provider
        self._reference_lookup = reference_lookup

    async def answer_prepared_snapshot(
        self, *, question: str, snapshot: Mapping[str, Any], reference_record: str | None = None
    ) -> str:
        """준비된 동일 환자의 EMR 문맥으로 한국어 답변을 생성한다."""
        with _provider_failure("prepared_snapshot_chat"):
            response = await self._provider.respond(self._chat_request(question, snapshot, reference_record))
        return response.answer

    async def stream_prepared_snapshot(
        self, *, question: str, snapshot: Mapping[str, Any], reference_record: str | None = None
    ) -> AsyncIterator[str]:
        """Yield a prepared-snapshot answer as provider text becomes available."""
        with _provider_failure("prepared_snapshot_chat_stream"):
            async for delta in self._provider.stream(self._chat_request(question, snapshot, reference_record)):
                yield delta

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
        self, question: str, snapshot: Mapping[str, Any], reference_record: str | None
    ) -> OpenAIRequest:
        references = self._reference_lookup.search_question(question) if self._reference_lookup else None
        return OpenAIRequest(
            instructions=CLINICAL_INSTRUCTIONS,
            input=compose_prepared_snapshot_question(question, snapshot, reference_record, references),
        )

    @staticmethod
    def _summary_request(snapshot: Mapping[str, Any]) -> OpenAIRequest:
        return OpenAIRequest(
            instructions=CLINICAL_INSTRUCTIONS,
            input=compose_prepared_snapshot_question(ONE_SHOT_PATIENT_SUMMARY_REQUEST, snapshot),
        )


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
