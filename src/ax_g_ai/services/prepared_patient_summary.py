"""환자 EMR 준비 완료 메시지의 화면용 구조화 계약.

같은 Pydantic 모델이 Provider에 보내는 Structured Output JSON Schema, 응답 검증, API 응답
본문을 모두 정의한다. 문자열 공백 검증은 ``AfterValidator``로 두어 OpenAI strict 모드가
지원하지 않는 ``minLength`` 키워드가 스키마에 들어가지 않게 한다.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, ValidationError

from ax_g_ai.services.ports import OpenAIProviderError


def _non_blank(value: str) -> str:
    value = value.strip()
    if not value:
        raise ValueError("must not be blank")
    return value


NonBlankText = Annotated[str, AfterValidator(_non_blank)]


class PreparedPatientSummarySection(BaseModel):
    """채팅 말풍선에서 라벨과 함께 표시할 EMR 요약 항목."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    label: NonBlankText = Field(description="예: 최근 혈압, 최근 검사, 복용 약물")
    content: NonBlankText = Field(
        description="해당 항목의 사실을 자연스럽게 연결한 읽기 쉬운 완결 한국어 문장. 단순 나열이나 데이터 부재 안내는 넣지 않는다.",
    )


class PreparedPatientSummary(BaseModel):
    """환자 선택 직후 표시하는 최초 AI 메시지의 안정된 형태."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    message: NonBlankText = Field(description="EMR에서 확인되는 환자 상태의 한 문장 요약.")
    sections: tuple[PreparedPatientSummarySection, ...] = Field(description="EMR 사실을 한 줄씩 담는 추가 요약 항목. 기록이 부족하면 항목 수를 줄인다.")
    guidance: NonBlankText = Field(
        description="추가 요약의 마지막 한 줄. EMR에서 확인되는 사실을 담는다.",
    )

    def as_dict(self) -> dict[str, object]:
        return self.model_dump(mode="json")

    @classmethod
    def from_provider_text(cls, text: str) -> "PreparedPatientSummary":
        """Structured Output을 검증해 내부 DTO로 바꾼다."""
        try:
            return cls.model_validate_json(text)
        except ValidationError as error:
            raise OpenAIProviderError("Provider returned an invalid patient summary") from error


PATIENT_SUMMARY_OUTPUT_SCHEMA: dict[str, object] = PreparedPatientSummary.model_json_schema()
