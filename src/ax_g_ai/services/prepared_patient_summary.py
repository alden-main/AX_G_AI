"""환자 EMR 준비 완료 메시지의 화면용 구조화 계약."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Mapping

from ax_g_ai.adapters.openai import OpenAIProviderError


@dataclass(frozen=True)
class PreparedPatientSummarySection:
    """채팅 말풍선에서 라벨과 함께 표시할 EMR 요약 항목."""

    label: str
    content: str

    def as_dict(self) -> dict[str, str]:
        return {"label": self.label, "content": self.content}


@dataclass(frozen=True)
class PreparedPatientSummary:
    """환자 선택 직후 표시하는 최초 AI 메시지의 안정된 형태."""

    message: str
    sections: tuple[PreparedPatientSummarySection, ...]
    guidance: str

    def as_dict(self) -> dict[str, object]:
        return {
            "message": self.message,
            "sections": [section.as_dict() for section in self.sections],
            "guidance": self.guidance,
        }

    @classmethod
    def from_provider_text(cls, text: str) -> "PreparedPatientSummary":
        """Responses Structured Output을 검증해 내부 DTO로 바꾼다."""
        try:
            value = json.loads(text)
        except json.JSONDecodeError as error:
            raise OpenAIProviderError("OpenAI returned invalid patient summary JSON") from error
        if not isinstance(value, Mapping):
            raise OpenAIProviderError("OpenAI returned an invalid patient summary shape")
        message = _required_text(value, "message")
        guidance = _required_text(value, "guidance")
        raw_sections = value.get("sections")
        if not isinstance(raw_sections, list):
            raise OpenAIProviderError("OpenAI returned invalid patient summary sections")
        sections = tuple(
            PreparedPatientSummarySection(
                label=_required_text(item, "label"),
                content=_required_text(item, "content"),
            )
            for item in raw_sections
            if isinstance(item, Mapping)
        )
        if len(sections) != len(raw_sections):
            raise OpenAIProviderError("OpenAI returned invalid patient summary sections")
        return cls(message=message, sections=sections, guidance=guidance)


def _required_text(value: Mapping[str, Any], key: str) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item.strip():
        raise OpenAIProviderError("OpenAI returned an invalid patient summary field")
    return item.strip()


PATIENT_SUMMARY_OUTPUT_SCHEMA: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "message": {
            "type": "string",
            "description": "첫 문장. 예: EMR 데이터가 로드되었습니다.",
        },
        "sections": {
            "type": "array",
            "description": "표시할 수 있는 EMR 항목만 담는다.",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "label": {"type": "string", "description": "예: 최근 혈압, 최근 검사, 복용 약물"},
                    "content": {"type": "string", "description": "해당 항목의 간결한 한국어 내용"},
                },
                "required": ["label", "content"],
            },
        },
        "guidance": {
            "type": "string",
            "description": "의료진 판단 보조 및 후속 확인을 안내하는 한두 문장",
        },
    },
    "required": ["message", "sections", "guidance"],
}
