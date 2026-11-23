"""임상 추천기의 입력·결과 타입과 인용 근거 문단 번호 매기기.

설계상 위치: 임상 추천 패키지의 데이터 모델 모듈. ``ClinicalRecommender``(``__init__``), ``rules``, ``presentation``,
``doses``가 이 타입을 주고받고, 채팅·추천 API가 ``Assessment.payload``와 ``prompt_view``로 응답·프롬프트를 만든다.

역할: 조건 잎(``Leaf``), 환자별 규칙 입력(``RuleInput``), 평가 결과(``Assessment``)를 정의하고, 평가 한 번에서
인용한 근거 문단에 처음 쓰인 순서대로 번호를 붙인다(``EvidenceNumbering``). LLM용 페이로드 축약(``prompt_view``)과
누락 입력 중복 제거(``unique_missing``)도 여기 있다.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any

from ax_g_ai.services.clinical_evidence import EvidenceChunk
from ax_g_ai.services.clinical_recommendation.schema import (
    AssessmentPayload,
    DoseAssessment,
    FindingConflict,
    LabInterpretation,
    MappingIssue,
    MedicationAlerts,
    MissingInput,
    RecommendationItem,
    UnmatchedDoseDrug,
)
from ax_g_ai.services.emr_mapping import EmrMapping


@dataclass(frozen=True)
class Leaf:
    """규칙 조건 트리의 잎 조건 하나와 이 환자에 대한 평가 결과.

    역할: ``rules.condition_leaves``가 만들며, 충족 조건(``matched``)·누락 입력(``missing``) 표시와 근거점 판정,
    항목 정렬에 쓰인다. ``result``는 3값(참·거짓·``None``=알 수 없음), ``negated``는 ``not`` 아래에 있는지 여부다.
    """

    kind: str
    key: str
    condition: Mapping[str, Any]
    result: bool | None
    negated: bool


@dataclass(frozen=True)
class RuleInput:
    """규칙 평가용 EMR 사실. 환자 맥락을 준비할 때 한 번 계산한다.

    역할: ``ClinicalRecommender.prepare``의 결과이자 ``assess``의 입력. 개념 ID·값·KCD 코드·성분 ID만 담고
    환자 식별 정보는 담지 않는다.
    """

    mapping: EmrMapping
    sex: str | None = None
    # 현재 처방의 (medCode, medName) — DUR 점검용 제품코드
    medications: tuple[tuple[str | None, str | None], ...] = ()


@dataclass(frozen=True)
class Assessment:
    """환자 한 명의 규칙 평가 결과: 페이로드 구성 요소(``schema`` 모델)와 그것들이 인용한 근거 문단.

    역할: ``ClinicalRecommender.assess``의 결과. ``evidence``의 번호는 각 항목의 ``evidence_refs``와 맞는다.
    ``payload``가 구성 요소로 API·프롬프트용 페이로드를 조립한다.
    """

    items: tuple[RecommendationItem, ...]
    lab_interpretations: tuple[LabInterpretation, ...]
    evidence: tuple[EvidenceChunk, ...]
    findings_used: Mapping[str, bool]
    mapping_issues: tuple[MappingIssue, ...] = ()
    unknown_findings: tuple[str, ...] = ()
    # 현재 처방에 대한 심평원 DUR 금기·주의 경고(None: DUR 미설정)
    medication_alerts: MedicationAlerts | None = None
    # 3단계 용량 결과(None: 용량 단계 꺼짐)와 요청했지만 용량 규칙이 없는 약
    dose_assessments: tuple[DoseAssessment, ...] | None = None
    unmatched_dose_drugs: tuple[UnmatchedDoseDrug, ...] = ()
    # 메모와 질문이 서로 다르게 적어 findings_used에서 뺀 소견: {"pregnancy": {"clinical_note": True, "question": False}}
    finding_conflicts: Mapping[str, Mapping[str, bool]] = field(default_factory=dict)

    @property
    def unreviewed(self) -> bool:
        """항목·검사값 해석·용량 결과 중 승인되지 않은(``approved``가 아닌) 것이 있으면 True."""
        return (any(item.review_status != "approved" for item in self.items)
                or any(lab.review_status != "approved" for lab in self.lab_interpretations)
                or any(dose.review_status != "approved" for dose in self.dose_assessments or ()))

    @cached_property
    def payload(self) -> AssessmentPayload:
        """구성 요소로 ``schema.AssessmentPayload``를 조립한다(한 번 만들고 캐시).

        역할: 기본 구성 요소는 그대로 넣고, 선택 묶음(DUR 경고, 용량 결과, 미매칭 용량 약, 소견 충돌)은 해당할 때만
        넣어 직렬화에서 빠지게 한다.
        입력: 없음.
        출력: ``AssessmentPayload``.
        """
        optional: dict[str, Any] = {}
        if self.medication_alerts is not None:
            optional["medication_alerts"] = self.medication_alerts
        if self.dose_assessments is not None:
            optional["dose_assessments"] = list(self.dose_assessments)
            if self.unmatched_dose_drugs:
                optional["unmatched_dose_drugs"] = list(self.unmatched_dose_drugs)
        if self.finding_conflicts:
            # ``question_facts.merge_question_findings``가 만든 값(채팅 경로와 함께 쓰는 일반 데이터).
            optional["finding_conflicts"] = {key: FindingConflict.model_validate(value) for key, value in self.finding_conflicts.items()}
        return AssessmentPayload(
            items=list(self.items), lab_interpretations=list(self.lab_interpretations), findings_used=dict(self.findings_used),
            unknown_findings=list(self.unknown_findings), mapping_issues=list(self.mapping_issues), **optional,
        )

    def as_dict(self) -> dict[str, object]:
        """지정된 키만 담은 JSON용 페이로드(매번 새 dict라 호출자가 고쳐도 된다)."""
        return self.payload.model_dump(mode="json")


# 답변은 내용을 출처(source_titles, region_label)로 밝히므로 ID와 검수 상태는 되풀이하지 않는다.
_PROMPT_DROP = frozenset({"claim_id", "chapter", "review_status", "record_id", "mapping_issues"})


def prompt_view(payload: Mapping[str, Any]) -> dict[str, Any]:
    """평가 페이로드를 LLM 프롬프트용으로 줄인다.

    역할: ID·검수 상태(``_PROMPT_DROP``)와 빈 값을 빼고, 화면에 보이는 전체 출처 레코드(``sources``)를 중복 없는
    ``source_titles``로 줄여 답변이 각 내용의 출처를 말할 수 있게 한다. 중첩된 dict·list에도 재귀로 적용한다.
    입력:
        payload: ``Assessment.as_dict()`` 형태의 평가 페이로드.
    출력: 축약한 새 dict.
    """

    def strip(value: Any) -> Any:
        if isinstance(value, Mapping):
            result = {}
            for key, item in value.items():
                if key in _PROMPT_DROP or item in (None, [], {}):
                    continue
                if key == "sources" and isinstance(item, list):
                    titles = [source["title"] for source in item if isinstance(source, Mapping) and source.get("title")]
                    if titles:
                        result["source_titles"] = list(dict.fromkeys(titles))
                    continue
                result[key] = strip(item)
            return result
        if isinstance(value, list):
            return [strip(item) for item in value]
        return value

    return strip(payload)


@dataclass
class EvidenceNumbering:
    """평가 한 번에서 인용한 근거 문단과, 처음 쓰인 순서대로 매긴 번호(``evidence_refs`` = [근거 n]).

    역할: 2단계 항목·검사값 해석·용량 항목이 같은 번호 체계를 공유하도록 문단 목록과 ``chunk_id`` → 번호를 누적한다.
    평가마다 새로 만든다.
    """

    chunks_by_claim: Mapping[str, tuple[EvidenceChunk, ...]]
    max_evidence: int
    per_item: int = 1
    evidence: list[EvidenceChunk] = field(default_factory=list)
    numbers: dict[str, int] = field(default_factory=dict)

    def refs(self, claim_id: object, limit: int | None = None) -> list[int]:
        """claim의 근거 문단에 번호를 붙이고 그 번호들을 돌려준다.

        역할: claim의 문단을 ``per_item``개까지 보고, 이미 번호가 있으면 재사용하고, 없으면 상한 안에서 새로 추가한다.
        상한을 넘는 새 문단은 건너뛴다.
        입력:
            claim_id: 규칙 레코드의 ``claim_id``. 문자열이 아니면 빈 목록.
            limit: 전체 문단 수 상한. ``None``이면 ``max_evidence``.
        출력: 근거 번호 목록.
        """
        refs: list[int] = []
        if not isinstance(claim_id, str):
            return refs
        for chunk in self.chunks_by_claim.get(claim_id, ())[: self.per_item]:
            number = self.numbers.get(chunk.chunk_id)
            if number is None:
                if len(self.evidence) >= (self.max_evidence if limit is None else limit):
                    continue
                self.evidence.append(chunk)
                number = self.numbers[chunk.chunk_id] = len(self.evidence)
            refs.append(number)
        return refs


def unique_missing(items: Iterable[MissingInput]) -> list[MissingInput]:
    """(kind, key)마다 처음 나온 것만 처음 순서대로 남긴다. 한 규칙이 같은 입력을 여러 조건에서 검사할 수 있기 때문이다."""
    unique: dict[tuple[str, str], MissingInput] = {}
    for item in items:
        unique.setdefault((item.kind, item.key), item)
    return list(unique.values())
