"""채팅 답변이나 추천이 기대는 근거(인용 근거 문단과 규칙 평가 결과)를 모으는 모듈.

설계상 위치: 서비스 계층에서 API 라우트와 ``ClinicalChatService`` 사이. 라우트가 채팅·추천 요청마다 이 모듈로
근거를 마련한 뒤 결과를 답변 메서드에 넘긴다. 둘 다 요청의 말뭉치 스냅샷(``CorpusSnapshot``)을 쓰므로
근거 문단·규칙 결과·DUR이 한 릴리스에서 온다.

역할:
- 채팅(``chat_grounding``): 기록된 값만 묻는 질문(``question_facts.is_record_lookup``)이 아니면 질문의 근거를 찾는다.
  진단·검사·약·용량을 묻는 질문(또는 약 이름을 대거나 임신을 언급한 질문)에는 2·3단계 규칙 결과도 붙이며,
  그 근거를 앞에 두어 ``[근거 n]`` 참조가 유효하게 한다.
- 추천(``recommendation_grounding``): 의료진의 명시적 소견 > 메모(LLM 추출 > 결정적 판독) > 질문 순으로 소견을
  합친 뒤 전체 규칙 평가를 만든다.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

from ax_g_ai.errors import RecommendationUnavailableError
from ax_g_ai.services.clinical_chat import ClinicalChatService
from ax_g_ai.services.clinical_evidence import EvidenceChunk
from ax_g_ai.services.clinical_recommendation import Assessment, ClinicalRecommender
from ax_g_ai.services.corpus_snapshot import CorpusSnapshot
from ax_g_ai.services.patient_context_store import PreparedRawEmrContext
from ax_g_ai.services.question_facts import (
    asks_for_clinical_rules,
    asks_for_recommendation,
    asks_only_for_dur,
    is_record_lookup,
    merge_question_findings,
    question_findings,
)

logger = logging.getLogger("uvicorn.error")

# 채팅 답변 자체의 근거 앞에 놓는 규칙 근거 문단 수.
RULE_EVIDENCE_IN_CHAT = 4


@dataclass(frozen=True)
class Grounding:
    """답변 하나가 기대는 근거 묶음.

    역할: 프롬프트와 API 응답에 함께 실을 인용 근거 문단과 규칙 평가 결과(쓰지 않으면 ``None``)를 담는다.
    """

    evidence: tuple[EvidenceChunk, ...]
    assessment: dict[str, object] | None  # 평가 결과 payload(``docs/assessment_schema.md``), None: 사용 안 함


def require_recommender(snapshot: CorpusSnapshot) -> ClinicalRecommender:
    """스냅샷의 규칙 추천기를 돌려준다. 없거나 규칙이 비어 있으면 ``RecommendationUnavailableError``."""
    if snapshot.recommender is None or len(snapshot.recommender) == 0:
        raise RecommendationUnavailableError()
    return snapshot.recommender


def with_rule_input(context: PreparedRawEmrContext, snapshot: CorpusSnapshot) -> PreparedRawEmrContext:
    """이 스냅샷의 규칙으로 매핑한 규칙 사실을 가진 문맥을 돌려준다.

    역할: 추천기가 없거나 이미 같은 릴리스로 매핑돼 있으면 그대로 두고, 아니면 ``rule_source``를 다시 매핑한다.
        매핑이 실패하면 경고 로그를 남기고 ``rule_input``을 ``None``으로 둔다(규칙 사실은 선택 사항).
    입력:
        context: 준비된 EMR 문맥.
        snapshot: 요청의 말뭉치 스냅샷.
    출력: ``rule_input``·``rule_release``가 갱신된(또는 그대로인) 문맥.
    """
    recommender = snapshot.recommender
    if recommender is None or (context.rule_input is not None and context.rule_release == snapshot.release_id):
        return context
    try:
        rule_input = recommender.prepare(context.rule_source)
    except Exception as error:  # 규칙 사실은 선택 사항이므로 EMR 준비나 채팅을 절대 막지 않는다
        logger.warning("clinical_rule_mapping_failed error_type=%s", type(error).__name__)
        return replace(context, rule_input=None)
    return replace(context, rule_input=rule_input, rule_release=snapshot.release_id)


def chat_grounding(question: str, context: PreparedRawEmrContext, service: ClinicalChatService, snapshot: CorpusSnapshot) -> Grounding:
    """채팅 질문 하나의 근거(진단·검사·약 질문에는 규칙 결과도)를 마련한다.

    역할: 기록 조회 질문이 아니면 근거를 검색하고, 결과가 없는 일반 추천 질문은 기록된 진단명을 덧붙여 다시 찾는다.
        추천기와 규칙 사실이 있고 질문에 규칙 입력(소견·약)이 있거나 추천을 묻는 경우 규칙을 평가하며,
        항목·검사 해석·DUR 경고·용량 결과 중 하나라도 있으면 규칙 근거 앞 ``RULE_EVIDENCE_IN_CHAT``개를 검색 근거 앞에 두고
        평가 결과의 근거 참조를 그 범위로 줄인다. 진단·검사·협진 질문(또는 임신 밖의 소견을 말한 질문)이 아니면
        DUR 점검 결과와 용량 결과만 싣고, 규칙 입력도 약·처방 질문도 아니면 현재 처방의 DUR 경고만 싣는다.
        서비스가 DUR만 묻는 질문(``asks_only_for_dur``)을 직접 답하면(``compact``) 근거를 검색하지 않는다.
        동기 함수다. 검색이 임베딩 서버와 데이터베이스를 부를 수 있으므로 호출자가 스레드에서 실행한다.
    입력:
        question: 의료진 질문.
        context: 준비된 EMR 문맥(``rule_input`` 포함).
        service: 근거 검색에 쓰는 채팅 서비스.
        snapshot: 요청의 말뭉치 스냅샷.
    출력: ``Grounding``. 규칙 결과를 쓰지 않으면 ``assessment``는 ``None``.
    """
    searched: tuple[EvidenceChunk, ...] = ()
    # compact 프로필은 DUR 점검만 묻는 질문을 서버가 DUR 점검 블록으로 답하므로(``ClinicalChatService``) 근거 문단이 필요 없다.
    if not is_record_lookup(question) and not (service.answers_dur_only_itself and asks_only_for_dur(question)):
        searched = service.find_evidence(question, snapshot.evidence)
        if not searched and asks_for_recommendation(question):
            # 일반적인 추천 질문("추천 검사는?")에는 주제어가 없으므로 기록된 진단명을 더한다.
            # 이 경우에만 그렇게 한다. 다른 질문에서는 진단명이 질문과 무관한 문단을 끌어온다.
            hint = diagnosis_hint(context)
            if hint:
                searched = service.find_evidence(f"{question} {hint}", snapshot.evidence)
    recommender = snapshot.recommender
    if recommender is None or context.rule_input is None:
        return Grounding(searched, None)
    stated = recommender.question_inputs(question)
    if not stated and not asks_for_recommendation(question):
        # 규칙 입력이 없는 질문(SOAP·요약 등)에도 현재 처방의 DUR 경고는 싣는다. 규칙 항목은 싣지 않는다
        # (질문과 무관한 possible 항목이 환자 사실처럼 답변에 섞이기 때문이다).
        medication_alerts = recommender.check_medications(context.rule_input)
        if medication_alerts is None or not medication_alerts.alerts:
            return Grounding(searched, None)
        return Grounding(searched, {"medication_alerts": medication_alerts.model_dump(mode="json")})
    assessment = recommender.assess(context.rule_input, stated.findings, stated.dose_drugs, stated.proposed_drugs)
    if not asks_for_clinical_rules(question) and set(stated.findings) <= {"pregnancy"}:
        # 약·처방·임신·용량 질문: DUR·용량 결과만 싣는다(진단·검사 규칙 항목과 그 근거 문단은 싣지 않는다).
        payload = _medication_view(assessment)
        return Grounding(searched, payload or None)
    alerts = assessment.medication_alerts.alerts if assessment.medication_alerts is not None else None
    if not assessment.items and not assessment.lab_interpretations and not alerts and not assessment.dose_assessments:
        return Grounding(searched, None)
    rule_chunks = assessment.evidence[:RULE_EVIDENCE_IN_CHAT]
    return Grounding(merge_evidence(rule_chunks, searched), _renumber(assessment, len(rule_chunks)))


async def recommendation_grounding(
    context: PreparedRawEmrContext, service: ClinicalChatService, snapshot: CorpusSnapshot, *,
    clinical_note: str | None, question: str | None, findings: Mapping[str, bool] | None, dose_drugs: Sequence[str] | None,
) -> tuple[PreparedRawEmrContext, Grounding]:
    """추천 요청의 전체 규칙 평가와 근거를 마련한다.

    역할: 스냅샷 규칙으로 규칙 사실을 매핑(실패하면 EMR 사실 없이)한 뒤, 메모 소견 추출(LLM)과 근거 검색을 동시에
        실행한다. 소견은 명시적 입력 > 메모(추출 > 결정적 판독) > 질문 순으로 합치고, 상태 변화 표현 없이 엇갈리면
        충돌로 넘긴다. 용량 약은 입력과 텍스트에서 찾은 것을 순서 유지·중복 제거해 합친다. 규칙 근거 뒤에 검색 근거를 붙인다.
    입력:
        context: 준비된 EMR 문맥.
        service: 소견 추출·근거 검색에 쓰는 채팅 서비스.
        snapshot: 요청의 말뭉치 스냅샷.
        clinical_note: 의료진 메모(없으면 ``None``).
        question: 의료진 질문(없으면 ``None``).
        findings: 의료진이 명시한 소견(키 → True/False).
        dose_drugs: 용량 평가를 요청한 약 목록.
    출력: (규칙 사실이 갱신된 문맥, ``Grounding``) 튜플.
    예외: 스냅샷에 규칙이 없으면 ``RecommendationUnavailableError``. 소견 추출 Provider 실패는
        ``ClinicalChatProviderUnavailableError``로 전파된다.
    """
    rules = require_recommender(snapshot)
    context = with_rule_input(context, snapshot)
    if context.rule_input is None:
        # EMR 매핑 실패: EMR 사실 없이 평가한다.
        context = replace(context, rule_input=rules.prepare({}))
    query = " ".join(part for part in (question, clinical_note, diagnosis_hint(context)) if part)
    # 메모 추출(LLM 호출)과 근거 검색은 서로 의존하지 않으므로 함께 실행한다.
    extracted, searched = await asyncio.gather(
        service.extract_findings(note=clinical_note, vocabulary=rules.finding_vocabulary) if clinical_note else _value({}),
        asyncio.to_thread(service.find_evidence, query, snapshot.evidence) if query else _value(()),
    )
    stated = rules.question_inputs(clinical_note, question)
    # 의료진의 명시적 입력이 메모(추출 결과가 결정적 판독보다 우선)보다 우선하고, 메모는 질문 텍스트보다 우선한다.
    # 단, 질문이 메모 이후의 변화("출산 후")를 말하면 질문이 이긴다. 그런 표현 없이 엇갈리면 미상으로 두고
    # 확인할 충돌로 보고한다.
    note_findings, conflicts = merge_question_findings({**question_findings(clinical_note), **extracted}, question)
    merged: dict[str, bool] = {**note_findings, **(findings or {})}
    drugs = tuple(dict.fromkeys((*(dose_drugs or ()), *stated.dose_drugs)))
    assessment = rules.assess(context.rule_input, merged, drugs, stated.proposed_drugs, conflicts)  # type: ignore[arg-type]
    evidence = merge_evidence(assessment.evidence, searched)
    logger.info(
        "clinical_recommendation_assessed items=%d met=%d findings=%d unknown_findings=%d evidence=%d unreviewed=%s doses=%d",
        len(assessment.items), sum(1 for item in assessment.items if item.status == "met"),
        len(assessment.findings_used), len(assessment.unknown_findings), len(evidence), assessment.unreviewed,
        len(assessment.dose_assessments or ()),
    )
    return context, Grounding(evidence, assessment.as_dict())


def diagnosis_hint(context: PreparedRawEmrContext) -> str:
    """주제어 없는 질문에 덧붙일 검색어로, 기록된 진단 한글명을 공백으로 이은 문자열(최대 300자)."""
    diagnoses = context.data.get("diagnoses")
    names = [item.get("name_ko") for item in diagnoses if isinstance(item, Mapping)] if isinstance(diagnoses, list) else []
    return " ".join(name for name in names if isinstance(name, str))[:300]


def merge_evidence(first: Sequence[EvidenceChunk], second: Sequence[EvidenceChunk]) -> tuple[EvidenceChunk, ...]:
    """``first``의 번호를 유지한 채 ``second`` 중 겹치지 않는 문단을 뒤에 잇는다(규칙 근거 참조가 ``first``를 가리킴)."""
    seen = {chunk.chunk_id for chunk in first}
    return (*first, *(chunk for chunk in second if chunk.chunk_id not in seen))


def _medication_view(assessment: Assessment) -> dict[str, object]:
    """평가 결과에서 DUR 점검 결과와 용량 결과만 남긴 payload(규칙 근거 문단을 싣지 않으므로 근거 번호는 뺀다).

    역할: 용량 항목은 근거 번호 대신 출처명으로 출처를 밝힌다. 둘 다 없으면 빈 사전.
    """
    payload = assessment.as_dict()
    view = {key: payload[key] for key in ("medication_alerts", "dose_assessments", "unmatched_dose_drugs") if payload.get(key)}
    doses = view.get("dose_assessments")
    for item in doses if isinstance(doses, list) else ():
        item.pop("evidence_refs", None)
    return view


def _renumber(assessment: Assessment, kept: int) -> dict[str, object]:
    """근거 참조가 앞의 ``kept``개 문단만 가리키도록 줄인 평가 결과 payload."""
    payload = assessment.as_dict()
    for group in ("items", "lab_interpretations", "dose_assessments"):
        items = payload.get(group)
        for item in items if isinstance(items, list) else ():
            item["evidence_refs"] = [ref for ref in item.get("evidence_refs", ()) if ref <= kept]
    return payload


async def _value[T](value: T) -> T:
    """``asyncio.gather``에 넣을 수 있도록 값을 그대로 돌려주는 코루틴."""
    return value
