"""말뭉치 활용 2단계(진단·검사·협진·추적 규칙)와 3단계(용량)를 EMR 사실에 적용하는 임상 추천기.

설계상 위치: 서비스 계층의 임상 추천 패키지 진입점. 채팅·추천 API가 환자 맥락을 준비할 때 ``ClinicalRecommender``를
만들어 ``prepare``(EMR 매핑)와 ``assess``(규칙 평가)를 호출하고, 결과 ``Assessment``를 API 응답과 LLM 프롬프트에 쓴다.

역할: 규칙은 근거 문단과 같은 생성 말뭉치(``text_data/corpus`` 또는 활성 PostgreSQL 릴리스)에서 오고, 인용 근거와
같은 관문(``clinical_evidence.is_citable``)을 통과한다. 모든 출처가 상업적 AI 사용 허가를 받았고, 원문과 대조됐으며,
폐기되지 않았어야 한다. 의료진이 승인한 규칙은 항상 쓰고, 초안 규칙은 ``include_unreviewed``가 켜졌을 때만 쓴다.
모든 결과에는 출처와 ``review_status``가 붙는다(데이터로만 — 경고 문구 없음: 이 서비스는 설계상 참고 도구다).

3단계(``doses=True``): ``dose_rules``를 현재 처방과 의료진이 지정한 약(``dose_drugs``)에 적용한다 — 체중 기반 공식,
환자가 속한 신기능 구간, 용량 조건 규칙, 처방 1일 용량과 구간의 수치 최대량 대조(``clinical_dosing``). 값은 참고
계산이며 항상 입력·공식·출처를 함께 담는다. 치료 선택(``treatment.jsonl``)은 다루지 않는다.

평가는 3값 논리다(``clinical_corpus.evaluate``). 규칙은 다음과 같이 보고된다.
- ``met``: 가진 사실로 조건이 참이다.
- ``possible``: 조건은 알 수 없지만 일부가 이미 이 환자에게 참이다(기록된 진단, 처방, 검사값 같은 근거점).
  알 수 없는 입력은 ``missing``으로 돌려주어 의료진이 확인하게 한다.
거짓인 규칙과, 알 수 없으면서 환자와 이어지는 근거점이 없는 규칙은 버린다.

모듈: ``corpus``(관문을 통과한 레코드), ``rules``(2단계 판정), ``doses``(3단계), ``presentation``(페이로드 항목과
표시 이름), ``model``(입력·결과 타입).
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from ax_g_ai.corpus.files import DOSE_FILE
from ax_g_ai.services.clinical_calculations import body_mass_index
from ax_g_ai.services.clinical_corpus import CorpusRuleError
from ax_g_ai.services.clinical_recommendation.corpus import RuleCorpus, read_rows
from ax_g_ai.services.clinical_recommendation.doses import DoseStage
from ax_g_ai.services.clinical_recommendation.model import Assessment, EvidenceNumbering, Leaf, RuleInput, prompt_view
from ax_g_ai.services.clinical_recommendation.presentation import Presenter
from ax_g_ai.services.clinical_recommendation.rules import (
    FLAG_KINDS,
    accepted_findings,
    derive_facts,
    evaluate_record,
    flag_vocabulary,
    rank_item,
)
from ax_g_ai.services.clinical_recommendation.schema import DoseAssessment, LabInterpretation, RecommendationItem, UnmatchedDoseDrug
from ax_g_ai.services.clinical_recommendation.schema import MappingIssue as PayloadMappingIssue
from ax_g_ai.services.emr_mapping import SiteCodeMap, map_emr, normalize_name
from ax_g_ai.services.question_facts import has_medication_context, question_findings

if TYPE_CHECKING:  # dur_check가 이 패키지의 schema를 가져오므로 실행 시점에 가져오면 순환 import가 된다
    from ax_g_ai.services.dur_check import DurChecker, ProposedDrug

__all__ = ["FLAG_KINDS", "Assessment", "ClinicalRecommender", "Leaf", "QuestionInputs", "RuleCorpus", "RuleInput", "prompt_view", "read_rows"]

logger = logging.getLogger("uvicorn.error")


@dataclass(frozen=True)
class QuestionInputs:
    """채팅 질문이나 진료 메모가 규칙·DUR·용량 점검을 위해 EMR에 더하는 정보.

    역할: ``ClinicalRecommender.question_inputs``의 결과로, ``assess``의 ``findings``·``proposed_drugs``·``dose_drugs``
    인자에 넘길 값을 묶는다. 하나라도 있으면 참으로 평가된다.
    """

    findings: Mapping[str, bool]  # 텍스트에 적힌 사실. 예: {"pregnancy": True}
    proposed_drugs: tuple[ProposedDrug, ...]  # 추가하려는 약(DUR로 점검)
    dose_drugs: tuple[str, ...]  # 텍스트에 나온 약 중 용량 규칙이 있는 성분 ID

    def __bool__(self) -> bool:
        return bool(self.findings or self.proposed_drugs or self.dose_drugs)


class ClinicalRecommender:
    """관문을 통과한 말뭉치 규칙을 EMR 매핑과 의료진 소견에 적용하는 추천기.

    역할: 규칙 말뭉치·사이트 코드 맵·DUR 점검기·용량 단계를 묶어, 원본 EMR을 규칙 입력으로 바꾸고(``prepare``),
    질문 텍스트에서 추가 입력을 뽑고(``question_inputs``), 환자 한 명의 평가 결과(``assess``)를 만든다.
    생성 후에는 상태를 바꾸지 않는다.
    """

    def __init__(
        self, corpus: RuleCorpus, site_map: SiteCodeMap | None = None, *,
        max_possible: int = 12, max_evidence: int = 10, evidence_per_item: int = 1, dur: DurChecker | None = None,
        doses: bool = False, max_dose_items: int = 12, max_dose_evidence: int = 6,
    ) -> None:
        """추천기를 구성한다.

        역할: 표시용 ``Presenter``와 3단계 ``DoseStage``를 만들고, 불러온 규칙이 검사하는 소견 키 사전을 모은다.
        DUR이 설정되면 임부금기 점검을 위해 ``pregnancy`` 소견을 사전에 넣는다.
        입력:
            corpus: 관문을 통과한 규칙 말뭉치.
            site_map: 병원별 EMR 코드 → 말뭉치 개념 매핑. 없으면 ``None``.
            max_possible: ``possible`` 항목과 ``reference``(참고) 항목을 각각 최대 몇 개까지 보일지.
            max_evidence: 2단계 항목에 번호를 붙여 인용할 근거 문단의 최대 수.
            evidence_per_item: 항목 하나가 인용하는 근거 문단 수.
            dur: 심평원 DUR 점검기. ``None``이면 DUR 점검을 하지 않는다.
            doses: 3단계(용량) 평가를 켤지 여부.
            max_dose_items: 용량 항목 최대 수.
            max_dose_evidence: 용량 항목에 추가로 허용하는 근거 문단 수.
        출력: 없음(추천기 생성).
        """
        self._corpus = corpus
        self._dur = dur
        self._site_map = site_map
        self._max_possible = max_possible
        self._max_evidence = max_evidence
        self._evidence_per_item = evidence_per_item
        self._doses = doses
        self._present = Presenter(corpus)
        self._dose_stage = DoseStage(
            corpus, self._present, corpus.rules.get(DOSE_FILE, ()) if doses else (),
            max_items=max_dose_items, max_evidence=max_dose_evidence,
        )
        self._vocabulary = flag_vocabulary(
            record for name, records in corpus.rules.items() if name != DOSE_FILE or doses for record in records
        )
        if dur is not None:
            # DUR 임부금기 점검은 이 소견을 검사하는 말뭉치 규칙이 없어도 이 값을 읽는다.
            self._vocabulary.setdefault("pregnancy", ("finding",))

    def __len__(self) -> int:
        return len(self._corpus)

    @property
    def unreviewed_count(self) -> int:
        return self._corpus.unreviewed_count

    @property
    def dose_rule_count(self) -> int:
        return self._dose_stage.rule_count

    @property
    def dose_drugs(self) -> tuple[str, ...]:
        """용량 규칙이 있는 성분·계열 ID(``dose_drugs`` 입력용). 3단계가 꺼져 있으면 비어 있다."""
        return tuple(sorted(self._dose_stage.targets))

    @property
    def finding_vocabulary(self) -> tuple[str, ...]:
        """불러온 규칙이 검사할 수 있는 소견·사건 키(의료진 입력이나 메모 추출용)."""
        return tuple(sorted(self._vocabulary))

    def question_inputs(self, *texts: str | None) -> QuestionInputs:
        """질문·메모 텍스트에서 규칙·DUR·용량 점검용 추가 입력을 뽑는다.

        역할: 텍스트에 적힌 소견(임신 등), 약물 맥락에서 언급된 약(DUR 점검용, DUR이 설정된 경우), 그리고 3단계가
        켜져 있으면 한글·영문 이름으로 찾은 약 중 용량 규칙이 있는 것을 모은다.
        입력:
            texts: 질문, 진료 메모 등 텍스트. ``None``이나 빈 값은 건너뛴다.
        출력: ``QuestionInputs``. 텍스트가 모두 비었으면 빈 값.
        """
        text = " ".join(item for item in texts if item)
        if not text:
            return QuestionInputs({}, (), ())
        proposed: tuple[ProposedDrug, ...] = ()
        if self._dur is not None and has_medication_context(text):
            proposed = self._dur.find_drugs(text)
        doses: tuple[str, ...] = ()
        if self._doses:
            catalog = self._corpus.catalog
            normalized = normalize_name(text)
            named = catalog.korean_named(normalized)
            named.update(catalog._english_all(text))
            # 용량 규칙이 있는 약만 넣는다: 규칙이 없는 약은 이름이 나와도 미매칭으로 보고하지 않는다.
            doses = tuple(sorted(cid for cid in named if self._dose_stage.resolve((cid,))[0]))
        return QuestionInputs(question_findings(*texts), tuple(proposed), doses)

    def prepare(self, source: Mapping[str, Any]) -> RuleInput:
        """원본 Bridge EMR 객체(봉투가 아닌 내부 객체)를 말뭉치 개념으로 매핑해 규칙 입력을 만든다.

        역할: ``map_emr``로 검사·진단·처방을 개념에 매핑하고, ``calculationContext``에서 성별을 읽으며, BMI가 없고
        체중·키가 그럴듯한 범위이면 BMI를 계산해 넣는다. DUR용으로 현재 처방의 (``medCode``, ``medName``)을 모은다.
        입력:
            source: Bridge EMR 객체(``calculationContext``, ``medicineDtailList`` 등 포함).
        출력: ``RuleInput``(개념 매핑, 성별, 처방 코드·이름).
        """
        mapping = map_emr(source, self._corpus.catalog, self._site_map)
        context = source.get("calculationContext")
        context = context if isinstance(context, Mapping) else {}
        sex = context.get("sex") if context.get("sex") in {"male", "female"} else None
        values = dict(mapping.facts.values)
        weight, height = context.get("weight_kg"), context.get("height_cm")
        if "BMI" not in values and _plausible(weight, 0.3, 500) and _plausible(height, 30, 300):
            values["BMI"] = round(body_mass_index(float(weight), float(height)), 1)  # type: ignore[arg-type]
            mapping = replace(mapping, facts=replace(mapping.facts, values=values))
        medications = tuple(
            (_text_or_none(item.get("medCode")), _text_or_none(item.get("medName")))
            for item in source.get("medicineDtailList") or () if isinstance(item, Mapping)
        )
        return RuleInput(mapping, sex, medications)

    def assess(
        self, rule_input: RuleInput, findings: Mapping[str, bool] | None = None, dose_drugs: Iterable[str] = (),
        proposed_drugs: Sequence[ProposedDrug] = (), conflicts: Mapping[str, Mapping[str, bool]] | None = None,
    ) -> Assessment:
        """환자 한 명에 대해 2단계 규칙·검사값 해석·DUR·3단계 용량을 평가한다.

        역할: 사전에 있는 소견만 받아들여 사실을 만들고, 진단·협진·추천 검사·추적 규칙을 평가해 정렬한다.
        ``possible``과 ``reference`` 항목은 각각 ``max_possible``개까지만 남기고, 정렬 순서대로 근거 번호를 붙인다.
        이어 검사값 해석(국내 기준 먼저), 매핑 문제, DUR 경고, 용량 항목을 모아 ``Assessment``로 돌려준다.
        평가할 수 없는 규칙(``CorpusRuleError``)은 경고 로그를 남기고 건너뛴다.
        입력:
            rule_input: ``prepare``가 만든 규칙 입력.
            findings: 의료진·질문이 준 소견(키 → 참/거짓). 사전에 없는 키는 ``unknown_findings``로 보고한다.
            dose_drugs: 용량 평가를 요청한 약(ID 또는 한글·영문 성분명).
            proposed_drugs: 질문에서 추가하려고 언급한 약(``QuestionInputs``). 현재 처방과 함께 DUR로 점검한다.
            conflicts: 메모와 질문이 서로 다르게 적은 소견(``question_facts.merge_question_findings``).
                알 수 없음으로 평가하고 의료진이 확인하도록 보고한다. ``findings``에 있는 키는 제외한다.
        출력: ``Assessment``(항목, 검사값 해석, 인용 근거, 사용한 소견, 매핑 문제, DUR 경고, 용량 결과, 소견 충돌).
        """
        conflicts = {key: value for key, value in (conflicts or {}).items() if key not in (findings or {})}
        mapping, sex = rule_input.mapping, rule_input.sex
        accepted, unknown = accepted_findings(findings or {}, self._vocabulary)
        facts = derive_facts(mapping.facts, accepted, sex, self._vocabulary)
        numbering = EvidenceNumbering(self._corpus.chunks_by_claim, self._max_evidence, self._evidence_per_item)
        candidates: list[tuple[tuple, RecommendationItem, Mapping[str, Any]]] = []
        for name in ("diagnosis.jsonl", "referral.jsonl", "workup.jsonl", "monitoring.jsonl"):
            for record in self._corpus.rules.get(name, ()):
                try:
                    result = evaluate_record(record, facts)
                except CorpusRuleError as error:
                    logger.warning("clinical_rule_unsupported record_id=%s error=%s", record.get("id"), error)
                    continue
                if result is not None:
                    status, leaves, extra = result
                    item = self._present.item(record, status, leaves, facts, extra)
                    candidates.append((rank_item(item, leaves), item, record))
        candidates.sort(key=lambda entry: entry[0])
        items: list[RecommendationItem] = []
        possible = references = 0
        for _rank_key, item, record in candidates:
            if item.priority == "reference":
                references += 1
                if references > self._max_possible:
                    continue
            elif item.status == "possible":
                possible += 1
                if possible > self._max_possible:
                    continue
            # 정렬 순서대로 번호를 매기므로 정렬한 뒤에 지정한다.
            item.evidence_refs = numbering.refs(record.get("claim_id"))
            items.append(item)
        labs: list[LabInterpretation] = []
        interpretations = sorted(self._present.lab_interpretations(mapping, facts, sex), key=lambda pair: pair[0].priority != "primary")
        for interpretation, record in interpretations:
            interpretation.evidence_refs = numbering.refs(record.get("claim_id"))
            labs.append(interpretation)
        issues = tuple(
            PayloadMappingIssue(**{key: value for key, value in vars(issue).items() if value is not None}) for issue in mapping.issues
        )
        alerts = None
        if self._dur is not None:
            pregnant = accepted.get("pregnancy")
            alerts = self._dur.check(rule_input.medications, age_years=facts.values.get("age"), sex=sex, pregnant=pregnant,
                                     proposed=proposed_drugs, pregnancy_conflict="pregnancy" in conflicts)
        doses: tuple[DoseAssessment, ...] | None = None
        unmatched: tuple[UnmatchedDoseDrug, ...] = ()
        if self._doses:
            requested, unmatched = self._dose_stage.resolve(dose_drugs)
            doses = tuple(self._dose_stage.assessments(mapping, facts, requested, numbering))
        return Assessment(tuple(items), tuple(labs), tuple(numbering.evidence), dict(accepted), issues, tuple(unknown), alerts,
                          doses, unmatched, conflicts)


def _text_or_none(value: object) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _plausible(value: object, low: float, high: float) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and low <= float(value) <= high
