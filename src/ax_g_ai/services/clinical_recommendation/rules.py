"""2단계 규칙 판정: 어떤 레코드가 환자에게 ``met`` 또는 ``possible``이고 어떤 순서로 보일지 정한다.

설계상 위치: 임상 추천 패키지의 판정 모듈. ``ClinicalRecommender.assess``가 레코드마다 ``evaluate_record``와
``rank_item``을 부르고, ``doses``가 용량 조건 규칙의 잎 조건을 얻으려고 ``condition_leaves``를 쓴다.

역할: 평가 자체는 3값 논리다(``clinical_corpus.evaluate``). 조건이 참이면 ``met``, 알 수 없지만 환자와 이어지는
근거점(기록된 진단, 처방, 검사값, 의료진 소견)이 있으면 ``possible``, 그 밖에는 버린다. 의료진 소견을 규칙의
플래그로 바꾸는 일(``accepted_findings``, ``derive_facts``)과 규칙이 검사하는 소견 키 사전(``flag_vocabulary``)도 맡는다.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import replace
from typing import Any

from ax_g_ai.services.clinical_corpus import PatientFacts, evaluate
from ax_g_ai.services.clinical_recommendation.model import Leaf
from ax_g_ai.services.clinical_recommendation.schema import RecommendationItem, Status

FLAG_KINDS = ("finding", "event", "risk", "indication", "lab_present", "dx_history")
_KIND_ORDER = {"diagnosis": 0, "referral": 1, "workup": 2, "monitoring": 3}
_STATUS_ORDER = {"met": 0, "possible": 1}
# 기록된 KCD 코드로 알 수 있는 당뇨병 병력(있음만 판정하고, 없음은 알 수 없음으로 둔다).
_DIABETES_KCD = ("E10", "E11", "E12", "E13", "E14")


def evaluate_record(
    record: Mapping[str, Any], facts: PatientFacts
) -> tuple[Status, list[Leaf], dict[str, Any]] | None:
    """규칙 레코드 하나를 환자 사실로 평가한다.

    역할: 종류별로 평가할 조건을 고른다 — 진단은 ``logic``(기계 평가 가능한 dict일 때만), 추천 검사는 ``trigger``,
    협진은 ``criteria``(목록이면 하나라도 충족), 추적 검사는 ``evaluate_monitoring``에 맡긴다. 그 밖의 종류나
    조건이 dict가 아니면 평가하지 않는다.
    입력:
        record: 말뭉치 규칙 레코드.
        facts: 환자 사실.
    출력: (상태, 잎 조건 목록, 추가 정보) 또는 보고하지 않을 때 ``None``.
    """
    kind = record.get("kind")
    extra: dict[str, Any] = {}
    if kind == "diagnosis":
        condition = record.get("logic")
        if not isinstance(condition, Mapping):
            return None  # 기준이 텍스트뿐 — 기계로 평가할 수 없다
    elif kind == "workup":
        condition = record.get("trigger")
    elif kind == "referral":
        criteria = record.get("criteria")
        condition = {"any": list(criteria)} if isinstance(criteria, list) else criteria
    elif kind == "monitoring":
        return evaluate_monitoring(record, facts)
    else:
        return None
    if not isinstance(condition, Mapping):
        return None
    status = rule_status(condition, facts)
    if status is None:
        return None
    return status[0], status[1], extra


def evaluate_monitoring(
    record: Mapping[str, Any], facts: PatientFacts
) -> tuple[Status, list[Leaf], dict[str, Any]] | None:
    """추적 검사(monitoring) 레코드를 평가한다.

    역할: ``after``가 있으면 그 키들이 모두 성립해야 하며(``rule_status``), 하위 규칙 중 ``when``이 참인 것의 조치를
    모은다. ``after``가 없으면 ``always``가 아닌 하위 규칙 중 ``met``인 것만 발동하고, 하나도 없으면 보고하지 않는다.
    입력:
        record: ``kind``가 ``monitoring``인 규칙 레코드.
        facts: 환자 사실.
    출력: (상태, 잎 조건 목록, ``{"actions": [...]}``) 또는 ``None``.
    """
    after = record.get("after")
    rules = [rule for rule in record.get("rules") or () if isinstance(rule, Mapping)]
    if isinstance(after, Mapping):
        # ``after``에는 여러 키가 섞여 있다(예: event + med_class). 모든 키가 성립해야 한다.
        condition = {"all": [{key: value} for key, value in after.items()]}
        status = rule_status(condition, facts)
        if status is None:
            return None
        actions = [rule.get("action") for rule in rules
                   if isinstance(rule.get("when"), Mapping) and evaluate(rule["when"], facts) is True and rule.get("action")]
        return status[0], status[1], {"actions": actions}
    fired = []
    leaves: list[Leaf] = []
    for rule in rules:
        when = rule.get("when")
        if not isinstance(when, Mapping) or when.get("always") is True:
            continue
        result = rule_status(when, facts)
        if result is not None and result[0] == "met":
            fired.append(rule.get("action"))
            leaves.extend(result[1])
    if not fired:
        return None  # 근거점 없이 주기만 있는 추적 규칙은 보고하지 않는다
    return "met", leaves, {"actions": [action for action in fired if action]}


def accepted_findings(findings: Mapping[str, bool], vocabulary: Mapping[str, tuple[str, ...]]) -> tuple[dict[str, bool], list[str]]:
    """요청 소견 중 규칙이 쓰는 키만 받아들인다.

    역할: ``finding:pregnancy``처럼 플래그 종류 접두어가 붙은 키는 접두어를 떼고, 사전에 있고 값이 bool인 것만 받는다.
    입력:
        findings: 요청 소견(키 → 값).
        vocabulary: ``flag_vocabulary``가 만든 소견 키 → 플래그 종류.
    출력: (받아들인 소견, 받아들이지 않은 원래 키 목록).
    """
    accepted: dict[str, bool] = {}
    unknown: list[str] = []
    for raw, value in findings.items():
        key = raw.split(":", 1)[1] if ":" in raw and raw.split(":", 1)[0] in FLAG_KINDS else raw
        if key in vocabulary and isinstance(value, bool):
            accepted[key] = value
        else:
            unknown.append(raw)
    return accepted, unknown


def derive_facts(
    base: PatientFacts, findings: Mapping[str, bool], sex: str | None, vocabulary: Mapping[str, tuple[str, ...]],
) -> PatientFacts:
    """EMR 사실에 성별·당뇨병 병력·의료진 소견을 플래그로 더한 사실을 만든다.

    역할: 성별로 ``finding:male``을, 당뇨병 KCD 코드로 ``dx_history:diabetes``를 기본값으로 넣고, 의료진 소견은
    사전에 등록된 모든 플래그 종류(``kind:key``)에 덮어쓴다.
    입력:
        base: EMR에서 만든 환자 사실.
        findings: 받아들인 소견.
        sex: ``male``·``female`` 또는 ``None``.
        vocabulary: 소견 키 → 플래그 종류.
    출력: 플래그를 바꾼 새 ``PatientFacts``.
    """
    flags: dict[str, bool] = dict(base.flags or {})
    if sex in {"male", "female"}:
        flags.setdefault("finding:male", sex == "male")
    codes = base.diagnosis_codes or frozenset()
    if any(code.startswith(_DIABETES_KCD) for code in codes):
        flags.setdefault("dx_history:diabetes", True)
    for key, value in findings.items():
        for kind in vocabulary.get(key, ()):
            flags[f"{kind}:{key}"] = value  # 의료진 입력이 유도한 플래그보다 우선한다
    return replace(base, flags=flags)


def rule_status(condition: Mapping[str, Any], facts: PatientFacts) -> tuple[Status, list[Leaf]] | None:
    """조건 하나의 보고 상태를 정한다.

    역할: 조건이 참이면 ``met``, 거짓이면 보고하지 않는다. 알 수 없을 때는 연령·성별이 아닌 근거점이 있으면
    ``possible``이고, 연령·성별만 성립하면 알 수 없는 잎이 모두 ``not``(예: "아직 검사하지 않음")일 때만 ``possible``이다.
    입력:
        condition: 규칙 조건 트리.
        facts: 환자 사실.
    출력: (상태, 잎 조건 목록) 또는 ``None``.
    """
    result = evaluate(condition, facts)
    leaves = list(condition_leaves(condition, facts, negated=False))
    if result is True:
        return "met", leaves
    if result is not None:
        return None
    anchors = [leaf for leaf in leaves if leaf.result is True and not leaf.negated]
    if any(not is_weak_anchor(leaf) for leaf in anchors):
        return "possible", leaves
    # 연령·성별만 성립: 알 수 없는 것이 "아직 하지 않음"뿐인 선별 규칙만 보고하고
    # (예: 이전 검사 묶음이 없는 성인 HBV 선별), 증상으로 시작하는 규칙은 보고하지 않는다.
    if anchors and all(leaf.negated for leaf in leaves if leaf.result is None):
        return "possible", leaves
    return None


def is_weak_anchor(leaf: Leaf) -> bool:
    """그것만으로는 규칙을 이 환자와 잇지 못하는 인구학적 근거점(연령, 유도한 성별)이면 True."""
    return (leaf.kind == "var" and leaf.key == "age") or (leaf.kind == "finding" and leaf.key == "male")


def condition_leaves(condition: Mapping[str, Any], facts: PatientFacts, *, negated: bool) -> Iterable[Leaf]:
    """조건 트리의 잎 조건을 환자 평가 결과와 함께 차례로 내놓는다.

    역할: ``all``·``any``는 자식을 따라 내려가되, 이미 충족된 ``any``는 참인 자식만 따른다(나머지 대안은 누락 입력이
    아니다). ``not``은 부정 여부를 뒤집어 내려가고, ``always``와 알 수 없는 형태의 조건은 건너뛴다.
    입력:
        condition: 조건 트리(또는 하위 조건).
        facts: 환자 사실.
        negated: 현재 위치가 ``not`` 아래인지 여부.
    출력: ``Leaf``를 내놓는 반복자.
    """
    if "all" in condition or "any" in condition:
        children = [child for child in condition.get("all") or condition.get("any") or () if isinstance(child, Mapping)]
        if "any" in condition and evaluate(condition, facts) is True:
            # 이미 충족됨: 알 수 없는 다른 대안은 상관없다(누락 입력이 아니다).
            children = [child for child in children if evaluate(child, facts) is True]
        for child in children:
            yield from condition_leaves(child, facts, negated=negated)
        return
    if "not" in condition:
        if isinstance(condition["not"], Mapping):
            yield from condition_leaves(condition["not"], facts, negated=not negated)
        return
    if condition.get("always") is True:
        return
    kind = next((key for key in ("var", "dx", "med", "med_class", *FLAG_KINDS) if key in condition), None)
    if kind is None:
        return
    yield Leaf(kind, str(condition[kind]), condition, evaluate(condition, facts), negated)


def rank_item(item: RecommendationItem, leaves: list[Leaf]) -> tuple:
    """2단계 항목의 정렬 키를 만든다.

    역할: 국내 기준(``primary``) → ``met`` → 강한 근거점 있음 → 종류(진단·협진·검사·추적) → 강한 근거점 수 많은 순
    → 확인된 잎 비율 높은 순 → 레코드 ID 순으로 정렬되게 한다.
    입력:
        item: 표시용 추천 항목.
        leaves: 그 규칙의 잎 조건.
    출력: 정렬 키 튜플(작을수록 앞).
    """
    anchors = [leaf for leaf in leaves if leaf.result is True and not leaf.negated]
    # 연령·성별이 아닌 근거점(진단, 처방, 검사값, 의료진 소견)이 있는 항목이 앞선다.
    strong = sum(1 for leaf in anchors if not is_weak_anchor(leaf))
    known = sum(1 for leaf in leaves if leaf.result is not None)
    return (
        0 if item.priority == "primary" else 1,  # 국내 기준(또는 경쟁 기준이 없는 것)이 먼저
        _STATUS_ORDER.get(item.status, 9), 0 if strong else 1, _KIND_ORDER.get(item.kind, 9),
        -strong, -(known / len(leaves) if leaves else 0), str(item.record_id),
    )


def flag_vocabulary(records: Iterable[Mapping[str, Any]]) -> dict[str, tuple[str, ...]]:
    """레코드 조건에서 플래그 키와 그 키가 쓰이는 플래그 종류를 모은다.

    역할: ``logic``·``trigger``·``criteria``·``after``·``rules``·``initiation`` 안을 재귀로 훑어 ``FLAG_KINDS`` 키의 문자열
    값을 수집한다. 소문자·숫자·밑줄로만 된 키만 남긴다.
    입력:
        records: 규칙 레코드들.
    출력: 소견 키 → 정렬한 플래그 종류 튜플.
    """
    found: dict[str, set[str]] = defaultdict(set)

    def walk(value: object) -> None:
        if isinstance(value, Mapping):
            for kind in FLAG_KINDS:
                if isinstance(value.get(kind), str):
                    found[value[kind]].add(kind)
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    for record in records:
        for key in ("logic", "trigger", "criteria", "after", "rules", "initiation"):
            walk(record.get(key))
    return {key: tuple(sorted(kinds)) for key, kinds in found.items() if re.fullmatch(r"[a-z0-9_]+", key)}
