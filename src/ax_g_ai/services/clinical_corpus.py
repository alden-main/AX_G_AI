"""말뭉치 규칙 조건을 평가하기 위한 환자 사실(fact)과 3값 평가기.

설계상 위치: 서비스 계층의 규칙 평가 핵심. ``emr_mapping``이 EMR에서 ``PatientFacts``를 만들고,
``clinical_recommendation``(규칙·용량)과 ``clinical_dosing``이 ``evaluate``·``CorpusRuleError``를 쓴다.

역할: 레코드는 이 모듈에 오기 전에 걸러진다(``clinical_recommendation.RuleCorpus``가 ``corpus.gate.is_citable``을 적용).

규칙 조건은 3값 논리를 쓴다: 환자 문맥에 없는 사실은 ``False``가 아니라 ``None``(알 수 없음)으로 평가된다.
호출자는 ``None``을 "입력 필요"로 다뤄야 하며, 이는 입력을 추정하지 않는 ``clinical_calculations``와 같은 원칙이다.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

_OPERATORS = {
    "<": lambda a, b: a < b, "<=": lambda a, b: a <= b,
    ">": lambda a, b: a > b, ">=": lambda a, b: a >= b, "==": lambda a, b: a == b,
}
# 불리언 플래그 맵으로 답하는 조건 키. 맵에 없는 플래그는 알 수 없음이다.
_FLAG_KEYS = ("finding", "event", "risk", "indication", "lab_present", "dx_history")


class CorpusRuleError(ValueError):
    """말뭉치 조건이 이 평가기가 지원하지 않는 연산자나 키를 쓴 경우.

    역할: 잘못된 규칙 데이터를 알리는 예외(``ValueError`` 하위).
    """


@dataclass(frozen=True)
class PatientFacts:
    """규칙이 검사할 수 있는 환자 사실. ``None``이면 그 범주 전체를 알 수 없다는 뜻이다.

    역할: ``evaluate``의 입력. ``medications_complete``는 일부 처방을 성분으로 매핑하지 못했을 때 False이며, 그러면
    집합에 없는 약이나 계열은 '없음'이 아니라 '알 수 없음'이다. ``uncertain_classes``는 있으면 참이지만 없을 때는
    알 수 없음으로 남는 계열이다(구성원 목록이 완전하지 않거나, 판단에 용량·기간·경로가 필요한 계열).
    """

    values: Mapping[str, float]
    diagnosis_codes: frozenset[str] | None = None
    medications: frozenset[str] | None = None
    medication_classes: frozenset[str] | None = None
    flags: Mapping[str, bool] | None = None
    medications_complete: bool = True
    uncertain_classes: frozenset[str] = frozenset()


def evaluate(condition: Mapping[str, Any], facts: PatientFacts) -> bool | None:
    """말뭉치 조건을 Kleene 3값 논리로 평가한다.

    역할: ``all``·``any``·``not``을 재귀로 결합하고, ``always``, 수치 비교(``var``·``op``·``value``), 진단 코드 접두사
    (``dx``), 성분(``med``), 약 계열(``med_class``), 플래그 키(``finding``·``event`` 등, ``"<키>:<값>"``으로 조회)를
    판정한다. 해당 범주의 사실이 없으면 ``None``이다.
    입력:
        condition: 말뭉치 규칙의 조건 객체.
        facts: 환자 사실.
    출력: ``True``·``False`` 또는 ``None``(알 수 없음).
    예외:
        CorpusRuleError: 지원하지 않는 연산자나 조건 키.
    """
    if "all" in condition:
        results = [evaluate(item, facts) for item in condition["all"]]
        return False if False in results else (None if None in results else True)
    if "any" in condition:
        results = [evaluate(item, facts) for item in condition["any"]]
        return True if True in results else (None if None in results else False)
    if "not" in condition:
        result = evaluate(condition["not"], facts)
        return None if result is None else not result
    if condition.get("always") is True:
        return True
    if "var" in condition:
        operator = _OPERATORS.get(str(condition.get("op")))
        if operator is None:
            raise CorpusRuleError(f"unsupported operator: {condition.get('op')!r}")
        value = facts.values.get(condition["var"])
        return None if value is None else operator(value, condition["value"])
    if "dx" in condition:
        if facts.diagnosis_codes is None:
            return None
        return any(code.startswith(condition["dx"]) for code in facts.diagnosis_codes)
    if "med" in condition:
        if facts.medications is None:
            return None
        return True if condition["med"] in facts.medications else (False if facts.medications_complete else None)
    if "med_class" in condition:
        if facts.medication_classes is None:
            return None
        if condition["med_class"] in facts.medication_classes:
            return True
        return None if not facts.medications_complete or condition["med_class"] in facts.uncertain_classes else False
    for key in _FLAG_KEYS:
        if key in condition:
            return None if facts.flags is None else facts.flags.get(f"{key}:{condition[key]}")
    raise CorpusRuleError(f"unsupported condition keys: {sorted(condition)}")
