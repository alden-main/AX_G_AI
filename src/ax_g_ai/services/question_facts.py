"""의료진이 채팅 질문이나 메모에 적은 사실 중 규칙·DUR 점검이 바로 쓸 수 있는 것을 정규식으로 읽는 모듈.

설계상 위치: 서비스 계층의 질문 해석 도구. ``clinical_grounding``(근거 검색 여부·소견 병합)과
``ClinicalRecommender.question_inputs``(질문의 소견·약 입력)가 이 모듈을 쓴다.

역할: 임신 여부와 상태 변화(출산·유산·새 임신) 진술, 약 사용 문맥, 추천을 묻는 의도, 기록값만 묻는 질문을 판정한다.
일부러 결정적으로 만든다. DUR 매칭은 코드 기반이라 모델의 판독에 좌우되면 안 되고, 채팅 답변이 두 번째 Provider 호출을
기다려서도 안 된다. 명시적 진술("임신 12주", "임신 아님")만 인정하며, 그 사실을 묻는 질문("임신 여부 확인")은 미상으로 남긴다.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import NamedTuple

_PREGNANCY_DENIED = re.compile(
    r"비\s*임신|not\s+pregnant"
    # "아니(에요)", "아닙니다", "아님", "아닌", "아냐", "아녜요"; "임신한 상태가 아닙니다"처럼 상태가 끼어도 부정이다.
    r"|임신\s*(?:중|한\s*상태|상태)?\s*(?:은|이|가|는)?\s*(?:아니|아닙|아님|아닌|아냐|아녜|안\s*(?:했|함|한|하)|하지\s*않|않|음성)"
    r"|임신\s*(?:가능성|계획)?\s*(?:은|이)?\s*없",
    re.I,
)
_PREGNANCY_STATED = re.compile(
    r"임신\s*(?:중|\d+\s*(?:주|개월)|했|하였|한\s*상태|상태|초기|중기|말기|확인됨|양성)"
    r"|임산부|임부(?:이|인|입|라|예|에|로|\s)|is\s+pregnant|pregnant\s+(?:patient|woman)",
    re.I,
)
# 질문에 나온 약은 약 사용 문맥일 때만 추가할 약으로 점검한다. 그래야 약 이름이기도 한 검사·증상 단어
# ("포도당 수치")를 처방으로 오인하지 않는다.
MEDICATION_CONTEXT = re.compile(
    r"추가|병용|같이|함께|복용|투여|투약|처방|먹어|먹여|먹으|써도|쓰려|쓰면|사용|시작|변경|바꾸|교체|상호\s*작용|금기|DUR"
    r"|interaction|\badd\b|\bstart\b",
    re.I,
)


# 진단·검사·약·용량을 묻는 채팅 질문은 2·3단계 규칙 결과(DUR 포함)도 받는다. 약 이름을 대거나 임신을 언급한
# 질문도 마찬가지다(``ClinicalRecommender.question_inputs``).
_RECOMMENDATION_INTENT = re.compile(
    r"추천|감별|의심|진단명|어떤\s*검사|검사가?\s*필요|필요한\s*검사|추가\s*검사|워크업|workup|work-up|협진|의뢰"
    r"|병용|금기|상호\s*작용|DUR|처방|복용|투약|약물|약제|(?<![요절계])약(?:을|이|은|들|과|도|\s|$|\?)"
    r"|먹어도|먹여도|(?:같이|함께)\s*(?:먹|복용|써|쓰|사용|투여)|임신|임부|임산부"
    r"|용량|투여량|감량|증량|mg\s*/\s*kg|신기능\s*(조정|보정)|dose|dosing|medication|drug", re.I,
)


# 메모 작성 이후의 상태 변화: 임신 종료(출산, 유산, 중절) 또는 새로 확인된 임신. 이를 말하는 질문은 예전 메모와
# 충돌하는 대신 메모를 덮어쓴다. "절박유산"과 "출산 예정"은 아직 임신 상태이므로 완료된 사건만 센다.
_PREGNANCY_ENDED = re.compile(
    r"(?:출산|분만|해산|제왕\s*절개)\s*(?:후|했|하였|함|완료|한\s*(?:뒤|후|지))|(?<![가-힣])산후|산욕기"
    r"|유산\s*(?:후|했|하였|됨|되었|됐|된\s*(?:뒤|후))|임신\s*(?:중절|종결)|중절\s*(?:수술\s*)?(?:후|했|받)|사산\s*(?:후|했|됨)"
    r"|post-?partum|after\s+(?:delivery|childbirth)|gave\s+birth|delivered\s+(?:a\s+|her\s+)?baby",
    re.I,
)
_PREGNANCY_NEW = re.compile(
    r"(?:새로|새롭게|다시|이번에|최근(?:에)?|방금)\s*임신(?!\s*(?:여부|가능성|계획))"
    r"|임신\s*(?:사실을?\s*)?(?:알게|판정\s*받|진단\s*받)|newly\s+pregnant|now\s+pregnant|became\s+pregnant",
    re.I,
)


class PregnancyStatement(NamedTuple):
    """텍스트에서 읽은 임신 진술.

    역할: 임신 여부 값과, 그것이 상태 변화로 진술되었는지를 함께 전달해 메모와 질문의 병합 규칙을 정하게 한다.
    """

    value: bool | None  # True: 임신, False: 임신 아님, None: 언급 없음
    changed: bool = False  # 상태 변화로 진술됨(출산, 유산, 새 임신)


def pregnancy_statement(text: str | None) -> PregnancyStatement:
    """텍스트의 임신 진술을 읽는다.

    역할: 부정 없는 새 임신 > 임신 종료 > 부정 > 임신 진술 순으로 판정한다("출산 후 다시 임신"은 새 임신).
    입력:
        text: 질문 또는 메모. 비어 있으면 언급 없음.
    출력: ``PregnancyStatement``. 새 임신은 (True, 변화), 종료는 (False, 변화), 부정은 False, 진술은 True, 그 외 None.
    """
    if not text:
        return PregnancyStatement(None)
    denied = _PREGNANCY_DENIED.search(text)
    if _PREGNANCY_NEW.search(text) and not denied:
        return PregnancyStatement(True, True)  # "출산 후 다시 임신"은 새 임신이다
    if _PREGNANCY_ENDED.search(text):
        return PregnancyStatement(False, True)
    if denied:
        return PregnancyStatement(False)
    if _PREGNANCY_STATED.search(text):
        return PregnancyStatement(True)
    return PregnancyStatement(None)


def stated_pregnancy(text: str | None) -> bool | None:
    """텍스트가 임신이라고 말하면 True, 임신이 아니라고 말하면 False, 그 밖에는 None."""
    return pregnancy_statement(text).value


def question_findings(*texts: str | None) -> dict[str, bool]:
    """텍스트에 진술된 규칙 소견(현재는 ``pregnancy``만; 진술이 있는 첫 텍스트가 이긴다)."""
    for text in texts:
        pregnant = stated_pregnancy(text)
        if pregnant is not None:
            return {"pregnancy": pregnant}
    return {}


def merge_question_findings(
    note: Mapping[str, bool], question: str | None,
) -> tuple[dict[str, bool], dict[str, dict[str, bool]]]:
    """메모 소견에 질문에 진술된 사실을 합치고, 둘이 엇갈리는 소견을 따로 돌려준다.

    역할: 메모(EMR 기록)가 질문보다 우선하되, 질문이 그 뒤의 상태 변화("출산 후", "새로 임신 확인")를 말하면 질문을 따른다.
        그런 표현 없이 엇갈리면 어느 쪽도 고르지 않고 소견을 미상으로 둔 채 충돌로 돌려주어, 점검이 의료진에게 확인을
        요청하게 한다. 메모에 없는 소견은 질문 값을 쓴다. 현재 질문에서 읽는 소견은 ``pregnancy``뿐이다.
    입력:
        note: 메모에서 읽은 소견(키 → True/False).
        question: 의료진 질문(없으면 ``None``).
    출력: (병합된 소견, 충돌 소견 → ``{"clinical_note": 메모 값, "question": 질문 값}``) 튜플.
    """
    findings = dict(note)
    conflicts: dict[str, dict[str, bool]] = {}
    stated = pregnancy_statement(question)
    if stated.value is not None:
        recorded = findings.get("pregnancy")
        if recorded is None or stated.changed:
            findings["pregnancy"] = stated.value
        elif recorded != stated.value:
            del findings["pregnancy"]
            conflicts["pregnancy"] = {"clinical_note": recorded, "question": stated.value}
    return findings, conflicts


def has_medication_context(*texts: str | None) -> bool:
    """텍스트 중 하나라도 약 사용 문맥(추가·병용·처방·DUR 등)을 담으면 True."""
    return any(text and MEDICATION_CONTEXT.search(text) for text in texts)


def asks_for_recommendation(question: str) -> bool:
    """질문이 진단·검사·의뢰·약·임신·용량에 대해 묻는지 여부."""
    return _RECOMMENDATION_INTENT.search(question) is not None


# 환자의 기록값만 묻는 질문("최근 혈압 기록 알려줘")에는 일반 근거가 필요 없다. 어휘 색인이 요청 표현("있으면",
# "주세요")에 일치해 무관한 문단을 찾고, 화면이 그것을 출처로 나열하게 된다. 값의 의미·목표·지침·치료도 묻는
# 질문은 그대로 검색한다.
_RECORD_LOOKUP = re.compile(
    r"기록|수치|값|측정|결과|목록|내역|날짜|최근|이전|추이|변화|비교|알려|정리|요약|보여|있나요|있는지|확인해|계산해|구해|얼마"
    r"|record|value|latest|trend",
    re.I,
)
_KNOWLEDGE_INTENT = re.compile(
    r"기준|목표|범위|정상|근거|출처|지침|가이드|권고|의미|해석|원인|이유|왜|어떻게|치료|관리\s*(?:방법|목표|계획)"
    r"|가능성|주의|위험|예후|평가|부작용|영향|설명해|일반적"
    r"|guideline|target|range|normal|meaning|why|how|treat",
    re.I,
)


def is_record_lookup(question: str) -> bool:
    """질문이 지식·추천·약 점검·용량이 아니라 환자의 기록값만 묻는지 여부."""
    return (_RECORD_LOOKUP.search(question) is not None and _KNOWLEDGE_INTENT.search(question) is None
            and not asks_for_recommendation(question))
