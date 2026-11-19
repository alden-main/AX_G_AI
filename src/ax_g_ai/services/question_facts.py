"""Facts a clinician states in a chat question or memo that the rule and DUR checks can use directly.

Deterministic on purpose: DUR matching is code-based and must not hinge on a model's reading, and a chat
answer should not wait on a second provider call. Only explicit statements count — "임신 12주", "임신 아님" —
never a question about the fact ("임신 여부 확인"), which stays unknown.
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
# A drug named in the question is checked as one to add only in a medication context, so a lab or symptom
# word that is also a drug name ("포도당 수치") is not mistaken for a prescription.
MEDICATION_CONTEXT = re.compile(
    r"추가|병용|같이|함께|복용|투여|투약|처방|먹어|먹여|먹으|써도|쓰려|쓰면|사용|시작|변경|바꾸|교체|상호\s*작용|금기|DUR"
    r"|interaction|\badd\b|\bstart\b",
    re.I,
)


# A chat question that asks for diagnoses, tests, medications or doses also gets the stage-2/3 rule results (with
# DUR). A question that names a drug, or states pregnancy, gets them as well (``ClinicalRecommender.question_inputs``).
_RECOMMENDATION_INTENT = re.compile(
    r"추천|감별|의심|진단명|어떤\s*검사|검사가?\s*필요|필요한\s*검사|추가\s*검사|워크업|workup|work-up|협진|의뢰"
    r"|병용|금기|상호\s*작용|DUR|처방|복용|투약|약물|약제|(?<![요절계])약(?:을|이|은|들|과|도|\s|$|\?)"
    r"|먹어도|먹여도|(?:같이|함께)\s*(?:먹|복용|써|쓰|사용|투여)|임신|임부|임산부"
    r"|용량|투여량|감량|증량|mg\s*/\s*kg|신기능\s*(조정|보정)|dose|dosing|medication|drug", re.I,
)


# A change of status since the memo was written: pregnancy ended (delivery, miscarriage, termination) or newly
# confirmed. A question that says so overrides an older memo instead of conflicting with it. "절박유산"(threatened)
# and "출산 예정" are still pregnancy, so only a completed event counts.
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
    value: bool | None  # True: pregnant, False: not, None: not stated
    changed: bool = False  # stated as a change of status (delivery, miscarriage, newly pregnant)


def pregnancy_statement(text: str | None) -> PregnancyStatement:
    if not text:
        return PregnancyStatement(None)
    denied = _PREGNANCY_DENIED.search(text)
    if _PREGNANCY_NEW.search(text) and not denied:
        return PregnancyStatement(True, True)  # "출산 후 다시 임신" is a new pregnancy
    if _PREGNANCY_ENDED.search(text):
        return PregnancyStatement(False, True)
    if denied:
        return PregnancyStatement(False)
    if _PREGNANCY_STATED.search(text):
        return PregnancyStatement(True)
    return PregnancyStatement(None)


def stated_pregnancy(text: str | None) -> bool | None:
    """True when the text states the patient is pregnant, False when it states she is not, else None."""
    return pregnancy_statement(text).value


def question_findings(*texts: str | None) -> dict[str, bool]:
    """Rule findings stated in the texts (first text that states one wins)."""
    for text in texts:
        pregnant = stated_pregnancy(text)
        if pregnant is not None:
            return {"pregnancy": pregnant}
    return {}


def merge_question_findings(
    note: Mapping[str, bool], question: str | None,
) -> tuple[dict[str, bool], dict[str, dict[str, bool]]]:
    """Memo findings with the facts stated in the question, and the findings on which the two disagree.

    The memo (EMR record) wins over the question, except when the question states a change of status since then
    ("출산 후", "새로 임신 확인"). When they disagree without such a cue neither is picked: the finding is left
    unknown and returned as a conflict, so the checks ask the clinician to confirm it.
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
    return any(text and MEDICATION_CONTEXT.search(text) for text in texts)


def asks_for_recommendation(question: str) -> bool:
    """The question asks about diagnoses, tests, referral, medications, pregnancy or doses."""
    return _RECOMMENDATION_INTENT.search(question) is not None
