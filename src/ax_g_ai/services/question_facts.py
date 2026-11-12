"""Facts a clinician states in a chat question or memo that the rule and DUR checks can use directly.

Deterministic on purpose: DUR matching is code-based and must not hinge on a model's reading, and a chat
answer should not wait on a second provider call. Only explicit statements count — "임신 12주", "임신 아님" —
never a question about the fact ("임신 여부 확인"), which stays unknown.
"""

from __future__ import annotations

import re

_PREGNANCY_DENIED = re.compile(
    r"비\s*임신|not\s+pregnant"
    r"|임신\s*(?:중\s*)?(?:은|이)?\s*(?:아니|아님|아닌|안\s*(?:했|함|한|하)|하지\s*않|않|음성)"
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


def stated_pregnancy(text: str | None) -> bool | None:
    """True when the text states the patient is pregnant, False when it states she is not, else None."""
    if not text:
        return None
    if _PREGNANCY_DENIED.search(text):
        return False
    if _PREGNANCY_STATED.search(text):
        return True
    return None


def question_findings(*texts: str | None) -> dict[str, bool]:
    """Rule findings stated in the texts (first text that states one wins)."""
    for text in texts:
        pregnant = stated_pregnancy(text)
        if pregnant is not None:
            return {"pregnancy": pregnant}
    return {}


def has_medication_context(*texts: str | None) -> bool:
    return any(text and MEDICATION_CONTEXT.search(text) for text in texts)
