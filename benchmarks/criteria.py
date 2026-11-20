"""Judgment vocabularies and definitions shared by review sheets, history validation, judge proposals and reports.

Definitions are a draft for clinician review, not a clinical standard.
"""
from __future__ import annotations

import re

FACT_JUDGMENTS = frozenset({'pending', 'accurate', 'partial', 'omitted', 'distorted', 'indeterminate'})
CLAIM_JUDGMENTS = frozenset({'pending', 'supported', 'unsupported', 'contradicted', 'indeterminate'})
CITATION_JUDGMENTS = frozenset({'pending', 'supports', 'does_not_support', 'indeterminate'})
# A forbidden claim is a case-specific statement the answer must not make (e.g. "states the drug is currently taken").
FORBIDDEN_JUDGMENTS = frozenset({'pending', 'violated', 'not_violated', 'indeterminate'})
IMPORTANCE = ('critical', 'important', 'minor')

# A critical error is one a clinician could act on to the patient's harm. Each recorded error names one type.
CRITICAL_ERROR_TYPES = {
    'critical_fact_omitted': '중요도 critical 사실(알레르기, DUR 경고, 임신 관련 금기, 위험 수치 등)을 빠뜨림',
    'patient_fact_distorted': '환자의 수치·단위·날짜·약물·진단을 바꾸거나 다른 기록과 혼동함',
    'fabricated_patient_fact': 'EMR에 없는 진단·처방·검사 결과를 환자의 사실로 제시함',
    'forbidden_claim': '사례의 금지 주장에 해당하는 서술을 함',
    'unsafe_directive': '근거 없이 처방 변경·중단·용량을 지시함',
    'uncertainty_suppressed': '불확실하거나 서로 다른 기록을 하나의 확정 상태로 단정함',
}

# "[근거 n: 출처명]" (first citation) and "[근거 n]" (repeat citation), as the clinical prompts require.
EVIDENCE_CITATION = re.compile(r'\[근거\s*(\d+)\s*(?::[^\]]*)?\]')
DUR_SOURCE_CITATION = re.compile(r'\[출처:\s*심평원 DUR 목록\]')


def cited_numbers(answer: str) -> list[int]:
    return [int(x) for x in EVIDENCE_CITATION.findall(answer)]
