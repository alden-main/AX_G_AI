"""검색 토크나이저: 한글 글자 바이그램과 라틴 문자 단어.

설계상 위치: 말뭉치 계약 계층(``corpus`` 패키지). 적재기는 각 문단의 ``terms()``를
``corpus.evidence_chunk.search_tokens``에 저장하고, 메모리 색인(``clinical_evidence``)은 같은 함수로
문단과 질문을 토큰화한다. 개념 이름 정규화(``normalize``)와 한글 판정(``HANGUL``)도 여기서 가져다 쓴다.

역할: NFKC + 소문자 정규화 후, 라틴 단어(2자 이상, 불용어 제외)와 한글 연속 구간의 2글자 바이그램
(주제가 없는 흔한 질문 바이그램 제외)으로 텍스트를 나눈다.
"""

from __future__ import annotations

import re
import unicodedata

HANGUL = re.compile(r"[가-힣]+")
_LATIN = re.compile(r"[a-z][a-z0-9]*(?:[-'][a-z0-9]+)*")
# 의료진 질문에 흔히 나오지만 주제를 담지 않는 말.
_STOP_BIGRAMS = frozenset({
    "환자", "치료", "어떻", "떻게", "무엇", "알려", "려줘", "해줘", "있나", "인가", "경우", "관련", "확인", "가능", "하나", "나요",
    "까요", "에서", "으로", "니다", "습니", "요약", "약해", "정리", "리해", "최근", "결과", "추이", "작성", "성해", "필요", "요할",
    "할까", "인데", "해도", "되나", "작해", "보여", "여줘", "설명", "명해", "오늘", "현재", "지금", "상태",
})
_STOP_WORDS = frozenset({"the", "and", "for", "with", "of", "in", "to"})


def normalize(text: str) -> str:
    """NFKC 정규화 후 소문자로 바꾼다."""
    return unicodedata.normalize("NFKC", text).lower()


def terms(text: str) -> list[str]:
    """텍스트를 검색 토큰 목록으로 나눈다(라틴 단어 먼저, 그다음 한글 바이그램; 중복 유지)."""
    lowered = normalize(text)
    result = [word for word in _LATIN.findall(lowered) if len(word) >= 2 and word not in _STOP_WORDS]
    for run in HANGUL.findall(lowered):
        if len(run) == 1:
            continue
        result.extend(bigram for bigram in (run[i:i + 2] for i in range(len(run) - 1)) if bigram not in _STOP_BIGRAMS)
    return result
