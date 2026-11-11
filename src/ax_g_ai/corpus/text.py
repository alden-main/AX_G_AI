"""Search tokenizer: Hangul character bigrams and Latin words.

The loader stores ``terms()`` of each passage in ``corpus.evidence_chunk.search_tokens``,
and the in-memory index tokenizes passages and questions with the same function.
"""

from __future__ import annotations

import re
import unicodedata

HANGUL = re.compile(r"[가-힣]+")
_LATIN = re.compile(r"[a-z][a-z0-9]*(?:[-'][a-z0-9]+)*")
# Common words in clinician questions that carry no topic.
_STOP_BIGRAMS = frozenset({
    "환자", "치료", "어떻", "떻게", "무엇", "알려", "려줘", "해줘", "있나", "인가", "경우", "관련", "확인", "가능", "하나", "나요",
    "까요", "에서", "으로", "니다", "습니", "요약", "약해", "정리", "리해", "최근", "결과", "추이", "작성", "성해", "필요", "요할",
    "할까", "인데", "해도", "되나", "작해", "보여", "여줘", "설명", "명해", "오늘", "현재", "지금", "상태",
})
_STOP_WORDS = frozenset({"the", "and", "for", "with", "of", "in", "to"})


def normalize(text: str) -> str:
    return unicodedata.normalize("NFKC", text).lower()


def terms(text: str) -> list[str]:
    lowered = normalize(text)
    result = [word for word in _LATIN.findall(lowered) if len(word) >= 2 and word not in _STOP_WORDS]
    for run in HANGUL.findall(lowered):
        if len(run) == 1:
            continue
        result.extend(bigram for bigram in (run[i:i + 2] for i in range(len(run) - 1)) if bigram not in _STOP_BIGRAMS)
    return result
