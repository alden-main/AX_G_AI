"""Cite updated-corpus evidence chunks (``text_data/corpus/chunks.jsonl``) for a clinician question.

Stage 1 of corpus use: evidence lookup and citation only — no patient-specific rule
evaluation. A chunk is citable only when every source is licensed for commercial AI
use (``ai_use: A``), the claim was checked against the primary text and the claim is
not retired or unverifiable. Clinician-approved chunks are always citable; draft chunks
only when ``include_unreviewed`` is enabled, and they carry an explicit unreviewed label.

Retrieval is in-memory lexical search (BM25 over Hangul character bigrams and Latin
words) boosted by the test/drug concepts named in the question. ``EvidenceRetriever``
is the seam for a database-backed (PostgreSQL full-text + pgvector) retriever.
"""

from __future__ import annotations

import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol

USABLE_VERIFICATIONS = frozenset({"primary", "local_db"})
EXCLUDED_CLAIM_STATUSES = frozenset({"폐기", "확인불가"})
REVIEW_LABELS = {"approved": "의료진 검수 완료", "draft": "의료진 미검수"}
_K1, _B = 1.2, 0.75
_CONCEPT_BONUS = 4.0
_CHAPTER_WEIGHT = 0.5   # terms found elsewhere in the same chapter (e.g. "ALS" only in its diagnosis chunk)
_MIN_SCORE = 9.0
_RELATIVE_CUTOFF = 0.5
_HANGUL = re.compile(r"[가-힣]+")
_LATIN = re.compile(r"[a-z][a-z0-9]*(?:[-'][a-z0-9]+)*")
# Common words in clinician questions that carry no topic.
_STOP_BIGRAMS = frozenset({
    "환자", "치료", "어떻", "떻게", "무엇", "알려", "려줘", "해줘", "있나", "인가", "경우", "관련", "확인", "가능", "하나", "나요",
    "까요", "에서", "으로", "니다", "습니", "요약", "약해", "정리", "리해", "최근", "결과", "추이", "작성", "성해", "필요", "요할",
    "할까", "인데", "해도", "되나", "작해", "보여", "여줘", "설명", "명해", "오늘", "현재", "지금", "상태",
})
_STOP_WORDS = frozenset({"the", "and", "for", "with", "of", "in", "to"})


@dataclass(frozen=True)
class EvidenceChunk:
    chunk_id: str
    claim_id: str
    chapter: str
    title: str
    text: str
    review_status: str
    sources: tuple[Mapping[str, Any], ...]
    conflicts: tuple[str, ...]
    updated_at: str | None
    doc_path: str | None

    @property
    def review_label(self) -> str:
        return REVIEW_LABELS.get(self.review_status, "의료진 미검수")

    def as_dict(self) -> dict[str, object]:
        return {
            "chunk_id": self.chunk_id, "claim_id": self.claim_id, "title": self.title, "text": self.text,
            "review_status": self.review_status, "review_label": self.review_label,
            "sources": [
                {key: source.get(key) for key in ("id", "title", "locator", "url") if source.get(key) is not None}
                for source in self.sources
            ],
            "conflicts": list(self.conflicts), "updated_at": self.updated_at,
        }


class EvidenceRetriever(Protocol):
    def search(self, question: str, limit: int) -> tuple[EvidenceChunk, ...]: ...


def is_citable(record: Mapping[str, Any], *, include_unreviewed: bool) -> bool:
    sources = record.get("sources")
    review = record.get("review_status")
    return (
        record.get("ai_tier") == "A"
        and isinstance(sources, list) and bool(sources)
        and all(isinstance(source, Mapping) and source.get("ai_use") == "A" for source in sources)
        and record.get("verification") in USABLE_VERIFICATIONS
        and record.get("claim_status") not in EXCLUDED_CLAIM_STATUSES
        and (review == "approved" or (include_unreviewed and review == "draft"))
    )


def _normalize(text: str) -> str:
    return unicodedata.normalize("NFKC", text).lower()


def terms(text: str) -> list[str]:
    lowered = _normalize(text)
    result = [word for word in _LATIN.findall(lowered) if len(word) >= 2 and word not in _STOP_WORDS]
    for run in _HANGUL.findall(lowered):
        if len(run) == 1:
            continue
        result.extend(bigram for bigram in (run[i:i + 2] for i in range(len(run) - 1)) if bigram not in _STOP_BIGRAMS)
    return result


class InMemoryEvidenceIndex:
    def __init__(self, chunks: Iterable[EvidenceChunk], concept_names: Mapping[str, frozenset[str]] | None = None) -> None:
        self._chunks = tuple(chunks)
        self._by_id = {chunk.chunk_id: chunk for chunk in self._chunks}
        self._concept_names = concept_names or {}   # normalized name → claim ids that use the concept
        self._doc_terms: list[Counter[str]] = []
        document_frequency: Counter[str] = Counter()
        for chunk in self._chunks:
            counts = Counter(terms(f"{chunk.title} {chunk.title} {chunk.text}"))
            self._doc_terms.append(counts)
            document_frequency.update(counts.keys())
        self._lengths = [sum(counts.values()) for counts in self._doc_terms]
        chapter_terms: dict[str, set[str]] = defaultdict(set)
        for chunk, counts in zip(self._chunks, self._doc_terms):
            chapter_terms[chunk.chapter].update(counts)
        self._chapter_terms = {chapter: frozenset(found) for chapter, found in chapter_terms.items()}
        self._average_length = (sum(self._lengths) / len(self._lengths)) if self._lengths else 1.0
        total = len(self._chunks)
        self._idf = {term: math.log(1 + (total - freq + 0.5) / (freq + 0.5)) for term, freq in document_frequency.items()}

    def __len__(self) -> int:
        return len(self._chunks)

    @property
    def unreviewed_count(self) -> int:
        return sum(1 for chunk in self._chunks if chunk.review_status != "approved")

    @classmethod
    def load(cls, corpus_dir: Path, *, include_unreviewed: bool) -> "InMemoryEvidenceIndex":
        return cls.from_records(
            _jsonl(corpus_dir / "chunks.jsonl"),
            concept_names(*(_jsonl(corpus_dir / name) for name in ("test_concepts.jsonl", "drug_concepts.jsonl", "drug_classes.jsonl"))),
            include_unreviewed=include_unreviewed,
        )

    @classmethod
    def from_records(
        cls, records: Iterable[Mapping[str, Any]], names: Mapping[str, frozenset[str]], *, include_unreviewed: bool
    ) -> "InMemoryEvidenceIndex":
        return cls((chunk_from_record(r) for r in records if is_citable(r, include_unreviewed=include_unreviewed)), names)

    def matched_claims(self, question: str) -> dict[str, int]:
        """Claim id → number of distinct concepts named in the question that the claim uses."""
        lowered = _normalize(question)
        compact = re.sub(r"\s+", "", lowered)
        hits: dict[str, int] = defaultdict(int)
        for name, claims in self._concept_names.items():
            if _HANGUL.fullmatch(name):
                found = name in compact
            else:
                found = re.search(r"(?<![a-z0-9])" + re.escape(name) + r"(?![a-z0-9])", lowered) is not None
            if found:
                for claim in claims:
                    hits[claim] += 1
        return hits

    def search(self, question: str, limit: int = 3) -> tuple[EvidenceChunk, ...]:
        if limit <= 0:
            return ()
        scored = self.scored(question)
        cutoff = lexical_cutoff(scored)
        return tuple(chunk for score, chunk in scored[:limit] if score >= cutoff)

    def chunk(self, chunk_id: str) -> EvidenceChunk | None:
        return self._by_id.get(chunk_id)

    def scored(self, question: str) -> list[tuple[float, EvidenceChunk]]:
        """Every chunk with a positive lexical+concept score, best first (no cutoff)."""
        if not question.strip() or not self._chunks:
            return []
        query = Counter(terms(question))
        concept_hits = self.matched_claims(question)
        scored = []
        for index, chunk in enumerate(self._chunks):
            counts = self._doc_terms[index]
            norm = _K1 * (1 - _B + _B * self._lengths[index] / self._average_length)
            score = sum(
                self._idf.get(term, 0.0) * counts[term] * (_K1 + 1) / (counts[term] + norm)
                for term in query if term in counts
            )
            chapter_terms = self._chapter_terms.get(chunk.chapter, frozenset())
            score += _CHAPTER_WEIGHT * sum(self._idf.get(term, 0.0) for term in query if term not in counts and term in chapter_terms)
            score += _CONCEPT_BONUS * min(concept_hits.get(chunk.claim_id, 0), 3)
            if score > 0:
                scored.append((score, chunk))
        scored.sort(key=lambda item: (-item[0], item[1].chunk_id))
        return scored


def lexical_cutoff(scored: list[tuple[float, EvidenceChunk]]) -> float:
    """Minimum lexical score to cite: absolute floor and a fraction of the best match."""
    return max(_MIN_SCORE, scored[0][0] * _RELATIVE_CUTOFF) if scored else _MIN_SCORE


def chunk_from_record(record: Mapping[str, Any]) -> EvidenceChunk:
    return EvidenceChunk(
        chunk_id=record["chunk_id"], claim_id=record["claim_id"], chapter=str(record.get("chapter", "")),
        title=" > ".join(record.get("title_path") or ()), text=record.get("text", ""),
        review_status=record["review_status"], sources=tuple(record["sources"]),
        conflicts=tuple(record.get("conflicts") or ()), updated_at=record.get("updated_at"),
        doc_path=record.get("doc_path"),
    )


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def concept_names(
    tests: Iterable[Mapping[str, Any]], drugs: Iterable[Mapping[str, Any]], classes: Iterable[Mapping[str, Any]]
) -> dict[str, frozenset[str]]:
    """Names a clinician may type (Korean names, aliases, ingredient ids) → claims using them."""
    names: dict[str, set[str]] = defaultdict(set)

    def add(raw_names: Iterable[object], uses: Iterable[Mapping[str, Any]]) -> None:
        claims = {use["claim_id"] for use in uses if use.get("claim_id")}
        if not claims:
            return
        for raw in raw_names:
            if not isinstance(raw, str):
                continue
            name = _normalize(raw).strip()
            hangul_only = _HANGUL.fullmatch(re.sub(r"\s+", "", name)) is not None
            if hangul_only:
                name = re.sub(r"\s+", "", name)
            if (hangul_only and len(name) >= 2) or (not hangul_only and len(name) >= 4 and re.fullmatch(r"[a-z0-9 \-]+", name)):
                names[name].update(claims)

    for row in tests:
        add([row.get("name_ko"), *(row.get("aliases") or ())], row.get("used_by") or ())
    for row in drugs:
        english = [row["id"].replace("_", " ")] if row.get("type") == "ingredient" else []
        add([*(row.get("ko") or ()), *english], [*(row.get("used_by") or ()), *(row.get("used_by_class") or ())])
    for row in classes:
        add([row.get("name_ko")], row.get("used_by") or ())
    return {name: frozenset(claims) for name, claims in names.items()}
