"""의료진 질문에 인용할 최신화 말뭉치 근거 문단(``text_data/corpus/chunks.jsonl``)을 찾는 모듈.

설계상 위치: 서비스 계층의 근거 검색(말뭉치 활용 1단계: 근거 조회와 인용만, 환자별 규칙 평가는 없음).
``runtime``이 말뭉치 행으로 이 색인을 만들어 ``CorpusSnapshot.evidence``에 두고, ``ClinicalChatService.find_evidence``가
검색한다. ``corpus_store.retriever``(DB 기반 검색기)는 이 색인으로 어휘·개념 점수를 매기고, 규칙 말뭉치
(``clinical_recommendation``)는 ``EvidenceChunk``·``chunk_from_record``·``mark_superseded``와 지역 우선순위 함수를 쓴다.
``EvidenceRetriever``는 데이터베이스 기반(PostgreSQL 전문 검색 + pgvector) 검색기로 바꿔 끼우는 접점이다.

역할: 인용 관문(``corpus.gate.is_citable``)을 통과한 문단만 색인한다. 모든 출처가 상업적 AI 이용 허가(``ai_use: A``)를
받았고, 주장이 원문과 대조되었으며, 폐기·확인불가가 아닐 때만 인용할 수 있다. 의료진 승인 문단은 항상, 초안 문단은
``include_unreviewed``가 켜졌을 때만 인용하며 검수 상태(``review_status``)가 함께 실린다.
검색은 메모리 내 어휘 검색(한글 글자 바이그램·라틴 단어 위의 BM25)에 같은 장(chapter)의 용어 가중치와 질문에 나온
검사·약 개념 보너스를 더한다. 출처 지역 우선순위(국내 > 미국 > 유럽 > 기타)와 ``preferred_over``로 밀린 문단 표시도 맡는다.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Protocol

from ax_g_ai.corpus.files import CHUNKS_FILE, CONCEPT_FILES
from ax_g_ai.corpus.gate import is_citable
from ax_g_ai.corpus.text import HANGUL, normalize, terms
from ax_g_ai.jsonl import read_jsonl
from ax_g_ai.services.term_matcher import TermMatcher

_K1, _B = 1.2, 0.75
_CONCEPT_BONUS = 4.0
_CHAPTER_WEIGHT = 0.5   # 같은 장의 다른 문단에만 있는 용어(예: 진단 문단에만 나오는 "ALS")
_MIN_SCORE = 9.0
_RELATIVE_CUTOFF = 0.5


# 인용한 문단·규칙마다 함께 보여 주는 출처 필드: 출처 표기에 충분한 정보(CC BY는 제목·저자·라이선스가 필요).
SOURCE_FIELDS = ("id", "title", "citation", "locator", "url", "license", "region")
# 출처 우선순위: 국내(한국) > 미국 > 유럽 > 기타. 우선순위가 더 높은 레코드의 ``preferred_over``에 주장이 올라 있는
# 레코드는 참고로만 보여 주고 추천으로 쓰지 않는다.
REGION_ORDER = ("KR", "US", "EU", "INTL")
REGION_LABELS = {"KR": "국내", "US": "미국", "EU": "유럽", "INTL": "국제·기타"}


def region_rank(sources: Iterable[Mapping[str, Any]]) -> int:
    """출처들 중 가장 높은 지역 우선순위(``REGION_ORDER`` 위치). 알려진 지역이 없으면 ``len(REGION_ORDER)``."""
    ranks = [REGION_ORDER.index(source["region"]) for source in sources
             if isinstance(source, Mapping) and source.get("region") in REGION_ORDER]
    return min(ranks, default=len(REGION_ORDER))


def region_of(sources: Iterable[Mapping[str, Any]]) -> str | None:
    """출처들 중 우선순위가 가장 높은 지역 코드. 알려진 지역이 없으면 None."""
    rank = region_rank(sources)
    return REGION_ORDER[rank] if rank < len(REGION_ORDER) else None


def sorted_sources(sources: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """``SOURCE_FIELDS`` 중 값이 있는 필드만 남긴 화면용 출처 목록(국내 출처 먼저)."""
    items = [{key: source.get(key) for key in SOURCE_FIELDS if source.get(key) is not None}
             for source in sources if isinstance(source, Mapping)]
    return sorted(items, key=lambda item: region_rank([item]))


@dataclass(frozen=True)
class EvidenceChunk:
    """인용할 수 있는 근거 문단 하나.

    역할: 말뭉치 레코드에서 검색·인용에 필요한 필드(본문, 검수 상태, 출처, 충돌, 대체 관계)를 담고,
    프롬프트·API 응답용 딕셔너리(``as_dict``)로 바꾼다.
    """

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
    preferred_over: tuple[str, ...] = ()
    # 이 문단을 대체하는, 적재된 상위 우선순위(국내) 문단의 주장 id. 색인 적재 시 설정한다.
    superseded_by: tuple[str, ...] = ()

    @property
    def priority(self) -> str:
        """다른 문단에 밀렸으면 ``reference``, 아니면 ``primary``."""
        return "reference" if self.superseded_by else "primary"

    @property
    def region(self) -> str | None:
        """출처 중 우선순위가 가장 높은 지역 코드."""
        return region_of(self.sources)

    def as_dict(self) -> dict[str, object]:
        """프롬프트·API 응답용 딕셔너리.

        역할: 식별자·제목·본문·검수 상태·지역(라벨, 미상이면 "미상")·우선순위·대체 관계(있을 때)·정렬된 출처·충돌·
            갱신 시각을 담는다. ``chapter``·``doc_path``·``preferred_over``는 넣지 않는다.
        입력: 없음.
        출력: 문단 딕셔너리.
        """
        region = self.region
        return {
            "chunk_id": self.chunk_id, "claim_id": self.claim_id, "title": self.title, "text": self.text,
            "review_status": self.review_status,
            "region": region, "region_label": REGION_LABELS.get(region or "", "미상"),
            "priority": self.priority, **({"superseded_by": list(self.superseded_by)} if self.superseded_by else {}),
            "sources": sorted_sources(self.sources),
            "conflicts": list(self.conflicts), "updated_at": self.updated_at,
        }


def mark_superseded(chunks: Iterable[EvidenceChunk]) -> list[EvidenceChunk]:
    """적재된 다른 문단의 ``preferred_over``에 주장이 올라 있는 문단에 대체 관계를 표시한다(해외보다 국내 우선).

    역할: 자기 자신을 가리키는 항목은 무시하고, 대체하는 주장 id를 순서 유지·중복 제거해 ``superseded_by``에 넣는다.
    입력:
        chunks: 색인할 근거 문단.
    출력: 입력 순서를 유지한 문단 목록(대체된 문단만 새 객체).
    """
    items = list(chunks)
    superseders: dict[str, list[str]] = defaultdict(list)
    for chunk in items:
        for target in chunk.preferred_over:
            if target != chunk.claim_id:
                superseders[target].append(chunk.claim_id)
    return [replace(chunk, superseded_by=tuple(dict.fromkeys(superseders[chunk.claim_id]))) if chunk.claim_id in superseders else chunk
            for chunk in items]


def prioritize(chunks: Iterable[EvidenceChunk]) -> list[EvidenceChunk]:
    """관련도 순서는 유지하되 대체된(참고 전용) 문단을 나머지 뒤로 보낸다."""
    items = list(chunks)
    return [chunk for chunk in items if not chunk.superseded_by] + [chunk for chunk in items if chunk.superseded_by]


class EvidenceRetriever(Protocol):
    """근거 검색기 인터페이스.

    역할: 메모리 색인과 데이터베이스 기반 검색기를 바꿔 끼울 수 있게 하는 접점. 질문으로 최대 ``limit``개 문단을 돌려준다.
    """

    def search(self, question: str, limit: int) -> tuple[EvidenceChunk, ...]: ...


class InMemoryEvidenceIndex:
    """메모리 내 어휘 검색 근거 색인.

    역할: 인용 가능한 문단을 BM25 통계·장별 용어 집합·개념 이름 매처와 함께 보관하고, 질문에 대한 점수 계산
    (``scored``), 컷오프를 적용한 검색(``search``), id 조회(``chunk``)를 제공한다. 생성 후 바뀌지 않는다.
    """

    def __init__(self, chunks: Iterable[EvidenceChunk], concept_names: Mapping[str, frozenset[str]] | None = None) -> None:
        """문단과 개념 이름으로 색인을 만든다.

        역할: 대체 관계를 표시하고, 한글·라틴 개념 이름 매처, 문단별 용어 빈도(제목은 두 번 셈), 문서 길이,
            장별 용어 집합, 평균 길이, IDF를 미리 계산한다.
        입력:
            chunks: 인용 관문을 통과한 근거 문단.
            concept_names: 정규화된 개념 이름 → 그 개념을 쓰는 주장 id. ``None``이면 개념 보너스 없음.
        출력: 없음.
        """
        self._chunks = tuple(mark_superseded(chunks))
        self._by_id = {chunk.chunk_id: chunk for chunk in self._chunks}
        self._concept_names = concept_names or {}   # 정규화된 이름 → 그 개념을 쓰는 주장 id
        # 한글 이름은 공백을 없앤 질문에서, 라틴 이름은 단어 단위로 찾는다.
        self._hangul_names = TermMatcher((name, name) for name in self._concept_names if HANGUL.fullmatch(name))
        self._latin_names = TermMatcher(
            ((name, name) for name in self._concept_names if not HANGUL.fullmatch(name)), boundary="a-z0-9",
        )
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
        """승인(``approved``)되지 않은 문단 수."""
        return sum(1 for chunk in self._chunks if chunk.review_status != "approved")

    @classmethod
    def load(cls, corpus_dir: Path, *, include_unreviewed: bool, require_license: bool = False) -> InMemoryEvidenceIndex:
        """말뭉치 폴더의 문단·개념 파일을 읽어 색인을 만든다(잘못된 줄은 건너뜀, ``strict=False``)."""
        rows = {name: read_jsonl(corpus_dir / name, strict=False) for name in (CHUNKS_FILE, *CONCEPT_FILES)}
        return cls.from_rows(rows, include_unreviewed=include_unreviewed, require_license=require_license)

    @classmethod
    def from_rows(
        cls, rows: Mapping[str, Iterable[Mapping[str, Any]]], *, include_unreviewed: bool, require_license: bool = False,
    ) -> InMemoryEvidenceIndex:
        """파일 이름별 말뭉치 행으로 색인을 만든다. ``RuleCorpus.from_rows``와 같은 매핑을 읽으므로 한 번 읽어 둘 다 쓴다."""
        return cls.from_records(
            rows.get(CHUNKS_FILE, ()), concept_names(*(rows.get(name, ()) for name in CONCEPT_FILES)),
            include_unreviewed=include_unreviewed, require_license=require_license,
        )

    @classmethod
    def from_records(
        cls, records: Iterable[Mapping[str, Any]], names: Mapping[str, frozenset[str]], *, include_unreviewed: bool,
        require_license: bool = False,
    ) -> InMemoryEvidenceIndex:
        """인용 관문(``is_citable``)을 통과한 문단 레코드만으로 색인을 만든다."""
        return cls((chunk_from_record(r) for r in records
                    if is_citable(r, include_unreviewed=include_unreviewed, require_license=require_license)), names)

    def matched_claims(self, question: str) -> dict[str, int]:
        """질문에 나온 서로 다른 개념 중 각 주장이 쓰는 개념의 수를 센다.

        역할: 한글 이름은 공백을 없앤 정규화 질문에서, 라틴 이름은 단어 경계를 지켜 정규화 질문에서 찾는다.
        입력:
            question: 의료진 질문.
        출력: 주장 id → 일치한 개념 수.
        """
        lowered = normalize(question)
        compact = re.sub(r"\s+", "", lowered)
        found = self._hangul_names.terms_in(compact) | self._latin_names.terms_in(lowered)
        hits: dict[str, int] = defaultdict(int)
        for name in found:
            for claim in self._concept_names[name]:
                hits[claim] += 1
        return hits

    def search(self, question: str, limit: int = 3) -> tuple[EvidenceChunk, ...]:
        """컷오프(``lexical_cutoff``) 이상인 문단을 대체된 문단을 뒤로 보내 정렬한 뒤 ``limit``개까지 돌려준다."""
        if limit <= 0:
            return ()
        scored = self.scored(question)
        cutoff = lexical_cutoff(scored)
        return tuple(prioritize(chunk for score, chunk in scored if score >= cutoff)[:limit])

    def chunk(self, chunk_id: str) -> EvidenceChunk | None:
        """id로 색인된 문단을 찾는다(없으면 None)."""
        return self._by_id.get(chunk_id)

    def scored(self, question: str) -> list[tuple[float, EvidenceChunk]]:
        """어휘+개념 점수가 양수인 모든 문단을 점수 높은 순으로 돌려준다(컷오프 없음).

        역할: 문단마다 BM25 점수에, 문단에는 없지만 같은 장에 있는 질문 용어의 IDF 합 × ``_CHAPTER_WEIGHT``와
            주장이 쓰는 질문 개념 수(최대 3) × ``_CONCEPT_BONUS``를 더한다. 같은 점수는 ``chunk_id`` 순.
        입력:
            question: 의료진 질문. 공백뿐이거나 색인이 비어 있으면 빈 목록.
        출력: (점수, 문단) 목록.
        """
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
    """인용할 최소 점수: 절대 하한(``_MIN_SCORE``)과 최고 점수의 일정 비율(``_RELATIVE_CUTOFF``) 중 큰 값."""
    return max(_MIN_SCORE, scored[0][0] * _RELATIVE_CUTOFF) if scored else _MIN_SCORE


def chunk_from_record(record: Mapping[str, Any]) -> EvidenceChunk:
    """말뭉치 문단 레코드를 ``EvidenceChunk``로 바꾼다(제목은 ``title_path``를 " > "로 잇는다)."""
    return EvidenceChunk(
        chunk_id=record["chunk_id"], claim_id=record["claim_id"], chapter=str(record.get("chapter", "")),
        title=" > ".join(record.get("title_path") or ()), text=record.get("text", ""),
        review_status=record["review_status"], sources=tuple(record["sources"]),
        conflicts=tuple(record.get("conflicts") or ()), updated_at=record.get("updated_at"),
        doc_path=record.get("doc_path"), preferred_over=tuple(record.get("preferred_over") or ()),
    )


def concept_names(
    tests: Iterable[Mapping[str, Any]], drugs: Iterable[Mapping[str, Any]], classes: Iterable[Mapping[str, Any]]
) -> dict[str, frozenset[str]]:
    """의료진이 입력할 만한 이름(한글명, 별칭, 성분 id)에서 그 이름을 쓰는 주장으로의 사전을 만든다.

    역할: 이름을 정규화하고, 한글만으로 된 이름은 공백을 없애 2자 이상, 그 밖의 이름은 4자 이상이면서
        소문자·숫자·공백·하이픈만일 때 등록한다. 성분(``type: ingredient``) 약은 id의 ``_``를 공백으로 바꿔 영문명으로 쓰고,
        약은 ``used_by``와 ``used_by_class``를 모두 사용처로 본다. 사용처 주장이 없는 행은 건너뛴다.
    입력:
        tests: 검사 개념 행(``name_ko``, ``aliases``, ``used_by``).
        drugs: 약 개념 행(``ko``, ``id``, ``type``, ``used_by``, ``used_by_class``).
        classes: 약 분류 행(``name_ko``, ``used_by``).
    출력: 정규화된 이름 → 주장 id ``frozenset``.
    """
    names: dict[str, set[str]] = defaultdict(set)

    def add(raw_names: Iterable[object], uses: Iterable[Mapping[str, Any]]) -> None:
        """조건을 만족하는 이름들을 ``uses``의 주장 id와 함께 등록한다."""
        claims = {use["claim_id"] for use in uses if use.get("claim_id")}
        if not claims:
            return
        for raw in raw_names:
            if not isinstance(raw, str):
                continue
            name = normalize(raw).strip()
            hangul_only = HANGUL.fullmatch(re.sub(r"\s+", "", name)) is not None
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
