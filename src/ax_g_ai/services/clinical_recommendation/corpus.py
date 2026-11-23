"""말뭉치 릴리스 하나(파일 또는 활성 PostgreSQL 릴리스)에서 관문을 통과한 규칙 레코드.

설계상 위치: 임상 추천 패키지의 데이터 적재 모듈. 서비스 시작 시 파일 폴더나 PostgreSQL에서 읽은 행으로
``RuleCorpus``를 만들고, ``ClinicalRecommender``·``Presenter``·``DoseStage``가 이를 읽기 전용으로 쓴다.

역할: 규칙 파일별로 인용 가능한(``corpus.gate.is_citable``) 레코드만 남기고, claim별 근거 문단과 개념 카탈로그,
상위(국내) 기준이 대체하는 레코드 관계(``preferred_over``)를 모은다.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ax_g_ai.corpus.files import CHUNKS_FILE, CONCEPT_FILES, RULE_FILES
from ax_g_ai.corpus.gate import is_citable
from ax_g_ai.jsonl import read_jsonl
from ax_g_ai.services.clinical_evidence import EvidenceChunk, chunk_from_record, mark_superseded
from ax_g_ai.services.emr_mapping import ConceptCatalog


@dataclass(frozen=True)
class RuleCorpus:
    """관문을 통과한 규칙 레코드, 그 claim들의 근거 문단, 개념 카탈로그.

    역할: 규칙 평가·표시에 필요한 말뭉치 데이터를 한 객체로 묶는다. ``skipped``는 관문에서 걸러진 규칙 레코드 수다.
    """

    rules: Mapping[str, tuple[Mapping[str, Any], ...]]
    chunks_by_claim: Mapping[str, tuple[EvidenceChunk, ...]]
    catalog: ConceptCatalog
    skipped: int = 0
    # claim 또는 레코드 ID → 그것을 대체하는, 불러온 상위 우선순위(국내) 레코드의 claim 목록
    superseded: Mapping[str, tuple[str, ...]] = field(default_factory=dict)

    @classmethod
    def from_rows(
        cls, rows: Mapping[str, Iterable[Mapping[str, Any]]], *, include_unreviewed: bool, require_license: bool = False,
    ) -> RuleCorpus:
        """파일 이름별 원본 행으로 말뭉치를 만든다.

        역할: ``RULE_FILES``마다 인용 가능한 레코드만 남기고(나머지는 ``skipped``로 셈), 인용 가능한 근거 문단을
        대체 표시(``mark_superseded``) 후 claim별로 묶는다. 규칙의 ``preferred_over``로 대체 관계를 만들고,
        개념 파일로 ``ConceptCatalog``를 만든다.
        입력:
            rows: 파일 이름(``diagnosis.jsonl`` 등) → 레코드 행.
            include_unreviewed: 승인되지 않은 초안 레코드도 쓸지 여부.
            require_license: 출처 라이선스 확인을 요구할지 여부.
        출력: ``RuleCorpus``.
        """

        def citable(record: Mapping[str, Any]) -> bool:
            return is_citable(record, include_unreviewed=include_unreviewed, require_license=require_license)

        rules: dict[str, tuple[Mapping[str, Any], ...]] = {}
        skipped = 0
        for name in RULE_FILES:
            kept = []
            for record in rows.get(name, ()):
                if citable(record):
                    kept.append(record)
                else:
                    skipped += 1
            rules[name] = tuple(kept)
        chunks: dict[str, list[EvidenceChunk]] = defaultdict(list)
        for chunk in mark_superseded(chunk_from_record(record) for record in rows.get(CHUNKS_FILE, ()) if citable(record)):
            chunks[chunk.claim_id].append(chunk)
        superseded: dict[str, list[str]] = defaultdict(list)
        for records in rules.values():
            for record in records:
                claim = record.get("claim_id")
                if not isinstance(claim, str):  # 모든 규칙 레코드에 있다. 형식이 틀린 레코드는 대체할 수 없다
                    continue
                for target in record.get("preferred_over") or ():
                    if target != claim and claim not in superseded[target]:
                        superseded[target].append(claim)
        catalog = ConceptCatalog.from_rows(*(rows.get(name, ()) for name in CONCEPT_FILES))
        return cls(rules, {claim: tuple(items) for claim, items in chunks.items()}, catalog, skipped,
                   {target: tuple(claims) for target, claims in superseded.items()})

    @classmethod
    def load(cls, corpus_dir: Path, *, include_unreviewed: bool, require_license: bool = False) -> RuleCorpus:
        """말뭉치 폴더의 JSONL 파일을 읽어 ``from_rows``로 만든다."""
        return cls.from_rows(read_rows(corpus_dir), include_unreviewed=include_unreviewed, require_license=require_license)

    def __len__(self) -> int:
        return sum(len(records) for records in self.rules.values())

    @property
    def unreviewed_count(self) -> int:
        return sum(1 for records in self.rules.values() for record in records if record.get("review_status") != "approved")


def read_rows(corpus_dir: Path) -> dict[str, list[dict[str, Any]]]:
    """말뭉치 폴더에서 규칙·개념·근거 문단 파일을 파일 이름별 행 목록으로 읽는다(없는 파일은 빈 목록, 형식이 틀린 줄은 건너뜀)."""
    return {name: read_jsonl(corpus_dir / name, strict=False) for name in (*RULE_FILES, *CONCEPT_FILES, CHUNKS_FILE)}
