"""Gated rule records of one corpus release (files or the active PostgreSQL release)."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from ax_g_ai.corpus.files import CHUNKS_FILE, CONCEPT_FILES, RULE_FILES
from ax_g_ai.corpus.gate import is_citable
from ax_g_ai.jsonl import read_jsonl
from ax_g_ai.services.clinical_evidence import EvidenceChunk, chunk_from_record, mark_superseded
from ax_g_ai.services.emr_mapping import ConceptCatalog


@dataclass(frozen=True)
class RuleCorpus:
    """Gated rule records, the evidence passages of their claims and the concept catalog."""

    rules: Mapping[str, tuple[Mapping[str, Any], ...]]
    chunks_by_claim: Mapping[str, tuple[EvidenceChunk, ...]]
    catalog: ConceptCatalog
    skipped: int = 0
    # claim or record id → claims of loaded higher-priority (domestic) records that supersede it
    superseded: Mapping[str, tuple[str, ...]] = field(default_factory=dict)

    @classmethod
    def from_rows(
        cls, rows: Mapping[str, Iterable[Mapping[str, Any]]], *, include_unreviewed: bool, require_license: bool = False,
    ) -> "RuleCorpus":
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
                for target in record.get("preferred_over") or ():
                    if target != record.get("claim_id") and record.get("claim_id") not in superseded[target]:
                        superseded[target].append(record.get("claim_id"))
        catalog = ConceptCatalog.from_rows(*(rows.get(name, ()) for name in CONCEPT_FILES))
        return cls(rules, {claim: tuple(items) for claim, items in chunks.items()}, catalog, skipped,
                   {target: tuple(claims) for target, claims in superseded.items()})

    @classmethod
    def load(cls, corpus_dir: Path, *, include_unreviewed: bool, require_license: bool = False) -> "RuleCorpus":
        return cls.from_rows(read_rows(corpus_dir), include_unreviewed=include_unreviewed, require_license=require_license)

    def __len__(self) -> int:
        return sum(len(records) for records in self.rules.values())

    @property
    def unreviewed_count(self) -> int:
        return sum(1 for records in self.rules.values() for record in records if record.get("review_status") != "approved")


def read_rows(corpus_dir: Path) -> dict[str, list[dict[str, Any]]]:
    return {name: read_jsonl(corpus_dir / name, strict=False) for name in (*RULE_FILES, *CONCEPT_FILES, CHUNKS_FILE)}
