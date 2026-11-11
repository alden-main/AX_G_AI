"""What one request uses from the corpus: evidence search and rules of the same release.

A request reads ``CorpusSource.current`` once and uses that snapshot to the end, so the cited
passages, the rule results and the DUR check never mix two releases. ``StaticCorpus`` holds a
fixed snapshot (file corpus, tests); ``corpus_store.live.LiveCorpus`` swaps in a new one when
the active corpus release or DUR dataset changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from ax_g_ai.services.clinical_evidence import EvidenceRetriever

if TYPE_CHECKING:
    from ax_g_ai.services.clinical_recommendation import ClinicalRecommender


@dataclass(frozen=True)
class CorpusSnapshot:
    evidence: EvidenceRetriever | None = None
    recommender: "ClinicalRecommender | None" = None
    # Corpus release in PostgreSQL (None: file corpus). Prepared patient contexts record it so their
    # rule facts are re-mapped when a newer release is active.
    release_id: int | None = None
    dur_dataset_id: int | None = None


class CorpusSource(Protocol):
    @property
    def current(self) -> CorpusSnapshot: ...

    def close(self) -> None: ...


class StaticCorpus:
    def __init__(self, snapshot: CorpusSnapshot | None = None) -> None:
        self._snapshot = snapshot or CorpusSnapshot()

    @property
    def current(self) -> CorpusSnapshot:
        return self._snapshot

    def close(self) -> None:
        return None
