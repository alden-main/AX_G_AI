"""The gate for every corpus record (passage or rule) the AI may use.

The SQL view ``corpus.citable_evidence`` applies the same rule in the database; this
function runs again in Python behind it, and alone for the file-based corpus.
"""

from __future__ import annotations

from typing import Any, Mapping

USABLE_VERIFICATIONS = frozenset({"primary", "local_db"})
EXCLUDED_CLAIM_STATUSES = frozenset({"폐기", "확인불가"})


def is_citable(record: Mapping[str, Any], *, include_unreviewed: bool, require_license: bool = False) -> bool:
    """Every source licensed for commercial AI use, checked against the primary text, not retired.

    Clinician-approved records always pass; drafts only with ``include_unreviewed``.
    ``require_license`` (attribution-only mode) also demands that each source names its
    license and title, i.e. that the source can actually be attributed on screen.
    """
    sources = record.get("sources")
    review = record.get("review_status")
    return (
        record.get("ai_tier") == "A"
        and isinstance(sources, list) and bool(sources)
        and all(isinstance(source, Mapping) and source.get("ai_use") == "A" for source in sources)
        and (not require_license or all(source.get("license") and source.get("title") for source in sources))
        and record.get("verification") in USABLE_VERIFICATIONS
        and record.get("claim_status") not in EXCLUDED_CLAIM_STATUSES
        and (review == "approved" or (include_unreviewed and review == "draft"))
    )
