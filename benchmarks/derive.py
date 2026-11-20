"""Expected derived facts computed with the service's own deterministic code, not written by hand.

DUR alerts come from ``DurChecker`` over the DUR snapshot shipped in ``data/dur`` (the same files the loader puts in
PostgreSQL), calculations from ``calculate_review_aids``, and the question reading from ``question_facts``. The
generator writes these into the gold file; ``schema.validate`` recomputes them, so a change in the service logic or
the snapshot makes validation fail instead of silently leaving stale references.

The server must run the same DUR dataset (``source.id`` and the snapshot ``content_sha256`` are recorded) for the
expected alerts to apply; the recommendation scorer checks the dataset the server reports.
"""
from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any

from ax_g_ai.services.clinical_calculations import calculate_review_aids
from ax_g_ai.services.dur_check import DurChecker, DurTables
from ax_g_ai.services.question_facts import (
    asks_for_recommendation,
    has_medication_context,
    merge_question_findings,
    question_findings,
)

DUR_DIR = Path(__file__).resolve().parent.parent / 'data' / 'dur'
TABLES = ('product', 'pair_rule', 'age_rule', 'pregnancy_rule', 'elderly_rule')


def dur_manifest() -> dict[str, Any]:
    return json.loads((DUR_DIR / 'manifest.json').read_text(encoding='utf-8'))


@lru_cache(maxsize=1)
def dur_checker() -> DurChecker:
    manifest = dur_manifest()
    rows = {}
    for table in TABLES:
        with (DUR_DIR / manifest['tables'][table]['file']).open(encoding='utf-8') as stream:
            rows[table] = [json.loads(line) for line in stream if line.strip()]
    return DurChecker(DurTables.from_rows(manifest['source'], rows))


def dur_version() -> dict[str, str]:
    manifest = dur_manifest()
    return {'source_id': manifest['source']['id'], 'content_sha256': manifest['content_sha256']}


def _medications(emr: Mapping[str, Any]) -> list[tuple[str | None, str | None]]:
    def text(value):
        return value.strip() if isinstance(value, str) and value.strip() else None
    return [(text(item.get('medCode')), text(item.get('medName')))
            for item in emr.get('medicineDtailList') or () if isinstance(item, Mapping)]


def _context(emr: Mapping[str, Any]) -> tuple[float | None, str | None]:
    context = emr.get('calculationContext') if isinstance(emr.get('calculationContext'), Mapping) else {}
    age = context.get('age_years')
    age = float(age) if isinstance(age, (int, float)) and not isinstance(age, bool) and 0 <= age <= 130 else None
    sex = context.get('sex') if context.get('sex') in {'male', 'female'} else None
    return age, sex


def alert_summary(alert: Mapping[str, Any]) -> dict[str, Any]:
    """The comparable part of a DUR alert: type, status, ingredients, whether a proposed (to-add) drug is involved."""
    products = alert.get('products') or []
    summary = {'type': alert['type'], 'status': alert['status'],
               'ingredients': sorted({p.get('ingredient_name') for p in products if p.get('ingredient_name')}),
               'proposed': any(p.get('proposed') for p in products), 'detail': alert.get('detail')}
    if alert.get('grade'):
        summary['grade'] = alert['grade']
    return summary


def alert_key(summary: Mapping[str, Any]) -> tuple:
    return summary['type'], summary['status'], tuple(summary['ingredients']), bool(summary['proposed']), summary.get('detail')


def _dur(emr, *, pregnant, proposed_text, conflict) -> dict[str, Any]:
    checker = dur_checker()
    age, sex = _context(emr)
    proposed = checker.find_drugs(proposed_text) if proposed_text and has_medication_context(proposed_text) else ()
    result = checker.check(_medications(emr), age_years=age, sex=sex, pregnant=pregnant, proposed=proposed,
                           pregnancy_conflict=conflict).as_dict()
    alerts = sorted((alert_summary(a) for a in result['alerts']), key=alert_key)
    return {'alerts': alerts, 'unmapped': sorted(u['med_code'] or '' for u in result['unmapped']),
            'proposed': sorted(p['ingredient_name'] for p in result.get('proposed', []))}


def chat_dur(emr: Mapping[str, Any], question: str) -> dict[str, Any] | None:
    """DUR results a chat answer receives (``clinical_grounding.chat_grounding``), or None when the question gets none.

    Mirrors the server only when it has corpus rules and the DUR dataset loaded; otherwise chat gets no DUR at all.
    """
    findings = question_findings(question)
    proposed = dur_checker().find_drugs(question) if has_medication_context(question) else ()
    if not findings and not proposed and not asks_for_recommendation(question):
        return None
    return _dur(emr, pregnant=findings.get('pregnancy'), proposed_text=question, conflict=False)


def recommendation_dur(emr: Mapping[str, Any], *, clinical_note: str | None, question: str | None,
                       findings: Mapping[str, bool] | None) -> dict[str, Any]:
    """``assessment.medication_alerts`` of the recommendation API for the deterministic inputs.

    The memo's LLM finding extraction is not reproduced: cases state pregnancy in words the deterministic reader
    understands, so a compliant extraction agrees with it.
    """
    note_findings, conflicts = merge_question_findings(question_findings(clinical_note), question)
    merged = {**note_findings, **(findings or {})}
    conflicts = {k: v for k, v in conflicts.items() if k not in (findings or {})}
    text = ' '.join(part for part in (clinical_note, question) if part)
    result = _dur(emr, pregnant=merged.get('pregnancy'), proposed_text=text, conflict='pregnancy' in conflicts)
    result['findings_used_pregnancy'] = merged.get('pregnancy')
    result['finding_conflicts'] = sorted(conflicts)
    return result


def calculations(emr: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    results, needed = calculate_review_aids(emr.get('calculationContext'))
    return [r.as_dict() for r in results], list(needed)


def proposed_names(text: str) -> Sequence[str]:
    return [drug.ingredient_name for drug in dur_checker().find_drugs(text)]
