"""Automatic structured checks of recommendation responses against service-derived expectations.

What is compared is deterministic output (``assessment.medication_alerts``, ``findings_used.pregnancy``,
``finding_conflicts``), so an exact comparison is a valid score of that output. It says nothing about the answer
text, the corpus rule items or clinical appropriateness; the ``[출처: 심평원 DUR 목록]`` check is a candidate only.
A server without DUR, or with a different DUR dataset, is reported as not comparable rather than as an error.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from benchmarks.criteria import DUR_SOURCE_CITATION
from benchmarks.derive import alert_key, alert_summary
from benchmarks.runner import sha
from benchmarks.schema import DATA, read_jsonl


def _ratio(n, d):
    return {'numerator': n, 'denominator': d, 'value': n / d if d else None}


def check_case(expected, response, expected_source_id):
    assessment = response.get('assessment') or {}
    alerts = assessment.get('medication_alerts')
    if alerts is None:
        return {'status': 'dur_unavailable'}
    source_id = (alerts.get('source') or {}).get('id')
    if source_id != expected_source_id:
        return {'status': 'dur_dataset_differs', 'server_source_id': source_id}
    want = {alert_key(a): a for a in expected['alerts']}
    got = {alert_key(s): s for s in (alert_summary(a) for a in alerts.get('alerts', []))}
    unmapped = sorted(u.get('med_code') or '' for u in alerts.get('unmapped', []))
    pregnancy = (assessment.get('findings_used') or {}).get('pregnancy')
    conflicts = sorted(assessment.get('finding_conflicts') or {})
    result = {'status': 'compared',
              'missing_alerts': [want[k] for k in sorted(want.keys() - got.keys())],
              'unexpected_alerts': [got[k] for k in sorted(got.keys() - want.keys())],
              'matched_alerts': len(want.keys() & got.keys()), 'expected_alerts': len(want),
              'unmapped_match': unmapped == expected['unmapped'],
              'pregnancy_match': pregnancy == expected['findings_used_pregnancy'],
              'conflicts_match': conflicts == expected['finding_conflicts'],
              # Candidate only: a marker in the text does not show the alert was stated correctly.
              'dur_source_marked': bool(DUR_SOURCE_CITATION.search(response.get('answer', '')))}
    result['exact'] = (not result['missing_alerts'] and not result['unexpected_alerts'] and result['unmapped_match']
                       and result['pregnancy_match'] and result['conflicts_match'])
    return result


def score(run_dir):
    run_dir = Path(run_dir)
    run = json.loads((run_dir / 'manifest.json').read_text(encoding='utf-8'))
    if run.get('suite') != 'recommendation':
        raise ValueError('not a recommendation run')
    dataset = json.loads((DATA / 'manifest.json').read_text(encoding='utf-8'))
    gold_path = DATA / 'recommendation_gold.jsonl'
    if (sha(DATA / 'recommendation_inputs.jsonl') != run['input_sha256'] or
            sha(gold_path) != dataset['hashes']['recommendation_gold.jsonl']):
        raise ValueError('dataset integrity check failed')
    gold = {g['case_id']: g['expected_medication_alerts'] for g in read_jsonl(gold_path)}
    scenarios = {c['case_id']: c['scenario'] for c in read_jsonl(DATA / 'recommendation_inputs.jsonl')}
    source_id = (run.get('expected_dur_snapshot') or {}).get('source_id')
    cases = []
    for record in read_jsonl(run_dir / 'responses.jsonl'):
        if record['task'] != 'recommendation':
            continue
        base = {'case_id': record['case_id'], 'repetition': record['repetition'], 'scenario': scenarios[record['case_id']]}
        if record['status'] != 'success':
            cases.append({**base, 'status': f"run_{record['status']}"})
            continue
        cases.append({**base, **check_case(gold[record['case_id']], record['response'], source_id)})
    compared = [c for c in cases if c['status'] == 'compared']
    expected_total = sum(c['expected_alerts'] for c in compared)
    matched = sum(c['matched_alerts'] for c in compared)
    unexpected = sum(len(c['unexpected_alerts']) for c in compared)
    by_scenario = {}
    for scenario in sorted({c['scenario'] for c in compared}):
        subset = [c for c in compared if c['scenario'] == scenario]
        by_scenario[scenario] = _ratio(sum(c['exact'] for c in subset), len(subset))
    missing_types = Counter(a['type'] for c in compared for a in c['missing_alerts'])
    with_alerts = [c for c in compared if c['expected_alerts']]
    return {'check': 'deterministic structured comparison; not a clinical judgment',
            'statuses': dict(Counter(c['status'] for c in cases)),
            'exact_cases': _ratio(sum(c['exact'] for c in compared), len(compared)),
            'alert_recall': _ratio(matched, expected_total), 'alert_precision': _ratio(matched, matched + unexpected),
            'missing_alert_types': dict(missing_types),
            'pregnancy_reading': _ratio(sum(c['pregnancy_match'] for c in compared), len(compared)),
            'conflict_reporting': _ratio(sum(c['conflicts_match'] for c in compared), len(compared)),
            'unmapped_reporting': _ratio(sum(c['unmapped_match'] for c in compared), len(compared)),
            'dur_source_marked_candidates': _ratio(sum(c['dur_source_marked'] for c in with_alerts), len(with_alerts)),
            'by_scenario': by_scenario, 'cases': cases}
