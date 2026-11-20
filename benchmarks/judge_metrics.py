"""Provisional automatic scores from judge-model proposals, kept apart from the reviewer-confirmed report.

The definitions match ``metrics.report`` so the two can be read side by side, but every number here is an
unverified model proposal: it can rank runs quickly, it cannot certify a run. Comparisons are only made between runs
judged by the same model and prompt version on the same inputs and references, and the agreement with the reviewers'
confirmed judgments is reported so the judge's reliability is visible next to its scores.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from benchmarks.criteria import IMPORTANCE
from benchmarks.history import effective
from benchmarks.metrics import ASSESSED, ratio
from benchmarks.runner import sha
from benchmarks.schema import DATA, read_jsonl

NOTICE = 'unverified judge-model proposals; not reviewer-confirmed and not a clinical judgment'


def _key(row):
    return row['case_id'], row['task'], row['repetition']


def jobs(run_dir):
    """Judge jobs of a run, oldest first."""
    metas = [json.loads(path.read_text(encoding='utf-8')) for path in Path(run_dir).glob('judge-*.meta.json')]
    return sorted(metas, key=lambda meta: meta['created_at'])


def load(run_dir, job=None):
    """A judge job's metadata and rows (the latest job unless ``job`` is given), checked against the current data."""
    run_dir = Path(run_dir)
    available = jobs(run_dir)
    if not available:
        raise ValueError('no judge results in this run (run the judge command first)')
    meta = next((m for m in available if m['job_id'] == job), None) if job else available[-1]
    if meta is None:
        raise ValueError(f'judge job not found: {job}')
    if (sha(run_dir / 'responses.jsonl') != meta['responses_sha256'] or sha(DATA / 'gold.jsonl') != meta['gold_sha256']
            or sha(DATA / 'inputs.jsonl') != meta['input_sha256']):
        raise ValueError('judge results do not match the current responses or dataset')
    return meta, read_jsonl(run_dir / f"judge-{meta['job_id']}.jsonl")


def _references():
    """(case_id, task) → fact id → reference fact (importance, source)."""
    facts = {}
    for g in read_jsonl(DATA / 'gold.jsonl'):
        for task in ('summary', 'chat'):
            facts[(g['case_id'], task)] = {f['id']: f for f in g['facts'] if task in f['required_for']}
    return facts


def _recall(counter):
    return ratio(counter['accurate'], sum(counter[k] for k in ASSESSED))


def score(rows, records):
    """Aggregate proposals with the definitions of ``metrics.report`` (critical + important = 'important')."""
    references = _references()
    attempted = sum(r['task'] in ('summary', 'chat') for r in records)
    proposed = [r for r in rows if r['status'] == 'proposed']
    by_importance, by_source, counts = defaultdict(Counter), defaultdict(Counter), Counter()
    claims, forbidden = Counter(), Counter()
    violation_cases = 0
    for row in proposed:
        facts = references[(row['case_id'], row['task'])]
        for item in row['facts']:
            fact = facts[item['id']]
            by_importance[fact['importance']][item['judgment']] += 1
            by_source[fact['source']['kind']][item['judgment']] += 1
            if fact['importance'] in ('critical', 'important'):
                counts[item['judgment']] += 1
        claims.update(c['judgment'] for c in row['claims'])
        forbidden.update(f['judgment'] for f in row['forbidden'])
        violation_cases += any(f['judgment'] == 'violated' for f in row['forbidden'])
    mentioned = sum(counts[k] for k in ('accurate', 'partial', 'distorted'))
    assessed_claims = sum(claims[k] for k in ('supported', 'unsupported', 'contradicted'))
    return {
        'status': NOTICE,
        # Failed API calls have no answer to judge: they lower coverage, they are not counted as omissions.
        'coverage': {'proposed': len(proposed), 'judge_errors': sum(r['status'] == 'judge_error' for r in rows),
                     'run_failures': attempted - len(rows), 'attempted_tasks': attempted},
        'fact_counts': dict(counts),
        'important_fact_recall': _recall(counts),
        'critical_fact_recall': _recall(by_importance['critical']),
        'fact_recall_by_importance': {k: _recall(by_importance[k]) for k in IMPORTANCE},
        'fact_recall_by_source': {k: _recall(v) for k, v in sorted(by_source.items())},
        'important_distortion': ratio(counts['distorted'], mentioned),
        'forbidden_claim_violation_cases': ratio(violation_cases, len(proposed)),
        'forbidden_counts': dict(forbidden),
        'claim_counts': dict(claims),
        'unsupported_claims': ratio(claims['unsupported'], assessed_claims),
        'contradicted_claims': ratio(claims['contradicted'], assessed_claims),
    }


def agreement(rows, reviews):
    """How often the judge's proposals equal the reviewers' confirmed judgments (facts and forbidden claims)."""
    confirmed = {_key(r): r for r in reviews if r.get('reviewer') and r.get('reviewed_at')}
    facts, forbidden = Counter(), Counter()
    for row in rows:
        review = confirmed.get(_key(row))
        if row['status'] != 'proposed' or review is None:
            continue
        human = {f['id']: f['judgment'] for f in review['facts'] if f.get('judgment', 'pending') != 'pending'}
        for item in row['facts']:
            if item['id'] in human:
                facts[(human[item['id']], item['judgment'])] += 1
        human_forbidden = {c['claim']: c['judgment'] for c in review.get('forbidden_checks', []) if c.get('judgment', 'pending') != 'pending'}
        for item in row['forbidden']:
            if item['claim'] in human_forbidden:
                forbidden[(human_forbidden[item['claim']], item['judgment'])] += 1

    def summary(counter):
        total = sum(counter.values())
        same = sum(n for (human, judge), n in counter.items() if human == judge)
        return {'agreement': ratio(same, total),
                'confusion_human_to_judge': {f'{human} -> {judge}': n for (human, judge), n in sorted(counter.items())}}
    return {'facts': summary(facts), 'forbidden_claims': summary(forbidden),
            'note': 'only items the reviewers confirmed; low agreement means the provisional scores are not reliable'}


def report(run_dir, job=None):
    run_dir = Path(run_dir)
    meta, rows = load(run_dir, job)
    reviews, review_source = effective(run_dir)
    return {'job_id': meta['job_id'], 'judge_model': meta['model'], 'prompt_version': meta['prompt_version'],
            'created_at': meta['created_at'], **score(rows, read_jsonl(run_dir / 'responses.jsonl')),
            'agreement_with_reviewers': agreement(rows, reviews), 'review_source': review_source}


def compare(left_dir, right_dir, left_job=None, right_job=None):
    """Pair two runs' proposals fact by fact; only for the same judge model, prompt version and dataset."""
    (left_meta, left_rows), (right_meta, right_rows) = load(left_dir, left_job), load(right_dir, right_job)
    differences = {k: [left_meta.get(k), right_meta.get(k)] for k in ('model', 'prompt_version', 'temperature', 'input_sha256', 'gold_sha256')
                   if left_meta.get(k) != right_meta.get(k)}
    result = {'status': NOTICE, 'jobs': [left_meta['job_id'], right_meta['job_id']], 'judge_differences': differences}
    if differences:
        result.update(comparable=False, note='different judge or dataset: provisional scores are not comparable')
        return result

    def index(rows):
        return {(*_key(r), f['id']): f['judgment'] for r in rows if r['status'] == 'proposed' for f in r['facts']}

    def violations(rows):
        return {_key(r): any(f['judgment'] == 'violated' for f in r['forbidden']) for r in rows if r['status'] == 'proposed'}
    a, b = index(left_rows), index(right_rows)
    transitions = Counter(f'{a[k]} -> {b[k]}' for k in a.keys() & b.keys() if a[k] != b[k])
    va, vb = violations(left_rows), violations(right_rows)
    left_records = read_jsonl(Path(left_dir) / 'responses.jsonl')
    right_records = read_jsonl(Path(right_dir) / 'responses.jsonl')
    left_score, right_score = score(left_rows, left_records), score(right_rows, right_records)
    headline = ('important_fact_recall', 'critical_fact_recall', 'important_distortion', 'forbidden_claim_violation_cases',
                'unsupported_claims', 'contradicted_claims')
    result.update(
        comparable=True,
        summary={k: {'left': left_score[k], 'right': right_score[k]} for k in headline},
        paired_facts=len(a.keys() & b.keys()), unpaired_facts=len(a.keys() ^ b.keys()),
        changed_fact_judgments=dict(transitions),
        forbidden_violation_changes=[{'case_id': k[0], 'task': k[1], 'repetition': k[2], 'left_violated': va[k], 'right_violated': vb[k]}
                                     for k in sorted(va.keys() & vb.keys()) if va[k] != vb[k]],
    )
    return result
