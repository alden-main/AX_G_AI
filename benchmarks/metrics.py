"""Aggregate explicit reviewer judgments; pending judgments never count as passes."""
from __future__ import annotations

import math
from collections import Counter, defaultdict

from benchmarks.criteria import IMPORTANCE

ASSESSED = ('accurate', 'partial', 'omitted', 'distorted')  # 'indeterminate' and 'pending' are outside the denominator


def percentile(values, p):
    if not values:
        return None
    values = sorted(values)
    return values[math.ceil(p * len(values)) - 1]


def ratio(n, d):
    return {'numerator': n, 'denominator': d, 'value': n / d if d else None}


def distribution(values):
    return {'n': len(values), 'p50': percentile(values, .50), 'p95': percentile(values, .95),
            'p99': percentile(values, .99), 'max': max(values) if values else None}


def compare(left, right):
    def index(records):
        return {(r['case_id'], r['task'], r['repetition']): r for r in records if r['task'] in ('summary', 'chat')}
    a, b = index(left), index(right)
    pairs = []
    for key in sorted(a.keys() | b.keys()):
        x, y = a.get(key), b.get(key)
        pairs.append({'case_id': key[0], 'task': key[1], 'repetition': key[2],
                      'left_status': x['status'] if x else 'missing', 'right_status': y['status'] if y else 'missing',
                      'left_elapsed_ms': x.get('elapsed_ms') if x else None,
                      'right_elapsed_ms': y.get('elapsed_ms') if y else None})
    return pairs


def compare_quality(left, right):
    """Pair reviewed facts only; missing and pending values never become successes."""
    def index(rows):
        result = {}
        for row in rows:
            verified = bool(row.get('reviewer') and row.get('reviewed_at'))
            for fact in row.get('facts', []):
                k = (row['case_id'], row['task'], row['repetition'], fact['id'])
                if k in result:
                    raise ValueError('duplicate review fact')
                result[k] = fact.get('judgment', 'pending') if verified else 'pending'
        return result
    a, b = index(left), index(right)
    pairs, transitions = [], Counter()
    valid = {'accurate', 'partial', 'omitted', 'distorted', 'indeterminate'}
    for case, task, repetition, fact in sorted(a.keys() | b.keys()):
        k = (case, task, repetition, fact)
        x, y = a.get(k, 'missing'), b.get(k, 'missing')
        if x in valid and y in valid:
            transitions[f'{x} -> {y}'] += 1
        pairs.append({'case_id': case, 'task': task, 'repetition': repetition, 'fact_id': fact, 'left': x, 'right': y})
    def critical_index(rows):
        return {(r['case_id'], r['task'], r['repetition']): bool(r['critical_errors'])
                for r in rows if r.get('reviewer') and r.get('reviewed_at') and r.get('critical_review_complete')}
    def forbidden_index(rows):
        return {(r['case_id'], r['task'], r['repetition']): any(c['judgment'] == 'violated' for c in r['forbidden_checks'])
                for r in rows if r.get('reviewer') and r.get('reviewed_at') and r.get('forbidden_checks')
                and all(c.get('judgment', 'pending') != 'pending' for c in r['forbidden_checks'])}
    left_forbidden, right_forbidden = forbidden_index(left), forbidden_index(right)
    left_critical, right_critical = critical_index(left), critical_index(right)
    critical = [{'case_id': k[0], 'task': k[1], 'repetition': k[2], 'left_has_critical_error': left_critical[k],
                 'right_has_critical_error': right_critical[k]} for k in sorted(left_critical.keys() & right_critical.keys())]
    return {'pairs': pairs, 'paired_confirmed': sum(transitions.values()),
            'unpaired_or_pending': len(pairs) - sum(transitions.values()), 'transitions': dict(transitions),
            'critical_error_pairs': critical,
            'critical_unpaired': len(left_critical.keys() ^ right_critical.keys()),
            'forbidden_violation_pairs': [
                {'case_id': k[0], 'task': k[1], 'repetition': k[2], 'left_violated': left_forbidden[k], 'right_violated': right_forbidden[k]}
                for k in sorted(left_forbidden.keys() & right_forbidden.keys())],
            'forbidden_unpaired': len(left_forbidden.keys() ^ right_forbidden.keys())}


def report(records, reviews):
    output = {'tasks': {}, 'review_status': 'draft; clinical review required', 'percentile_method': 'nearest-rank',
              'token_metrics': 'N/A; SSE chunks are not tokens'}
    for task in ('prepare', 'summary', 'chat', 'recommendation'):
        subset = [r for r in records if r['task'] == task]
        if not subset:  # a recommendation run has no summary or chat, a core run no recommendation
            continue
        times = [r['elapsed_ms'] for r in subset if r['status'] == 'success' and 'elapsed_ms' in r]
        failures = [r['elapsed_ms'] for r in subset if r['status'] != 'success' and 'elapsed_ms' in r]
        output['tasks'][task] = {'statuses': dict(Counter(r['status'] for r in subset)),
                                 'success': ratio(len(times), len(subset)), 'latency_ms': distribution(times),
                                 'failure_latency_ms': failures}
        streams = [r for r in subset if r['status'] == 'success' and 'delta_offsets_ms' in r]
        if any('delta_offsets_ms' in r for r in subset):
            intervals = [x for r in streams for x in r['delta_intervals_ms']]
            output['tasks'][task]['streaming'] = {
                'first_text_ms': distribution([r['first_text_ms'] for r in streams if r['first_text_ms'] is not None]),
                'delta_intervals_pooled_ms': distribution(intervals),
                'delta_intervals_by_request': [
                    {'case_id': r['case_id'], 'repetition': r['repetition'], 'interval_ms': distribution(r['delta_intervals_ms'])}
                    for r in streams],
                'ttft_ms': None, 'token_intervals_ms': None, 'token_throughput_per_second': None,
            }
    expected_reviews = sum(r['task'] in ('summary', 'chat') and r['status'] == 'success' for r in records)
    counts = Counter()  # critical and important facts
    by_importance = defaultdict(Counter)
    by_source = defaultdict(Counter)
    claims = Counter()
    reviewable_claims = 0
    linked = 0
    citations = Counter()
    critical_complete = critical_cases = 0
    critical_types = Counter()
    forbidden = Counter()
    forbidden_complete = forbidden_cases = 0
    incomplete = 0
    for r in reviews:
        confirmed = bool(r.get('reviewer') and r.get('reviewed_at'))
        for f in r.get('facts', []):
            judgment = f.get('judgment', 'pending') if confirmed else 'pending'
            importance = f.get('importance', 'important')
            by_importance[importance][judgment] += 1
            by_source[(f.get('source') or {}).get('kind', 'emr')][judgment] += 1
            if importance in ('critical', 'important'):
                counts[judgment] += 1
        if not confirmed or not r.get('claim_extraction_complete'):
            incomplete += 1
        if confirmed:
            for c in r.get('claims', []):
                claims[c.get('judgment', 'pending')] += 1
                if c.get('source_required') is True:
                    reviewable_claims += 1
                    linked += bool(c.get('sources'))
                for citation in c.get('citations', []):
                    citations[citation.get('judgment', 'pending')] += 1
            if r.get('critical_review_complete'):
                critical_complete += 1
                critical_cases += bool(r.get('critical_errors'))
                critical_types.update(e.get('type', 'untyped') for e in r.get('critical_errors', []))
            checks = r.get('forbidden_checks', [])
            forbidden.update(c.get('judgment', 'pending') for c in checks)
            if checks and all(c.get('judgment', 'pending') != 'pending' for c in checks):
                forbidden_complete += 1
                forbidden_cases += any(c['judgment'] == 'violated' for c in checks)
        else:
            forbidden.update('pending' for _ in r.get('forbidden_checks', []))
    assessed = sum(counts[k] for k in ASSESSED)
    mentioned = sum(counts[k] for k in ('accurate', 'partial', 'distorted'))
    assessed_claims = sum(claims[k] for k in ('supported', 'unsupported', 'contradicted'))
    assessed_citations = citations['supports'] + citations['does_not_support']
    def recall(counter):
        return ratio(counter['accurate'], sum(counter[k] for k in ASSESSED))
    output['quality'] = {'fact_counts': dict(counts), 'important_fact_recall': ratio(counts['accurate'], assessed),
        'critical_fact_recall': recall(by_importance['critical']),
        'fact_recall_by_importance': {k: recall(by_importance[k]) for k in IMPORTANCE},
        'fact_recall_by_source': {k: recall(v) for k, v in sorted(by_source.items())},
        'important_distortion': ratio(counts['distorted'], mentioned), 'claim_counts': dict(claims),
        'unsupported_claims': ratio(claims['unsupported'], assessed_claims),
        'contradicted_claims': ratio(claims['contradicted'], assessed_claims),
        'source_linkage': ratio(linked, reviewable_claims), 'citation_accuracy': ratio(citations['supports'], assessed_citations),
        'citation_counts': dict(citations), 'critical_error_cases': ratio(critical_cases, critical_complete),
        'critical_error_types': dict(critical_types),
        # Cases with at least one violated forbidden claim, over cases whose forbidden checks are all judged.
        'forbidden_claim_violation_cases': ratio(forbidden_cases, forbidden_complete), 'forbidden_check_counts': dict(forbidden),
        'claims_incomplete_reviews': incomplete, 'reviews': len(reviews),
        'fact_completion': ratio(assessed + counts['indeterminate'], sum(counts.values())),
        'missing_review_sheets': max(0, expected_reviews - len(reviews)),
        'provisional': bool(counts['pending'] or forbidden['pending'] or incomplete or len(reviews) != expected_reviews)}
    return output
