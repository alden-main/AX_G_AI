"""Versioned reviewer snapshots. Hash chain detects accidental edits, not malicious rewriting."""
from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path

from benchmarks.criteria import CITATION_JUDGMENTS as CITATION
from benchmarks.criteria import CLAIM_JUDGMENTS as CLAIM
from benchmarks.criteria import CRITICAL_ERROR_TYPES
from benchmarks.criteria import FACT_JUDGMENTS as FACT
from benchmarks.criteria import FORBIDDEN_JUDGMENTS as FORBIDDEN
from benchmarks.runner import sha
from benchmarks.schema import DATA, read_jsonl


def key(row):
    return row['case_id'], row['task'], row['repetition']


def validate(run_dir, rows, *, confirmed=False):
    run_dir = Path(run_dir)
    meta = json.loads((run_dir / 'review_metadata.json').read_text(encoding='utf-8'))
    run = json.loads((run_dir / 'manifest.json').read_text(encoding='utf-8'))
    if sha(DATA / 'inputs.jsonl') != run['input_sha256'] or sha(DATA / 'gold.jsonl') != meta['gold_sha256'] or sha(run_dir / 'responses.jsonl') != meta['responses_sha256']:
        raise ValueError('dataset or response integrity check failed')
    references = {g['case_id']: g for g in read_jsonl(DATA / 'gold.jsonl')}
    responses = {key(r): r for r in read_jsonl(run_dir / 'responses.jsonl') if r['task'] in ('summary', 'chat') and r['status'] == 'success'}
    if len(rows) != len(responses) or len({key(r) for r in rows}) != len(rows) or {key(r) for r in rows} != set(responses):
        raise ValueError('review rows must match successful responses')
    for row in rows:
        ref, response = references[row['case_id']], responses[key(row)]['response']
        facts = [f for f in ref['facts'] if row['task'] in f['required_for']]
        if row['answer'] != response['answer'] or row['evidence'] != (response.get('evidence') or []) or row['expected'] != ref['expected_' + row['task']] or row['forbidden_claims'] != ref['forbidden_claims']:
            raise ValueError('immutable review content changed')
        if len(row['facts']) != len(facts) or {f['id'] for f in row['facts']} != {f['id'] for f in facts}:
            raise ValueError('review fact IDs differ')
        original = {f['id']: f for f in facts}
        for fact in row['facts']:
            if any(fact.get(k) != v for k, v in original[fact['id']].items()) or fact.get('judgment') not in FACT:
                raise ValueError('invalid or modified fact')
            if fact['judgment'] != 'pending' and not str(fact.get('reason', '')).strip():
                raise ValueError('fact judgment requires reason')
        if not isinstance(row.get('claims'), list) or type(row.get('claim_extraction_complete')) is not bool or type(row.get('critical_review_complete')) is not bool or not isinstance(row.get('critical_errors'), list):
            raise ValueError('invalid review flags')
        for claim in row['claims']:
            if (claim.get('judgment') not in CLAIM or type(claim.get('source_required')) is not bool or
                not isinstance(claim.get('sources'), list) or not isinstance(claim.get('citations'), list) or
                not isinstance(claim.get('text'), str) or not claim['text'].strip()):
                raise ValueError('invalid claim')
            if any(c.get('judgment') not in CITATION for c in claim['citations']):
                raise ValueError('invalid citation judgment')
        checks = row.get('forbidden_checks')
        if not isinstance(checks, list) or [c.get('claim') for c in checks] != ref['forbidden_claims']:
            raise ValueError('forbidden checks must list the reference forbidden claims')
        for check in checks:
            if check.get('judgment') not in FORBIDDEN:
                raise ValueError('invalid forbidden claim judgment')
            if check['judgment'] != 'pending' and not str(check.get('reason', '')).strip():
                raise ValueError('forbidden claim judgment requires reason')
        for error in row['critical_errors']:
            if (not isinstance(error, dict) or error.get('type') not in CRITICAL_ERROR_TYPES or
                    not str(error.get('description', '')).strip()):
                raise ValueError(f'critical error needs a type of {sorted(CRITICAL_ERROR_TYPES)} and a description')
        if confirmed and (not str(row.get('reviewer', '')).strip() or not str(row.get('reviewed_at', '')).strip()):
            raise ValueError('reviewer and reviewed_at required')
    return rows


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def load_history(run_dir):
    folder = Path(run_dir) / 'review_history'
    prev = None
    result = []
    for index, path in enumerate(sorted(folder.glob('*.json')), 1):
        if not re.fullmatch(r'\d{6}\.json', path.name) or path.name != f'{index:06d}.json':
            raise ValueError('history sequence invalid')
        entry = json.loads(path.read_text(encoding='utf-8'))
        claimed = entry.pop('hash')
        if entry['previous_hash'] != prev or digest(entry) != claimed or entry['sequence'] != index:
            raise ValueError('history integrity check failed')
        entry['hash'] = claimed
        prev = claimed
        result.append(entry)
    return result


def commit(run_dir, review_file, actor):
    run_dir = Path(run_dir)
    if not actor.strip():
        raise ValueError('actor required')
    rows = validate(run_dir, read_jsonl(Path(review_file)))
    entries = load_history(run_dir)
    entry = {'sequence': len(entries) + 1, 'previous_hash': entries[-1]['hash'] if entries else None,
             'actor': actor, 'committed_at': datetime.now(UTC).isoformat(), 'rows': rows}
    entry['hash'] = digest(entry)
    folder = run_dir / 'review_history'
    folder.mkdir(exist_ok=True)
    path = folder / f"{entry['sequence']:06d}.json"
    with path.open('x', encoding='utf-8') as stream:
        json.dump(entry, stream, ensure_ascii=False, indent=2)
    return path


def effective(run_dir):
    entries = load_history(run_dir)
    if entries:
        return validate(run_dir, entries[-1]['rows']), 'history'
    path = Path(run_dir) / 'review.jsonl'
    return (validate(run_dir, read_jsonl(path)), 'uncommitted') if path.exists() else ([], 'none')
