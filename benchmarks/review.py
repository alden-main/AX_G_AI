"""Create review sheets; no semantic score is inferred from string matching."""
from __future__ import annotations

import json
from pathlib import Path

from benchmarks.criteria import DUR_SOURCE_CITATION, cited_numbers
from benchmarks.runner import sha
from benchmarks.schema import DATA, read_jsonl


def create(run_dir):
    run_dir = Path(run_dir)
    dest = run_dir / 'review.jsonl'
    if dest.exists():
        raise FileExistsError('review file already exists; preserve previous judgments')
    manifest = json.loads((DATA / 'manifest.json').read_text(encoding='utf-8'))
    if sha(DATA / 'gold.jsonl') != manifest['hashes']['gold.jsonl']:
        raise ValueError('reference data does not match dataset manifest')
    run_manifest = json.loads((run_dir / 'manifest.json').read_text(encoding='utf-8'))
    if run_manifest.get('suite', 'core') != 'core':
        raise ValueError('review sheets are for the core suite; recommendation runs are checked by report')
    if sha(DATA / 'inputs.jsonl') != run_manifest['input_sha256']:
        raise ValueError('run inputs and current dataset differ')
    gold = {x['case_id']: x for x in read_jsonl(DATA / 'gold.jsonl')}
    out = []
    for r in read_jsonl(run_dir / 'responses.jsonl'):
        if r['task'] not in ('summary', 'chat') or r['status'] != 'success':
            continue
        g = gold[r['case_id']]
        response = r['response']
        cited = cited_numbers(response['answer'])
        evidence = response.get('evidence') or []
        available = {e.get('n') for e in evidence if isinstance(e, dict)}
        out.append({'case_id': r['case_id'], 'task': r['task'], 'repetition': r['repetition'],
                    'answer': response['answer'], 'evidence': evidence, 'expected': g['expected_' + r['task']],
                    'forbidden_claims': g['forbidden_claims'],
                    'facts': [{**f, 'judgment': 'pending', 'reason': '', 'answer_location': ''}
                              for f in g['facts'] if r['task'] in f['required_for']],
                    'forbidden_checks': [{'claim': claim, 'judgment': 'pending', 'reason': '', 'answer_location': ''}
                                         for claim in g['forbidden_claims']],
                    # Automatic candidates only: a number that resolves is not a citation that supports the sentence.
                    'citation_check': {'numbers': cited, 'missing_numbers': sorted(set(cited) - available),
                                       'dur_source_marked': bool(DUR_SOURCE_CITATION.search(response['answer']))},
                    'claims': [], 'claim_extraction_complete': False, 'critical_review_complete': False,
                    'critical_errors': [], 'reviewer': '', 'reviewed_at': '', 'review_version': 1})
    dest.write_text(''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in out), encoding='utf-8')
    (run_dir / 'review_metadata.json').write_text(json.dumps({'gold_sha256': sha(DATA / 'gold.jsonl'),
        'responses_sha256': sha(run_dir / 'responses.jsonl'), 'review_status': 'draft'}, indent=2), encoding='utf-8')
    return dest
