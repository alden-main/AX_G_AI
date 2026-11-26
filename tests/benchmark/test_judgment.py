from __future__ import annotations

import json

import httpx
import pytest

from benchmarks.history import commit, effective, load_history
from benchmarks.judge import evaluate
from benchmarks.metrics import compare_quality, report
from benchmarks.review import create
from benchmarks.runner import run
from benchmarks.schema import read_jsonl
from tests.benchmark.conftest import sse


@pytest.fixture
def prepared(output_root):
    def handler(request):
        if request.url.path.endswith('patient-context/stream'):
            return httpx.Response(503)
        return sse('환자 기록을 확인했습니다.')
    out = run('http://localhost', 'validation', output_root, label='judge-fake', transport=httpx.MockTransport(handler))
    create(out)
    return out


def test_judge_proposals_do_not_change_review(prepared):
    before = (prepared / 'review.jsonl').read_bytes()
    def handler(request):
        data = json.loads(json.loads(request.content)['messages'][1]['content'])
        facts = [{'id': f['id'], 'judgment': 'accurate', 'reason': '검토 필요', 'answer_quote': ''} for f in data['facts']]
        forbidden = [{'claim': c, 'judgment': 'not_violated', 'reason': '해당 서술 없음', 'answer_quote': ''} for c in data['forbidden_claims']]
        content = json.dumps({'facts': facts, 'claims': [], 'forbidden': forbidden})
        return httpx.Response(200, json={'choices': [{'message': {'content': content}}]})
    out = evaluate(prepared, url='http://judge/v1/chat/completions', model='test', transport=httpx.MockTransport(handler))
    assert all(x['status'] == 'proposed' for x in read_jsonl(out))
    assert (prepared / 'review.jsonl').read_bytes() == before
    meta = json.loads(out.with_suffix('.meta.json').read_text())
    assert meta['status'] == 'unverified_proposal'
    assert meta['proposed_fact_recall']['denominator'] > 0
    assert meta['forbidden_counts']['not_violated'] > 0
    assert all(r['reviewer'] == '' for r in effective(prepared)[0])


def test_invalid_judge_output_is_not_a_score(prepared):
    response = {'choices': [{'message': {'content': '{"facts": []}'}}]}
    out = evaluate(prepared, url='http://judge/v1/chat/completions', model='test',
                   transport=httpx.MockTransport(lambda request: httpx.Response(200, json=response)))
    assert all(x['status'] == 'judge_error' for x in read_jsonl(out))
    assert json.loads(out.with_suffix('.meta.json').read_text())['proposed_fact_recall']['value'] is None


def test_version_history_preserves_and_detects_tampering(prepared):
    path = prepared / 'review.jsonl'
    rows = read_jsonl(path)
    rows[0]['reviewer'], rows[0]['reviewed_at'] = 'reviewer', '2026-11-19'
    rows[0]['facts'][0].update(judgment='accurate', reason='원문과 동일')
    path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
    first = commit(prepared, path, 'reviewer')
    rows[0]['facts'][0].update(judgment='distorted', reason='재검토 후 수정')
    path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
    commit(prepared, path, 'reviewer')
    assert load_history(prepared)[0]['rows'][0]['facts'][0]['judgment'] == 'accurate'
    assert effective(prepared)[0][0]['facts'][0]['judgment'] == 'distorted'
    saved = json.loads(first.read_text())
    saved['actor'] = 'changed'
    first.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match='integrity'):
        load_history(prepared)


def test_modified_reference_and_bad_decisions_rejected(prepared):
    path = prepared / 'review.jsonl'
    rows = read_jsonl(path)
    rows[0]['expected'] = 'tampered'
    path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
    with pytest.raises(ValueError, match='immutable'):
        commit(prepared, path, 'reviewer')


def test_quality_pairs_do_not_treat_unreviewed_as_success():
    def row(judgment, reviewer='r'):
        return {'case_id': '1', 'task': 'summary', 'repetition': 1, 'reviewer': reviewer, 'reviewed_at': 'now',
                'facts': [{'id': 'f', 'judgment': judgment}]}
    assert compare_quality([row('omitted')], [row('accurate')])['transitions'] == {'omitted -> accurate': 1}
    assert compare_quality([row('accurate', '')], [row('accurate')])['unpaired_or_pending'] == 1


def test_forbidden_checks_and_typed_critical_errors(prepared):
    path = prepared / 'review.jsonl'
    rows = read_jsonl(path)
    rows[0].update(reviewer='reviewer', reviewed_at='2026-11-19', critical_review_complete=True)
    rows[0]['forbidden_checks'][0].update(judgment='violated')
    path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
    with pytest.raises(ValueError, match='requires reason'):
        commit(prepared, path, 'reviewer')
    rows[0]['forbidden_checks'][0]['reason'] = '금지 서술 포함'
    rows[0]['critical_errors'] = ['설명만 있는 오류']
    path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
    with pytest.raises(ValueError, match='critical error'):
        commit(prepared, path, 'reviewer')
    rows[0]['critical_errors'] = [{'type': 'forbidden_claim', 'description': '금지 주장 서술'}]
    rows[0]['forbidden_checks'][0]['claim'] = 'changed'
    path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
    with pytest.raises(ValueError, match='forbidden checks'):
        commit(prepared, path, 'reviewer')
    rows[0]['forbidden_checks'][0]['claim'] = rows[0]['forbidden_claims'][0]
    for check in rows[0]['forbidden_checks'][1:]:
        check.update(judgment='not_violated', reason='해당 서술 없음')
    path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
    commit(prepared, path, 'reviewer')
    quality = report(read_jsonl(prepared / 'responses.jsonl'), effective(prepared)[0])['quality']
    assert quality['critical_error_types'] == {'forbidden_claim': 1}
    assert quality['forbidden_claim_violation_cases'] == {'numerator': 1, 'denominator': 1, 'value': 1.0}
    assert quality['forbidden_check_counts']['violated'] == 1
    assert quality['provisional']
