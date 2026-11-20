from __future__ import annotations

import json

import httpx
import pytest

from benchmarks import judge_metrics
from benchmarks.history import commit
from benchmarks.judge import evaluate
from benchmarks.review import create
from benchmarks.runner import run
from benchmarks.schema import DATA, read_jsonl

GOLD = {g['case_id']: g for g in read_jsonl(DATA / 'gold.jsonl')}


def prepared_run(output_root, *, fail_first_summary=False):
    calls = []
    def handler(request):
        calls.append(request.url.path)
        if fail_first_summary and request.url.path.endswith('/emr-summary') and calls.count(request.url.path) == 1:
            return httpx.Response(503)
        if request.url.path.endswith('patient-context'):
            return httpx.Response(200, json={'status': 'ready'})
        return httpx.Response(200, json={'answer': '환자 기록을 확인했습니다.'})
    out = run('http://localhost', 'validation', output_root, label='judge-score', transport=httpx.MockTransport(handler))
    create(out)
    return out


def judge(run_dir, decide, *, model='judge-a', broken=()):
    """A fake judge that proposes ``decide(fact)`` for each fact, and returns invalid output for ``broken`` cases."""
    def handler(request):
        data = json.loads(json.loads(request.content)['messages'][1]['content'])
        case_id = next(cid for cid, g in GOLD.items() if g['expected_summary'] == data['expected'] or g['expected_chat'] == data['expected'])
        if case_id in broken:
            return httpx.Response(200, json={'choices': [{'message': {'content': '{}'}}]})
        facts = [{'id': f['id'], 'judgment': decide(f), 'reason': '제안', 'answer_quote': ''} for f in data['facts']]
        forbidden = [{'claim': c, 'judgment': 'violated' if i == 0 and case_id.endswith('1') else 'not_violated',
                      'reason': '제안', 'answer_quote': ''} for i, c in enumerate(data['forbidden_claims'])]
        claims = [{'text': '환자 기록을 확인했습니다.', 'judgment': 'supported', 'reason': '제안'}]
        content = json.dumps({'facts': facts, 'claims': claims, 'forbidden': forbidden}, ensure_ascii=False)
        return httpx.Response(200, json={'choices': [{'message': {'content': content}}]})
    return evaluate(run_dir, url='http://judge/v1/chat/completions', model=model, transport=httpx.MockTransport(handler))


def test_scores_follow_proposals_and_report_coverage(output_root):
    out = prepared_run(output_root, fail_first_summary=True)
    validation = [c['case_id'] for c in read_jsonl(DATA / 'inputs.jsonl') if c['split'] == 'validation']
    broken = {validation[-1]}  # its summary and chat get invalid judge output
    judge(out, lambda f: 'omitted' if f['importance'] == 'critical' else 'accurate', broken=broken)
    result = judge_metrics.report(out)
    assert result['status'].startswith('unverified')
    coverage = result['coverage']
    assert coverage['run_failures'] == 1 and coverage['judge_errors'] == 2
    assert coverage['proposed'] + coverage['judge_errors'] + coverage['run_failures'] == coverage['attempted_tasks'] == 50
    assert result['fact_recall_by_importance']['critical']['denominator'] > 0
    assert result['critical_fact_recall']['value'] == 0.0
    assert result['fact_recall_by_importance']['minor']['value'] == 1.0
    assert 0 < result['important_fact_recall']['value'] < 1
    assert result['forbidden_claim_violation_cases']['numerator'] > 0
    assert result['unsupported_claims']['value'] == 0.0
    assert result['agreement_with_reviewers']['facts']['agreement']['denominator'] == 0


def test_agreement_with_confirmed_reviewer_judgments(output_root):
    out = prepared_run(output_root)
    path = out / 'review.jsonl'
    rows = read_jsonl(path)
    rows[0].update(reviewer='reviewer', reviewed_at='2026-11-20')
    rows[0]['facts'][0].update(judgment='omitted', reason='답변에 없음')
    for fact in rows[0]['facts'][1:]:
        fact.update(judgment='accurate', reason='일치')
    path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
    commit(out, path, 'reviewer')
    judge(out, lambda f: 'accurate')
    agreement = judge_metrics.report(out)['agreement_with_reviewers']['facts']
    total = len(rows[0]['facts'])
    assert agreement['agreement'] == {'numerator': total - 1, 'denominator': total, 'value': (total - 1) / total}
    assert agreement['confusion_human_to_judge']['omitted -> accurate'] == 1


def test_compare_only_with_the_same_judge(output_root):
    left, right = prepared_run(output_root), prepared_run(output_root)
    judge(left, lambda f: 'omitted')
    judge(right, lambda f: 'accurate')
    result = judge_metrics.compare(left, right)
    assert result['comparable']
    assert result['summary']['important_fact_recall']['left']['value'] == 0.0
    assert result['summary']['important_fact_recall']['right']['value'] == 1.0
    assert result['changed_fact_judgments'] == {'omitted -> accurate': result['paired_facts']}
    judge(right, lambda f: 'accurate', model='judge-b')  # the latest job now uses another judge model
    different = judge_metrics.compare(left, right)
    assert not different['comparable'] and different['judge_differences']['model'] == ['judge-a', 'judge-b']
    older = judge_metrics.jobs(right)[0]['job_id']
    assert judge_metrics.compare(left, right, right_job=older)['comparable']


def test_stale_or_missing_judge_results_are_rejected(output_root):
    out = prepared_run(output_root)
    with pytest.raises(ValueError, match='no judge results'):
        judge_metrics.report(out)
    judge(out, lambda f: 'accurate')
    with pytest.raises(ValueError, match='judge job not found'):
        judge_metrics.report(out, 'missing')
    responses = out / 'responses.jsonl'
    responses.write_text(responses.read_text() + '\n')
    with pytest.raises(ValueError, match='do not match'):
        judge_metrics.report(out)
