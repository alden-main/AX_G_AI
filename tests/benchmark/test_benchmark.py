from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from benchmarks.criteria import cited_numbers
from benchmarks.metrics import percentile, report
from benchmarks.review import create
from benchmarks.runner import run
from benchmarks.schema import DATA, contains_value, read_jsonl, validate
from tests.benchmark.conftest import sse


def load():
    return read_jsonl(DATA / 'inputs.jsonl'), read_jsonl(DATA / 'gold.jsonl')


def test_dataset_and_integrity():
    cases, gold = load()
    result = validate(cases, gold)
    assert result['cases'] == 100
    assert set(result['facts']) == {'critical', 'important', 'minor'}
    emr_fact = next(f for f in gold[0]['facts'] if f['source']['kind'] == 'emr')
    emr_fact['source']['value'] = 'wrong'
    with pytest.raises(ValueError, match='source mismatch'):
        validate(cases, gold)


def test_dataset_uses_the_emr_contract_lists_and_dur_codes():
    cases, gold = load()
    keys = {k for c in cases for k in c['emr_payload']}
    assert {'bloodPressureList', 'bloodSugarList', 'oxygenSaturationList', 'glycatedHemoglobinList', 'labResultList',
            'medicineDtailList', 'diagnosisList'} <= keys
    assert all('examCode' in row for c in cases for row in c['emr_payload'].get('labResultList', []))
    meds = [m for c in cases for m in c['emr_payload'].get('medicineDtailList', [])]
    assert all({'medCode', 'dosage', 'frequency', 'durationDays', 'instructions'} <= set(m) for m in meds)
    assert any(len(c['emr_payload'].get('medicineDtailList', [])) >= 3 for c in cases)
    kinds = {f['source']['kind'] for g in gold for f in g['facts']}
    assert {'dur', 'calculation', 'recorded_none', 'absent', 'calculation_needed', 'answer_requirement'} <= kinds


def test_duplicates_shared_templates_and_stale_dur_are_rejected():
    cases, gold = load()
    duplicate = [dict(c) for c in cases]
    duplicate[1] = {**duplicate[1], 'emr_payload': {**duplicate[0]['emr_payload'], 'patient': duplicate[1]['emr_payload']['patient']},
                    'question': duplicate[0]['question']}
    with pytest.raises(ValueError, match='duplicate clinical content'):
        validate(duplicate, gold)
    shared = [dict(c) for c in cases]
    holdout = next(i for i, c in enumerate(shared) if c['split'] == 'holdout')
    development = next(c for c in shared if c['split'] == 'development')
    shared[holdout] = {**shared[holdout], 'question_template': development['question_template']}
    with pytest.raises(ValueError, match='templates are shared'):
        validate(shared, gold)
    stale = read_jsonl(DATA / 'gold.jsonl')
    reference = next(g for g in stale if any(f['source']['kind'] == 'dur' for f in g['facts']))
    reference['facts'] = [f for f in reference['facts'] if f['source']['kind'] != 'dur']
    with pytest.raises(ValueError, match='DUR facts differ'):
        validate(cases, stale)


def test_no_gold_sent_and_failure_recorded(output_root):
    bodies = []
    def handler(request):
        body = json.loads(request.content)
        bodies.append(body)
        if request.url.path.endswith('/patient-context/stream'):
            return httpx.Response(503, json={'code': 'failure'})
        return sse()
    out = run('http://localhost', 'validation', output_root, label='fake', transport=httpx.MockTransport(handler))
    records = read_jsonl(out / 'responses.jsonl')
    assert len(records) == 75
    assert sum(r['status'] == 'prepare_error' for r in records) == 25
    assert all(set(body) == {'patient_id', 'emr_payload'} for body in bodies)
    assert all('expected_summary' not in json.dumps(body) for body in bodies)
    path = create(out)
    assert len(read_jsonl(path)) == 25
    with pytest.raises(FileExistsError):
        create(out)
    assert report(records, [])['quality']['provisional']


def test_chat_body_allowlist(output_root):
    bodies = []
    def handler(request):
        bodies.append((request.url.path, json.loads(request.content)))
        if request.url.path.endswith('/patient-context/stream'):
            return sse(status='ready')
        return sse()
    run('http://localhost', 'holdout', output_root, label='fake', allow_holdout=True,
        transport=httpx.MockTransport(handler))
    assert all(set(body) == ({'patient_id', 'question'} if path.endswith('/clinical-chat/stream') else {'patient_id', 'emr_payload'})
               for path, body in bodies)
    with pytest.raises(ValueError, match='holdout'):
        run('http://localhost', 'holdout', output_root, label='fake')


def test_metrics_use_only_confirmed_judgments_and_nearest_rank():
    records = [{'task': 'summary', 'status': 'success', 'elapsed_ms': 10}]
    review = {'reviewer': 'reviewer', 'reviewed_at': '2026-11-19', 'claim_extraction_complete': True,
              'critical_review_complete': True, 'critical_errors': [],
              'facts': [{'importance': 'important', 'judgment': 'accurate'},
                        {'importance': 'important', 'judgment': 'omitted'},
                        {'importance': 'important', 'judgment': 'indeterminate'}],
              'claims': [{'judgment': 'unsupported', 'source_required': True, 'sources': []}]}
    result = report(records, [review])
    assert result['quality']['important_fact_recall'] == {'numerator': 1, 'denominator': 2, 'value': .5}
    assert result['quality']['unsupported_claims']['value'] == 1
    assert percentile([1, 2, 3, 4], .95) == 4
    assert percentile([], .95) is None


def test_docker_context_excludes_benchmark():
    assert 'benchmarks' in Path('.dockerignore').read_text().splitlines()
    assert 'benchmarks/output/' in Path('.gitignore').read_text().splitlines()


def test_citation_numbers_cover_first_and_repeat_forms():
    answer = '목표는 7% 미만입니다 [근거 1: 대한당뇨병학회 지침]. 재확인 [근거 1], [근거2] [출처: 심평원 DUR 목록]'
    assert cited_numbers(answer) == [1, 1, 2]


def test_mapped_value_check_is_exact_not_substring():
    assert contains_value({'a': [{'value': 13.6}]}, 13.6)
    assert not contains_value({'a': [{'value': 13.6}]}, 3.6)
    assert not contains_value({'a': '고혈압 진단'}, '고혈압')
    assert not contains_value({'a': True}, 1)


def test_runs_stay_out_of_the_real_output_directory(output_root):
    before = set(Path('benchmarks/output').glob('*')) if Path('benchmarks/output').exists() else set()
    run('http://localhost', 'validation', output_root, label='fake',
        transport=httpx.MockTransport(lambda request: httpx.Response(503)))
    after = set(Path('benchmarks/output').glob('*')) if Path('benchmarks/output').exists() else set()
    assert before == after
    with pytest.raises(ValueError, match='output must be within'):
        run('http://localhost', 'validation', Path('benchmarks/output'), label='fake',
            transport=httpx.MockTransport(lambda request: httpx.Response(503)))


def test_committed_dataset_matches_the_generator():
    from benchmarks.generate import build
    cases, gold = build()
    assert (cases, gold) == load()


def test_recall_is_reported_by_importance_and_source():
    records = [{'task': 'chat', 'status': 'success', 'elapsed_ms': 5}]
    review = {'reviewer': 'r', 'reviewed_at': '2026-11-19', 'claim_extraction_complete': True, 'forbidden_checks': [],
              'facts': [{'importance': 'critical', 'judgment': 'omitted', 'source': {'kind': 'dur'}},
                        {'importance': 'important', 'judgment': 'accurate', 'source': {'kind': 'emr'}},
                        {'importance': 'minor', 'judgment': 'accurate', 'source': {'kind': 'emr'}}]}
    quality = report(records, [review])['quality']
    assert quality['critical_fact_recall'] == {'numerator': 0, 'denominator': 1, 'value': 0.0}
    assert quality['important_fact_recall'] == {'numerator': 1, 'denominator': 2, 'value': .5}
    assert quality['fact_recall_by_source']['emr']['value'] == 1.0
    assert quality['fact_recall_by_importance']['minor']['denominator'] == 1


def test_progress_events_cover_every_task_and_count_finished_cases(output_root, capsys):
    from benchmarks.runner import _duration, stderr_progress
    events = []
    def handler(request):
        if request.url.path.endswith('/patient-context/stream'):
            return sse(status='ready')
        return sse()
    run('http://localhost', 'validation', output_root, label='progress', repetitions=2, progress=events.append,
        transport=httpx.MockTransport(handler))
    assert len(events) == 25 * 2 * 3
    assert [e['task'] for e in events[:3]] == ['summary', 'prepare', 'chat']
    assert events[-1]['done_units'] == events[-1]['total_units'] == 50 and events[-1]['index'] == 25
    assert sum(e['last_task'] for e in events) == 50
    stderr_progress(events[-1])
    line = capsys.readouterr().err
    assert '[ 25/25]' in line and '남은 예상' in line
    assert _duration(75) == '1m15s' and _duration(3725) == '1h02m'


def test_dur_facts_outside_a_single_drug_question_are_minor():
    from benchmarks.generate import _question_drugs
    cases, gold = load()
    by_id = {g['case_id']: g for g in gold}
    scoped = [c for c in cases if _question_drugs(c['question'])]
    assert scoped, 'expected questions limited to one drug'
    for case in scoped:
        prefixes = _question_drugs(case['question'])
        for fact in by_id[case['case_id']]['facts']:
            if fact['source']['kind'] == 'dur':
                involved = any(i.lower().startswith(p) for i in fact['source']['alert']['ingredients'] for p in prefixes)
                assert involved or fact['importance'] == 'minor', (case['case_id'], fact['id'])
    assert _question_drugs('현재 처방 약끼리 병용금기가 있나요? 심바스타틴 포함') == ()
