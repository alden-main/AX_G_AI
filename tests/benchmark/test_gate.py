from __future__ import annotations

import json

import httpx

from benchmarks import gate
from benchmarks.runner import run
from tests.benchmark.conftest import sse


def fake_run(output_root, answer):
    def handler(request):
        if request.url.path.endswith('/patient-context/stream'):
            return sse('요약', status='ready')
        return sse(answer)
    return run('http://localhost', 'validation', output_root, label='gate', transport=httpx.MockTransport(handler))


def test_an_empty_answer_run_fails_the_floor_and_a_residual_violation_is_counted(output_root):
    metrics = gate.measure(fake_run(output_root, '공복 혈당 확인이 필요합니다. 심바스타틴을 중단하세요.'))
    assert metrics['chat_success_rate'] == 1.0
    assert metrics['chat_emr_value_mention'] < 0.5  # the answer names no EMR fact
    assert metrics['residual_violation_answers'] == 50  # the directive, in every summary and chat answer
    result = gate.judge(metrics)
    assert not result['passed']
    failed = {check['name'] for check in result['checks'] if check['ok'] is False and check['level'] == 'block'}
    assert {'chat_emr_value_mention', 'residual_violation_answers'} <= failed
    unmeasured = next(check for check in result['checks'] if check['name'] == 'recommendation_exact_cases')
    assert unmeasured['ok'] is None  # no recommendation run given: shown, not judged


def test_thresholds_come_from_the_gates_file(tmp_path):
    gates = tmp_path / 'gates.json'
    gates.write_text(json.dumps({'checks': [{'name': 'a', 'min': 0.5, 'level': 'block'}, {'name': 'b', 'max': 1, 'level': 'warn'}]}))
    assert gate.judge({'a': 0.6, 'b': 2}, gates)['passed']  # a warning does not fail the gate
    assert not gate.judge({'a': 0.4, 'b': 0}, gates)['passed']
