"""출시 게이트: 벤치마크 실행 결과가 품질 하한선(``gates.json``)을 넘는지 자동 지표로 판정한다.

사람 판정(``review``·``report``)을 대신하지 않는다. 모델·프롬프트·설정을 바꿀 때마다 같은 기준으로 비교해, 기준에 못 미치는
변경을 배포 전에 걸러 내는 용도다. 답변 검사는 서비스의 문장 검증기(``ax_g_ai.services.answer_verification``)와 같은 코드다.

지표:
- ``<task>_success_rate``: 업무별 성공 비율.
- ``<task>_emr_value_mention``: 요약·채팅이 답해야 할 중요(critical·important) EMR 사실 값이 답변에 나온 비율(문자열 대조 근사).
- ``residual_violation_answers``: 검증기를 통과해 나간 답변에 남은 위반 건수. 측정 시점 단정·처방일 차용·처방 지시만 센다
  (날짜·수치는 실행 기록에 규칙 평가가 없어 DUR 고시일 등을 대조할 수 없으므로 세지 않는다).
- ``fallback_answer_rate``: 고정 안내 문장으로 답한 비율. ``omission_note_rate``: 문장을 빼고 생략 안내를 붙인 비율.
- ``recommendation_exact_cases``: 추천 DUR·임신 자동 대조 완전 일치율.
- ``<task>_first_text_p90_ms``: 첫 글자까지 걸린 시간의 90번째 백분위.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from ax_g_ai.adapters.main_bridge import RawEmrPayloadMapper
from ax_g_ai.services.answer_review import fallback_answer
from ax_g_ai.services.answer_verification import OMISSION_NOTE, SentenceSplitter, build_facts, check_sentence
from benchmarks.schema import DATA, read_jsonl

GATES = Path(__file__).with_name('gates.json')
RESIDUAL = {'untimed_glucose_timing', 'borrowed_prescription_date', 'prescription_directive'}
FALLBACKS = {fallback_answer(has_notice=False), fallback_answer(has_notice=True)}


def mentioned(fact, answer):
    """사실 값이 답변에 나왔는지 근사 판정한다. 사람 판정이 필요한 종류는 ``None``."""
    source = fact['source']
    value = source.get('value')
    if source['kind'] == 'calculation':
        return f"{source['value']:g}" in answer
    if source['kind'] != 'emr' or value is None:
        return None
    text = str(value)
    if source['path'].endswith('medName'):  # 제품명 또는 괄호 안 성분명의 앞 세 글자
        stems = {text.split('(')[0][:3], *(name[:3] for name in re.findall(r'\(([^)]+)\)', text))}
        return any(stem and stem in answer for stem in stems)
    if source['path'].endswith('kcdName'):
        code = re.search(r'\(([A-Z]\d{2,4})\)', fact['text'])
        return text[:5] in answer or bool(code and code.group(1) in answer)
    if isinstance(value, float):
        return f'{value:g}' in answer
    return text[:10] in answer if re.fullmatch(r'\d{4}-\d\d-\d\d.*', text) else text in answer


def measure(core_dir, recommendation_dir=None):
    """실행 폴더에서 게이트 지표를 계산한다."""
    gold = {g['case_id']: g for g in read_jsonl(DATA / 'gold.jsonl')}
    inputs = {c['case_id']: c for c in read_jsonl(DATA / 'inputs.jsonl')}
    rows = read_jsonl(Path(core_dir) / 'responses.jsonl')
    mapper = RawEmrPayloadMapper()
    metrics, mention, residual, answers, fallbacks, omissions = {}, {}, 0, 0, 0, 0
    for task in ('summary', 'prepare', 'chat'):
        subset = [r for r in rows if r['task'] == task]
        metrics[f'{task}_success_rate'] = sum(r['status'] == 'success' for r in subset) / len(subset) if subset else None
        firsts = sorted(r['first_text_ms'] for r in subset if r.get('first_text_ms') is not None)
        metrics[f'{task}_first_text_p90_ms'] = firsts[max(0, int(len(firsts) * 0.9) - 1)] if firsts else None
    for r in rows:
        if r['task'] not in ('summary', 'chat') or r['status'] != 'success':
            continue
        case, answer = inputs[r['case_id']], r['response']['answer']
        answers += 1
        fallbacks += answer.strip() in FALLBACKS
        omissions += OMISSION_NOTE in answer
        hit, total = mention.get(r['task'], (0, 0))
        for fact in gold[r['case_id']]['facts']:
            if r['task'] in fact['required_for'] and fact['importance'] != 'minor' and (found := mentioned(fact, answer)) is not None:
                hit, total = hit + found, total + 1
        mention[r['task']] = (hit, total)
        facts = build_facts(mapper.map(case['patient_id'], case['emr_payload']).data, case.get('question'), r['response'].get('evidence'))
        splitter = SentenceSplitter()
        body = answer.split('[처방 점검 —', 1)[0]
        residual += any(check_sentence(sentence, facts) in RESIDUAL for sentence in splitter.feed(body) + splitter.flush())
    for task, (hit, total) in mention.items():
        metrics[f'{task}_emr_value_mention'] = hit / total if total else None
    metrics['residual_violation_answers'] = residual
    metrics['fallback_answer_rate'] = fallbacks / answers if answers else None
    metrics['omission_note_rate'] = omissions / answers if answers else None
    if recommendation_dir is not None:
        report = json.loads((Path(recommendation_dir) / 'report.json').read_text(encoding='utf-8'))
        metrics['recommendation_exact_cases'] = (report.get('recommendation_checks') or {}).get('exact_cases', {}).get('value')
    return metrics


def judge(metrics, gates=GATES):
    """지표를 기준과 비교한다. 측정하지 않은 지표(``None``)는 판정에서 빼고 표시만 한다."""
    checks = []
    for gate in json.loads(Path(gates).read_text(encoding='utf-8'))['checks']:
        value = metrics.get(gate['name'])
        ok = None if value is None else (value >= gate['min'] if 'min' in gate else value <= gate['max'])
        checks.append({**gate, 'value': value, 'ok': ok})
    passed = all(check['ok'] is not False for check in checks if check['level'] == 'block')
    return {'passed': passed, 'checks': checks}
