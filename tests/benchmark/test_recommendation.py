from __future__ import annotations

import copy
import json

import httpx
import pytest

from benchmarks.derive import dur_version
from benchmarks.recommendation import score
from benchmarks.review import create
from benchmarks.runner import run
from benchmarks.schema import DATA, read_jsonl, validate_recommendations

SOURCE = {'id': dur_version()['source_id'], 'title': 'DUR'}


def expectations():
    return {g['case_id']: g['expected_medication_alerts'] for g in read_jsonl(DATA / 'recommendation_gold.jsonl')}


def server(transform=lambda case_id, assessment: assessment, *, stream=False):
    """A fake recommendation API that answers with the expected alerts in the service's own alert shape."""
    expected = expectations()
    patients = {}
    def handler(request):
        body = json.loads(request.content)
        if request.url.path.startswith('/internal/v1/patient-context'):
            patients[body['patient_id']] = body
            if stream:
                return httpx.Response(200, headers={'content-type': 'text/event-stream'},
                                      content=b'event: meta\ndata: {"status":"ready"}\n\nevent: delta\ndata: {"text":"s"}\n\nevent: done\ndata: {}\n\n')
            return httpx.Response(200, json={'status': 'ready'})
        want = expected[body['patient_id']]
        alerts = [{'type': a['type'], 'status': a['status'], 'detail': a['detail'],
                   'products': [{'ingredient_name': name, **({'proposed': True} if a['proposed'] and i == 0 else {})}
                                for i, name in enumerate(sorted(a['ingredients'], key=lambda n: n not in want['proposed']))],
                   **({'grade': a['grade']} if 'grade' in a else {})} for a in want['alerts']]
        assessment = {'items': [], 'findings_used': {} if want['findings_used_pregnancy'] is None else {'pregnancy': want['findings_used_pregnancy']},
                      'medication_alerts': {'alerts': alerts, 'unmapped': [{'med_code': c} for c in want['unmapped']], 'source': SOURCE},
                      **({'finding_conflicts': {k: {'clinical_note': True, 'question': False} for k in want['finding_conflicts']}}
                         if want['finding_conflicts'] else {})}
        assessment = transform(body['patient_id'], assessment)
        answer = '점검 결과 [출처: 심평원 DUR 목록]'
        if stream:
            meta = json.dumps({'assessment': assessment}, ensure_ascii=False).encode()
            return httpx.Response(200, headers={'content-type': 'text/event-stream'},
                                  content=b'event: meta\ndata: ' + meta + b'\n\nevent: delta\ndata: {"text":"' + answer.encode() + b'"}\n\nevent: done\ndata: {}\n\n')
        return httpx.Response(200, json={'answer': answer, 'assessment': assessment})
    return httpx.MockTransport(handler)


def test_recommendation_dataset_matches_service_checks():
    cases = read_jsonl(DATA / 'recommendation_inputs.jsonl')
    gold = read_jsonl(DATA / 'recommendation_gold.jsonl')
    assert validate_recommendations(cases, gold)['cases'] == 30
    tampered = copy.deepcopy(gold)
    tampered[0]['expected_medication_alerts']['alerts'] = []
    with pytest.raises(ValueError, match='differs from the service'):
        validate_recommendations(cases, tampered)


@pytest.mark.parametrize('mode', ['json', 'sse'])
def test_matching_server_scores_exact(output_root, mode):
    out = run('http://localhost', 'development', output_root, label='rec', suite='recommendation', mode=mode,
              transport=server(stream=mode == 'sse'))
    result = score(out)
    assert result['statuses'] == {'compared': 10}
    assert result['exact_cases'] == {'numerator': 10, 'denominator': 10, 'value': 1.0}
    assert result['alert_recall']['value'] == 1.0 and result['alert_precision']['value'] == 1.0
    with pytest.raises(ValueError, match='core suite'):
        create(out)


def test_missing_extra_and_unavailable_alerts_are_reported(output_root):
    expected = expectations()
    target = next(cid for cid, e in expected.items() if e['alerts'] and cid.endswith(('001', '004', '007', '010', '016', '022', '028')))
    def transform(case_id, assessment):
        if case_id == target:
            assessment['medication_alerts']['alerts'] = []
        elif case_id.endswith('019'):  # male patient: an alert the service must not give
            assessment['medication_alerts']['alerts'] = [{'type': '임부금기', 'status': 'possible', 'detail': 'x',
                                                          'products': [{'ingredient_name': 'atorvastatin'}]}]
        elif case_id.endswith('013'):
            del assessment['medication_alerts']
        elif case_id.endswith('025'):
            assessment['medication_alerts']['source'] = {'id': 'OTHER'}
        return assessment
    out = run('http://localhost', 'development', output_root, label='rec', suite='recommendation', transport=server(transform))
    result = score(out)
    assert result['statuses'] == {'compared': 8, 'dur_unavailable': 1, 'dur_dataset_differs': 1}
    assert result['alert_recall']['value'] < 1 and result['alert_precision']['value'] < 1
    assert result['exact_cases']['numerator'] == 6
    by_case = {c['case_id']: c for c in result['cases']}
    assert by_case[target]['missing_alerts'] and not by_case[target]['exact']
