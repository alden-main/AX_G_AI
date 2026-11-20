"""Validate input and reference records without forwarding references to the API."""
from __future__ import annotations

import copy
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from ax_g_ai.adapters.main_bridge import RawEmrPayloadMapper
from benchmarks.criteria import IMPORTANCE

DATA = Path(__file__).parent / 'data'
SCHEMA_VERSION = 2
CATEGORIES = ('chronic', 'laboratory', 'medication', 'safety', 'uncertainty')
SOURCE_KINDS = frozenset({'emr', 'calculation', 'calculation_needed', 'dur', 'dur_unmapped', 'recorded_none', 'absent',
                          'answer_requirement'})


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def pointer(doc, path):
    if not isinstance(path, str) or not path.startswith('/'):
        raise ValueError('invalid JSON pointer')
    for part in path[1:].split('/'):
        part = part.replace('~1', '/').replace('~0', '~')
        doc = doc[int(part)] if isinstance(doc, list) else doc[part]
    return doc


def contains_value(doc, value) -> bool:
    """True when a leaf of the mapped context equals ``value`` (not a substring: 3.6 is not found in 13.6)."""
    if isinstance(doc, dict):
        return any(contains_value(v, value) for v in doc.values())
    if isinstance(doc, list):
        return any(contains_value(v, value) for v in doc)
    number = (int, float)
    if isinstance(doc, number) and isinstance(value, number):
        return not isinstance(doc, bool) and not isinstance(value, bool) and doc == value
    return type(doc) is type(value) and doc == value


def clinical_signature(case) -> str:
    """The case content without the synthetic identifiers: two cases with the same signature are duplicates."""
    emr = copy.deepcopy(case['emr_payload'])
    emr.pop('patient', None)
    return json.dumps([emr, case['question']], ensure_ascii=False, sort_keys=True)


def _check_derived(case, facts, context):
    """Derived facts must equal what the service code computes now (a stale reference fails here)."""
    from benchmarks.derive import calculations, chat_dur

    results, needed = calculations(case['emr_payload'])
    by_name = {r['name']: r for r in results}
    for fact in facts:
        source = fact['source']
        if source['kind'] == 'calculation':
            result = by_name.get(source['name'])
            if result is None or result['value'] != source['value'] or result.get('category') != source.get('category'):
                raise ValueError(f"calculation differs from service: {case['case_id']} {fact['id']}")
        elif source['kind'] == 'calculation_needed' and source['message'] not in needed:
            raise ValueError(f"calculation input message differs: {case['case_id']} {fact['id']}")
        elif source['kind'] == 'recorded_none' and source['label'] not in context.get('recorded_none', []):
            raise ValueError(f"empty list not recorded as none: {case['case_id']} {fact['id']}")
        elif source['kind'] == 'absent' and source['list'] in case['emr_payload']:
            raise ValueError(f"absent list is present: {case['case_id']} {fact['id']}")
    dur = chat_dur(case['emr_payload'], case['question'])
    expected_alerts = sorted(json.dumps(a, ensure_ascii=False, sort_keys=True) for a in (dur or {}).get('alerts', []))
    recorded = sorted(json.dumps(f['source']['alert'], ensure_ascii=False, sort_keys=True) for f in facts if f['source']['kind'] == 'dur')
    if expected_alerts != recorded:
        raise ValueError(f"DUR facts differ from the service check: {case['case_id']}")
    unmapped = sorted(f['source']['med_code'] for f in facts if f['source']['kind'] == 'dur_unmapped')
    if unmapped != sorted((dur or {}).get('unmapped', [])):
        raise ValueError(f"unmapped DUR facts differ: {case['case_id']}")


def validate(cases, gold):
    if len(cases) != 100 or len(gold) != 100:
        raise ValueError('expected exactly 100 inputs and references')
    ids = [c['case_id'] for c in cases]
    if len(set(ids)) != 100 or set(ids) != {g['case_id'] for g in gold} or len({g['case_id'] for g in gold}) != 100:
        raise ValueError('case IDs must be unique and match')
    if Counter(c['split'] for c in cases) != {'development': 50, 'validation': 25, 'holdout': 25}:
        raise ValueError('unexpected split sizes')
    for category in CATEGORIES:
        if Counter(c['split'] for c in cases if c['category'] == category) != {'development': 10, 'validation': 5, 'holdout': 5}:
            raise ValueError('unexpected category distribution')
    signatures = Counter(clinical_signature(c) for c in cases)
    if max(signatures.values()) > 1:
        raise ValueError('duplicate clinical content across cases')
    templates = {split: {c['question_template'] for c in cases if c['split'] == split} for split in ('development', 'validation', 'holdout')}
    if templates['holdout'] & (templates['development'] | templates['validation']) or templates['validation'] & templates['development']:
        raise ValueError('question templates are shared across splits')
    groups = {}
    references = {g['case_id']: g for g in gold}
    mapper = RawEmrPayloadMapper()
    for c in cases:
        if c['origin'] != 'synthetic' or c['schema_version'] != SCHEMA_VERSION:
            raise ValueError(f'only version {SCHEMA_VERSION} synthetic cases are allowed')
        if c['patient_group_id'] in groups and groups[c['patient_group_id']] != c['split']:
            raise ValueError('patient group crosses splits')
        groups[c['patient_group_id']] = c['split']
        if c['patient_id'] != c['emr_payload']['patient']['patientId'] or not c['question'].strip():
            raise ValueError('patient ID or question invalid')
        context = mapper.map(c['patient_id'], c['emr_payload'], prepared_at=datetime(2026, 11, 1, tzinfo=UTC)).data
        g = references[c['case_id']]
        if g['schema_version'] != SCHEMA_VERSION or not g['expected_summary'] or not g['expected_chat'] or not g['forbidden_claims']:
            raise ValueError('reference incomplete')
        if g['expected_summary'] == g['expected_chat']:
            raise ValueError(f"summary and chat references must differ: {c['case_id']}")
        facts = g['facts']
        if not facts or len({f['id'] for f in facts}) != len(facts):
            raise ValueError('invalid fact IDs')
        for task in ('summary', 'chat'):
            if not any(task in f['required_for'] for f in facts):
                raise ValueError(f"no {task} fact: {c['case_id']}")
        for fact in facts:
            if fact.get('importance') not in IMPORTANCE:
                raise ValueError(f"invalid importance: {c['case_id']} {fact['id']}")
            if not set(fact['required_for']) <= {'summary', 'chat'} or not fact['required_for']:
                raise ValueError('invalid task')
            source = fact.get('source') or {}
            if source.get('kind') not in SOURCE_KINDS:
                raise ValueError(f"invalid fact source: {c['case_id']} {fact['id']}")
            if source['kind'] == 'emr':
                if pointer(c['emr_payload'], source['path']) != source['value']:
                    raise ValueError(f"fact source mismatch: {c['case_id']} {fact['id']}")
                if not contains_value(context, source['value']):
                    raise ValueError(f"fact lost in mapping: {c['case_id']} {fact['id']}")
        _check_derived(c, facts, context)
    return {'cases': len(cases), 'tasks': len(cases) * 2, 'splits': dict(Counter(c['split'] for c in cases)),
            'facts': dict(Counter(f['importance'] for g in gold for f in g['facts'])),
            'fact_sources': dict(Counter(f['source']['kind'] for g in gold for f in g['facts']))}


def validate_recommendations(cases, gold):
    """The recommendation suite: 10 scenarios × 3 splits, expectations equal to the service's deterministic check."""
    from benchmarks.derive import recommendation_dur

    ids = [c['case_id'] for c in cases]
    if len(cases) != 30 or len(set(ids)) != 30 or set(ids) != {g['case_id'] for g in gold} or len(gold) != 30:
        raise ValueError('expected 30 recommendation cases with matching references')
    if Counter(c['split'] for c in cases) != {'development': 10, 'validation': 10, 'holdout': 10}:
        raise ValueError('unexpected recommendation split sizes')
    if max(Counter(json.dumps([clinical_signature({**c, 'question': ''}), c['request']], ensure_ascii=False, sort_keys=True)
                   for c in cases).values()) > 1:
        raise ValueError('duplicate recommendation cases')
    references = {g['case_id']: g for g in gold}
    for c in cases:
        if c['origin'] != 'synthetic' or c['schema_version'] != SCHEMA_VERSION or c['patient_id'] != c['emr_payload']['patient']['patientId']:
            raise ValueError(f"invalid recommendation case: {c['case_id']}")
        request = c['request']
        if not request or not set(request) <= {'clinical_note', 'question', 'findings'}:
            raise ValueError(f"invalid recommendation request: {c['case_id']}")
        expected = recommendation_dur(c['emr_payload'], clinical_note=request.get('clinical_note'), question=request.get('question'),
                                      findings=request.get('findings'))
        if expected != references[c['case_id']]['expected_medication_alerts']:
            raise ValueError(f"recommendation expectation differs from the service check: {c['case_id']}")
    return {'cases': len(cases), 'scenarios': dict(Counter(c['scenario'] for c in cases))}
