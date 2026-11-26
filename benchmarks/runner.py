"""API runner over the SSE endpoints (the only ones the API serves). This module never loads or transmits benchmark references."""
from __future__ import annotations

import hashlib
import json
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import httpx

from benchmarks.schema import DATA, read_jsonl
from benchmarks.streaming import stream_post

OUTPUT_ROOT = Path(__file__).parent / 'output'  # excluded from Docker and Git; tests point it at a temporary directory
SUITES = {'core': 'inputs.jsonl', 'recommendation': 'recommendation_inputs.jsonl'}
REQUEST_FIELDS = ('clinical_note', 'question', 'findings')  # recommendation request fields a case may set


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _valid(task, data):
    if not isinstance(data, dict):
        return False
    if task == 'prepare':  # the patient-context stream says "ready" once the chat context is stored
        return data.get('meta_status') == 'ready'
    answered = isinstance(data.get('answer'), str) and bool(data['answer'].strip())
    return answered and (task != 'recommendation' or isinstance(data.get('assessment'), dict))


def _duration(seconds):
    seconds = int(seconds)
    return f'{seconds // 3600}h{seconds % 3600 // 60:02d}m' if seconds >= 3600 else f'{seconds // 60}m{seconds % 60:02d}s'


def stderr_progress(event):
    """Default CLI progress line: one per task, with elapsed time and an estimate from the cases done so far."""
    import sys

    elapsed = event['elapsed_s']
    done, total = event['done_units'], event['total_units']
    eta = f", 남은 예상 {_duration(elapsed / done * (total - done))}" if done and event['last_task'] else ''
    ms = event.get('elapsed_ms')
    took = '-' if ms is None else f'{ms:.0f}ms' if ms < 1000 else f'{ms / 1000:.1f}s'
    print(f"[{event['index']:>3}/{event['total_cases']}] {event['case_id']} r{event['repetition']} {event['task']:<14} "
          f"{event['status']:<16} {took:>7} | 경과 {_duration(elapsed)}{eta}", file=sys.stderr, flush=True)


def run(base_url, split, output, *, label, repetitions=1, timeout=120, allow_holdout=False, transport=None,
        suite='core', progress=None):
    """``progress``: optional callable receiving one event per finished task (the CLI prints them to stderr)."""
    if split not in {'development', 'validation', 'holdout'} or (split == 'holdout' and not allow_holdout):
        raise ValueError('holdout requires --allow-holdout')
    if repetitions < 1 or timeout <= 0 or not label or suite not in SUITES:
        raise ValueError('invalid run settings')
    input_file = DATA / SUITES[suite]
    # A strict allowlist, not the whole case object: reference answers and labels cannot reach the service.
    dataset = json.loads((DATA / 'manifest.json').read_text(encoding='utf-8'))
    if sha(input_file) != dataset['hashes'][input_file.name]:
        raise ValueError('input data does not match dataset manifest')
    cases = [c for c in read_jsonl(input_file) if c['split'] == split]
    if not cases or any(c.get('origin') != 'synthetic' for c in cases):
        raise ValueError('only synthetic cases are permitted')
    run_id = uuid4().hex
    output = Path(output).resolve()
    allowed = Path(OUTPUT_ROOT).resolve()
    if not output.is_relative_to(allowed):
        raise ValueError('output must be within benchmarks/output (excluded from Docker and Git)')
    out = output / run_id
    out.mkdir(parents=True, exist_ok=False)
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], stderr=subprocess.DEVNULL, text=True).strip()
        dirty = bool(subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = 'unknown', 'unknown'
    manifest = {'run_id': run_id, 'label': label, 'suite': suite, 'split': split, 'repetitions': repetitions,
                'start_utc': datetime.now(UTC).isoformat(), 'input_file': input_file.name, 'input_sha256': sha(input_file),
                'commit': commit, 'working_tree_dirty': dirty, 'model': 'unknown', 'provider': 'unknown',
                'prompt_version': 'unknown', 'corpus_version': 'unknown', 'dur_version': 'unknown',
                'expected_dur_snapshot': dataset.get('dur_snapshot'),
                'concurrency': 1, 'timeout_seconds': timeout, 'retries': 0, 'mode': 'sse',
                'token_metrics': 'unavailable', 'dataset_review_status': 'draft'}
    records = []
    with httpx.Client(base_url=base_url, timeout=timeout, transport=transport, trust_env=False) as client:
        def call(path, body, task):
            result = stream_post(client, f'{path}/stream', body)
            # A stream that completes must still carry what the task requires (ready status, assessment).
            if result['status'] == 'success' and not _valid(task, result['response']):
                result['status'] = 'invalid_response'
            return result

        started = time.monotonic()
        total_units = len(cases) * repetitions
        done_units = 0
        tasks = ('summary', 'prepare', 'chat') if suite == 'core' else ('prepare', 'recommendation')

        def record(item, position):
            records.append(item)
            if progress is not None:
                last = item['task'] == tasks[-1]
                progress({'index': position, 'total_cases': len(cases), 'case_id': item['case_id'],
                          'repetition': item['repetition'], 'task': item['task'], 'status': item['status'],
                          'elapsed_ms': item.get('elapsed_ms'), 'elapsed_s': time.monotonic() - started,
                          'done_units': done_units + last, 'total_units': total_units, 'last_task': last})

        for position, c in enumerate(cases, 1):
            for rep in range(1, repetitions + 1):
                body = {'patient_id': c['patient_id'], 'emr_payload': c['emr_payload']}
                base = {'case_id': c['case_id'], 'split': split, 'category': c['category'], 'repetition': rep}
                if suite == 'core':
                    record({**base, 'task': 'summary', **call('/internal/v1/emr-summary', body, 'summary')}, position)
                # The streaming preparation (first summary text) is what the screen shows first.
                prep = call('/internal/v1/patient-context', body, 'prepare')
                record({**base, 'task': 'prepare', **prep}, position)
                if suite == 'core':
                    task, path = 'chat', '/internal/v1/clinical-chat'
                    request = {'patient_id': c['patient_id'], 'question': c['question']}
                else:
                    task, path = 'recommendation', '/internal/v1/clinical-recommendations'
                    request = {'patient_id': c['patient_id'], **{k: c['request'][k] for k in REQUEST_FIELDS if k in c['request']}}
                result = call(path, request, task) if prep['status'] == 'success' else {'status': 'prepare_error'}
                record({**base, 'task': task, **result}, position)
                done_units += 1
    manifest['end_utc'] = datetime.now(UTC).isoformat()
    (out / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    (out / 'responses.jsonl').write_text(''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in records), encoding='utf-8')
    return out
