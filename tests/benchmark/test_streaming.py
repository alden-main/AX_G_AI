from __future__ import annotations

import json

import httpx
import pytest

from benchmarks.metrics import report
from benchmarks.runner import run
from benchmarks.schema import read_jsonl
from benchmarks.streaming import stream_post


class Clock:
    def __init__(self):
        self.value = 0.0

    def __call__(self):
        return self.value


class TimedStream(httpx.SyncByteStream):
    def __init__(self, chunks, clock, error=None):
        self.chunks, self.clock, self.error = chunks, clock, error

    def __iter__(self):
        for delay, chunk in self.chunks:
            self.clock.value += delay
            yield chunk
        if self.error:
            raise self.error


def client_for(chunks, clock, error=None, *, content_type='text/event-stream'):
    def handler(request):
        return httpx.Response(200, headers={'content-type': content_type}, stream=TimedStream(chunks, clock, error))
    return httpx.Client(base_url='http://localhost', transport=httpx.MockTransport(handler))


def test_stream_timings_skip_meta_and_empty_delta_and_join_split_frames():
    clock = Clock()
    chunks = [(0.1, b'event: meta\ndata: {"evidence": []}\n\n'),
              (0.1, b'event: delta\ndata: {"text":""}\n\n'),
              (0.1, b'event: delta\ndata: {"text":"A'),
              (0.2, b'"}\n\nevent: delta\ndata: {"text":"B"}\n\n'),
              (0.3, b'event: done\ndata: {}\n\n')]
    with client_for(chunks, clock) as client:
        result = stream_post(client, '/stream', {}, clock=clock)
    assert result['status'] == 'success'
    assert result['response']['answer'] == 'AB'
    assert result['first_text_ms'] == pytest.approx(500)
    assert result['delta_intervals_ms'] == pytest.approx([0])
    assert result['elapsed_ms'] == pytest.approx(800)
    assert result['ttft_ms'] is None
    assert result['token_intervals_ms'] is None


@pytest.mark.parametrize(('tail', 'expected'), [
    (b'event: error\ndata: {"code":"failed"}\n\n', 'stream_error'),
    (b'', 'stream_incomplete'),
    (b'event: delta\ndata: invalid\n\n', 'invalid_response'),
])
def test_incomplete_error_and_invalid_streams_preserve_partial(tail, expected):
    clock = Clock()
    chunks = [(0.1, b'event: delta\ndata: {"text":"partial"}\n\n'), (0.1, tail)]
    with client_for(chunks, clock) as client:
        result = stream_post(client, '/stream', {}, clock=clock)
    assert result['status'] == expected
    assert result['response']['answer'] == 'partial'


def test_timeout_preserves_partial_and_bad_content_type_rejected():
    clock = Clock()
    chunks = [(0.1, b'event: delta\ndata: {"text":"a"}\n\n')]
    with client_for(chunks, clock, httpx.ReadTimeout('timeout')) as client:
        assert stream_post(client, '/stream', {}, clock=clock)['status'] == 'timeout'
    with client_for(chunks, clock, content_type='application/json') as client:
        assert stream_post(client, '/stream', {}, clock=clock)['status'] == 'invalid_response'


def test_http_failure_and_done_without_text():
    with httpx.Client(base_url='http://localhost', transport=httpx.MockTransport(
        lambda request: httpx.Response(503, json={'code': 'unavailable'})
    )) as client:
        result = stream_post(client, '/stream', {})
    assert result['status'] == 'request_error'
    assert result['http_status'] == 503
    clock = Clock()
    with client_for([(0.1, b'event: done\ndata: {}\n\n')], clock) as client:
        assert stream_post(client, '/stream', {}, clock=clock)['status'] == 'invalid_response'


def test_sse_runner_and_report(output_root):
    requests = []
    def handler(request):
        requests.append((request.url.path, json.loads(request.content)))
        meta = b'{"status":"ready"}' if request.url.path.endswith('/patient-context/stream') else b'{}'
        return httpx.Response(200, headers={'content-type': 'text/event-stream'},
                              content=b'event: meta\ndata: ' + meta + b'\n\nevent: delta\ndata: {"text":"a"}\n\nevent: done\ndata: {}\n\n')
    out = run('http://localhost', 'validation', output_root, label='sse-test', mode='sse',
              transport=httpx.MockTransport(handler))
    records = read_jsonl(out / 'responses.jsonl')
    assert len(records) == 75
    assert all(r['status'] == 'success' for r in records)
    assert {path for path, _ in requests} == {'/internal/v1/emr-summary/stream', '/internal/v1/patient-context/stream',
                                              '/internal/v1/clinical-chat/stream'}
    assert all(set(body) == ({'patient_id', 'question'} if path.endswith('/clinical-chat/stream') else {'patient_id', 'emr_payload'})
               for path, body in requests)
    tasks = report(records, [])['tasks']
    assert tasks['prepare']['streaming']['first_text_ms']['n'] == 25
    metrics = tasks['chat']['streaming']
    assert metrics['first_text_ms']['n'] == 25
    assert metrics['delta_intervals_pooled_ms']['n'] == 0
    assert metrics['ttft_ms'] is None


def test_sse_preparation_without_ready_status_fails_the_chat(output_root):
    def handler(request):
        return httpx.Response(200, headers={'content-type': 'text/event-stream'},
                              content=b'event: meta\ndata: {}\n\nevent: delta\ndata: {"text":"a"}\n\nevent: done\ndata: {}\n\n')
    out = run('http://localhost', 'validation', output_root, label='sse-test', mode='sse', transport=httpx.MockTransport(handler))
    records = read_jsonl(out / 'responses.jsonl')
    assert {r['status'] for r in records if r['task'] == 'prepare'} == {'invalid_response'}
    assert {r['status'] for r in records if r['task'] == 'chat'} == {'prepare_error'}
