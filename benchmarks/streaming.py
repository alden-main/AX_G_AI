"""Measure the public SSE protocol at the client, without assuming chunks are tokens."""
from __future__ import annotations

import json
import time

import httpx


def stream_post(client, path, body, *, clock=time.monotonic):
    start = clock()
    offsets = []
    parts = []
    meta = {}
    status = 'stream_incomplete'
    http_status = None
    # Token timing is unavailable in this API: delta.text has no token IDs or timestamps.
    def result():
        elapsed = (clock() - start) * 1000
        data = {'status': status, 'elapsed_ms': elapsed, 'first_text_ms': offsets[0] if offsets else None,
                'delta_offsets_ms': offsets, 'delta_intervals_ms': [b - a for a, b in zip(offsets, offsets[1:], strict=False)],
                'delta_count': len(offsets), 'ttft_ms': None, 'token_intervals_ms': None,
                'token_throughput_per_second': None}
        if http_status is not None:
            data['http_status'] = http_status
        if parts:
            data['response'] = {'answer': ''.join(parts), 'evidence': meta.get('evidence')}
            if 'assessment' in meta:  # recommendation stream: the rule results come first, in meta
                data['response']['assessment'] = meta['assessment']
            if 'status' in meta:  # patient-context stream: "ready" once the chat context is stored
                data['response']['meta_status'] = meta['status']
        return data

    try:
        with client.stream('POST', path, json=body) as response:
            http_status = response.status_code
            if not response.is_success:
                status = 'request_error'
                return result()
            if not response.headers.get('content-type', '').lower().startswith('text/event-stream'):
                status = 'invalid_response'
                return result()
            event = 'message'
            data_lines = []
            for line in response.iter_lines():
                if line == '':
                    if not data_lines:
                        event = 'message'
                        continue
                    try:
                        payload = json.loads('\n'.join(data_lines))
                    except (ValueError, TypeError):
                        status = 'invalid_response'
                        return result()
                    if not isinstance(payload, dict):
                        status = 'invalid_response'
                        return result()
                    if event == 'meta':
                        meta = payload
                    elif event == 'delta':
                        text = payload.get('text')
                        if not isinstance(text, str):
                            status = 'invalid_response'
                            return result()
                        if text:
                            offsets.append((clock() - start) * 1000)
                            parts.append(text)
                    elif event == 'error':
                        status = 'stream_error'
                        return result()
                    elif event == 'done':
                        status = 'success' if parts else 'invalid_response'
                        return result()
                    event, data_lines = 'message', []
                elif line.startswith(':'):
                    continue
                else:
                    field, sep, value = line.partition(':')
                    if sep and value.startswith(' '):
                        value = value[1:]
                    if field == 'event':
                        event = value
                    elif field == 'data':
                        data_lines.append(value)
    except httpx.TimeoutException:
        status = 'timeout'
    except (httpx.HTTPError, UnicodeError):
        status = 'request_error'
    return result()
