from __future__ import annotations

import pytest

import benchmarks.runner


@pytest.fixture
def output_root(tmp_path, monkeypatch):
    """Runs write to a temporary directory, never to the real ``benchmarks/output``."""
    monkeypatch.setattr(benchmarks.runner, 'OUTPUT_ROOT', tmp_path)
    return tmp_path


def sse(answer='답변', **meta):
    """A fake SSE endpoint response: ``meta`` (the fields given), one ``delta`` with ``answer``, ``done``."""
    import json

    import httpx

    body = (f'event: meta\ndata: {json.dumps(meta, ensure_ascii=False)}\n\n'
            f'event: delta\ndata: {json.dumps({"text": answer}, ensure_ascii=False)}\n\nevent: done\ndata: {{}}\n\n')
    return httpx.Response(200, headers={'content-type': 'text/event-stream'}, content=body.encode())
