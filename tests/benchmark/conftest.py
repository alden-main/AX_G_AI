from __future__ import annotations

import pytest

import benchmarks.runner


@pytest.fixture
def output_root(tmp_path, monkeypatch):
    """Runs write to a temporary directory, never to the real ``benchmarks/output``."""
    monkeypatch.setattr(benchmarks.runner, 'OUTPUT_ROOT', tmp_path)
    return tmp_path
