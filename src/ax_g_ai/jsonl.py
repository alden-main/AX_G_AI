"""JSON Lines reading shared by the corpus, the DUR snapshot and their loaders."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def parse_jsonl(text: str, *, strict: bool = True) -> list[dict[str, Any]]:
    """One object per non-blank line.

    ``strict`` (loaders, build checks): a malformed line raises ``json.JSONDecodeError`` and a
    non-object line ``ValueError``. Otherwise such lines are skipped, so one hand-edited line
    cannot stop the serving process from starting.
    """
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            if strict:
                raise
            continue
        if isinstance(row, dict):
            rows.append(row)
        elif strict:
            raise ValueError(f"line {number}: expected a JSON object")
    return rows


def read_jsonl(path: Path, *, strict: bool = True) -> list[dict[str, Any]]:
    """``parse_jsonl`` of a UTF-8 file; a missing file reads as empty."""
    return parse_jsonl(path.read_text(encoding="utf-8"), strict=strict) if path.is_file() else []
