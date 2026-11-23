"""말뭉치·DUR 스냅샷과 그 적재기가 함께 쓰는 JSON Lines 읽기 도구.

설계상 위치: 패키지 공용 유틸리티. 파일 말뭉치 읽기, DUR 스냅샷 생성·적재, 말뭉치 적재기 등이 쓴다.

역할: 한 줄에 JSON 객체 하나인 텍스트·파일을 dict 목록으로 읽는다. 엄격 모드에서는 잘못된 줄에서 멈추고,
관대 모드에서는 그 줄만 건너뛴다.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def parse_jsonl(text: str, *, strict: bool = True) -> list[dict[str, Any]]:
    """비어 있지 않은 줄마다 객체 하나를 읽는다.

    역할: 빈 줄은 건너뛴다. ``strict``(적재기·빌드 검사)이면 형식이 틀린 줄은 ``json.JSONDecodeError``,
    객체가 아닌 줄은 ``ValueError``를 낸다. 아니면 그런 줄을 건너뛰므로, 손으로 고친 줄 하나 때문에
    서비스 프로세스가 기동하지 못하는 일이 없다.
    입력:
        text: JSON Lines 텍스트.
        strict: 잘못된 줄에서 예외를 낼지 여부.
    출력: 줄 순서대로의 dict 목록.
    예외: ``strict``일 때 ``json.JSONDecodeError``(형식 오류), ``ValueError``(객체가 아닌 줄).
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
    """UTF-8 파일을 ``parse_jsonl``로 읽는다. 파일이 없으면 빈 목록이다."""
    return parse_jsonl(path.read_text(encoding="utf-8"), strict=strict) if path.is_file() else []
