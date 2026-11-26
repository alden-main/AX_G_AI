"""SSE 엔드포인트(``…/stream``) 호출을 예전 JSON 응답처럼 읽는 테스트 도우미.

API는 스트리밍만 제공한다. 성공(200)이면 ``meta`` 필드에 ``answer``(``delta`` 이어 붙임)를 더한 사전을, 실패면 JSON 오류 본문을
``json()``으로 돌려준다.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class StreamResult:
    status_code: int
    meta: dict[str, Any] = field(default_factory=dict)
    text: str = ""
    deltas: list[str] = field(default_factory=list)
    events: list[str] = field(default_factory=list)
    error: dict[str, Any] | None = None
    body: Any = None
    headers: Any = None

    def json(self) -> Any:
        if self.status_code != 200:
            return self.body
        return {**self.meta, "answer": self.text}


def parse_sse(raw: str) -> list[tuple[str, dict[str, Any]]]:
    """SSE 본문을 (이벤트 이름, data JSON) 목록으로 읽는다."""
    events = []
    for block in raw.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines() if ": " in line)
        if "event" in lines:
            events.append((lines["event"], json.loads(lines.get("data", "{}"))))
    return events


def post_stream(client: Any, path: str, **kwargs: Any) -> StreamResult:
    """``path``(``/stream`` 없이 써도 된다)의 스트리밍 엔드포인트를 호출한다."""
    if not path.endswith("/stream"):
        path += "/stream"
    response = client.post(path, **kwargs)
    if response.status_code != 200 or not response.headers.get("content-type", "").startswith("text/event-stream"):
        return StreamResult(response.status_code, body=response.json(), headers=response.headers)
    result = StreamResult(200, headers=response.headers)
    for name, data in parse_sse(response.text):
        result.events.append(name)
        if name == "meta":
            result.meta = data
        elif name == "delta":
            result.deltas.append(data["text"])
        elif name == "error":
            result.error = data
    result.text = "".join(result.deltas)
    return result


async def collect(deltas: Any) -> str:
    """서비스 스트림의 텍스트 조각을 이어 붙인다."""
    return "".join([delta async for delta in deltas])
