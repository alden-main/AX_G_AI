"""OpenAI Responses API의 서버 측 Adapter와 Provider 공통 경계.

환자 문맥은 호출자가 만든 최소 비식별 프롬프트만 ``input``으로 전달한다. API Key와
OpenAI의 원문 오류 응답은 이 모듈 밖으로 노출하지 않는다. Responses API의 기본 보관을
피하기 위해 모든 요청에 ``store: false``를 명시한다.

서비스 계층은 ``ProviderClient``의 ``respond``/``stream``만 사용한다. vLLM 등 OpenAI 호환
Chat Completions 서버는 ``openai_compatible`` 모듈의 Client로 교체한다.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from collections.abc import Callable, Iterator
from typing import Mapping, Protocol
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import httpx


class OpenAIProviderError(RuntimeError):
    """안전하게 분류된 OpenAI 호출 또는 응답 검증 실패다."""


@dataclass(frozen=True)
class OpenAIRequest:
    """Responses API에 전달할 고정 지시와 비식별 입력이다."""

    instructions: str
    input: str
    output_schema: Mapping[str, object] | None = None
    output_schema_name: str = "patient_context_summary"


@dataclass(frozen=True)
class ProviderAnswer:
    """검증된 모델 텍스트 응답이다."""

    answer: str


@dataclass(frozen=True)
class HttpResponse:
    status_code: int
    content_type: str | None
    body: bytes


@dataclass(frozen=True)
class HttpStreamResponse:
    """An HTTP response whose body is read incrementally.

    ``close`` releases the connection even when ``lines`` was never iterated,
    for example after a non-success status.
    """

    status_code: int
    content_type: str | None
    lines: Iterator[bytes]
    close: Callable[[], None] | None = None


class ProviderClient(Protocol):
    """서비스 계층이 의존하는 AI Provider 호출 경계."""

    def respond(self, request: OpenAIRequest) -> ProviderAnswer: ...

    def stream(self, request: OpenAIRequest) -> Iterator[str]: ...


class HttpTransport(Protocol):
    def post(self, url: str, headers: dict[str, str], body: bytes, timeout_seconds: float) -> HttpResponse: ...

    def post_stream(
        self, url: str, headers: dict[str, str], body: bytes, timeout_seconds: float
    ) -> HttpStreamResponse: ...


class UrllibHttpTransport:
    """표준 라이브러리로 OpenAI HTTPS API를 호출하는 transport다."""

    def post(self, url: str, headers: dict[str, str], body: bytes, timeout_seconds: float) -> HttpResponse:
        request = Request(url, data=body, headers=headers, method="POST")
        try:
            with urlopen(request, timeout=timeout_seconds) as response:
                return HttpResponse(response.status, response.headers.get("Content-Type"), response.read())
        except HTTPError as error:
            return HttpResponse(error.code, error.headers.get("Content-Type"), error.read())

    def post_stream(
        self, url: str, headers: dict[str, str], body: bytes, timeout_seconds: float
    ) -> HttpStreamResponse:
        request = Request(url, data=body, headers=headers, method="POST")
        try:
            response = urlopen(request, timeout=timeout_seconds)
        except HTTPError as error:
            response = error

        def read_lines() -> Iterator[bytes]:
            with response:
                while line := response.readline():
                    yield line

        return HttpStreamResponse(
            response.status, response.headers.get("Content-Type"), read_lines(), response.close
        )


class HttpxHttpTransport:
    """Reuse pooled keep-alive connections across Provider calls.

    ``urllib`` opens a new TCP/TLS connection for every request, which adds a
    handshake before the first token.  One ``httpx.Client`` is thread-safe and
    is shared by FastAPI's worker threads.
    """

    def __init__(self, max_connections: int = 64, transport: httpx.BaseTransport | None = None) -> None:
        self._client = httpx.Client(
            limits=httpx.Limits(max_connections=max_connections, max_keepalive_connections=max_connections),
            transport=transport,
        )

    def post(self, url: str, headers: dict[str, str], body: bytes, timeout_seconds: float) -> HttpResponse:
        response = self._client.post(url, headers=headers, content=body, timeout=timeout_seconds)
        return HttpResponse(response.status_code, response.headers.get("Content-Type"), response.content)

    def post_stream(
        self, url: str, headers: dict[str, str], body: bytes, timeout_seconds: float
    ) -> HttpStreamResponse:
        request = self._client.build_request("POST", url, headers=headers, content=body, timeout=timeout_seconds)
        response = self._client.send(request, stream=True)

        def read_lines() -> Iterator[bytes]:
            try:
                for line in response.iter_lines():
                    yield line.encode("utf-8")
            finally:
                response.close()

        return HttpStreamResponse(
            response.status_code, response.headers.get("Content-Type"), read_lines(), response.close
        )


def closing_lines(response: HttpStreamResponse, deltas: Iterator[str]) -> Iterator[str]:
    """Release the provider connection when the caller stops or fails early."""
    try:
        yield from deltas
    finally:
        close_response(response)


def close_response(response: HttpStreamResponse) -> None:
    if response.close is not None:
        try:
            response.close()
        except Exception:
            pass


class OpenAIResponsesClient:
    """Responses API 요청 직렬화와 안전한 텍스트 응답 추출을 담당한다."""

    def __init__(
        self,
        transport: HttpTransport,
        api_key: str,
        model: str,
        timeout_seconds: float,
        endpoint: str = "https://api.openai.com/v1/responses",
        *,
        reasoning_effort: str | None = None,
        max_output_tokens: int | None = None,
    ) -> None:
        if not api_key.strip():
            raise ValueError("OpenAI API key is required")
        if not model.strip():
            raise ValueError("OpenAI model is required")
        if not endpoint.startswith("https://"):
            raise ValueError("OpenAI endpoint must be an HTTPS URL")
        if timeout_seconds <= 0:
            raise ValueError("OpenAI timeout must be positive")
        self._transport = transport
        self._api_key = api_key
        self._model = model
        self._endpoint = endpoint
        self._timeout_seconds = timeout_seconds
        self._reasoning_effort = reasoning_effort
        self._max_output_tokens = max_output_tokens

    def respond(self, request: OpenAIRequest) -> ProviderAnswer:
        body = self._request_body(request)
        try:
            response = self._transport.post(
                self._endpoint,
                {"Content-Type": "application/json", "Authorization": f"Bearer {self._api_key}"},
                body,
                self._timeout_seconds,
            )
        except Exception as error:
            raise OpenAIProviderError("OpenAI transport failed") from error
        return self._parse_response(response)

    def stream(self, request: OpenAIRequest) -> Iterator[str]:
        """Yield text deltas from an OpenAI Responses API SSE response.

        Only ``response.output_text.delta`` is exposed to callers.  Provider
        event names and response IDs intentionally remain an adapter concern.
        """
        body = self._request_body(request, stream=True)
        try:
            response = self._transport.post_stream(
                self._endpoint,
                {"Content-Type": "application/json", "Authorization": f"Bearer {self._api_key}"},
                body,
                self._timeout_seconds,
            )
        except Exception as error:
            raise OpenAIProviderError("OpenAI transport failed") from error
        if not 200 <= response.status_code < 300:
            close_response(response)
            raise OpenAIProviderError(f"OpenAI returned a non-success status (status={response.status_code})")
        if not response.content_type or not response.content_type.lower().startswith("text/event-stream"):
            close_response(response)
            raise OpenAIProviderError("OpenAI returned an unexpected streaming content type")
        return closing_lines(response, self._parse_stream(response.lines))

    def _request_body(self, request: OpenAIRequest, *, stream: bool = False) -> bytes:
        if not request.instructions.strip() or not request.input.strip():
            raise OpenAIProviderError("OpenAI request requires instructions and input")
        payload: dict[str, object] = {
            "model": self._model,
            "instructions": request.instructions,
            "input": request.input,
            "store": False,
        }
        if stream:
            payload["stream"] = True
        if self._reasoning_effort is not None:
            # Reasoning tokens are generated before any visible text, so the
            # effort level directly controls time-to-first-token.
            payload["reasoning"] = {"effort": self._reasoning_effort}
        if self._max_output_tokens is not None:
            payload["max_output_tokens"] = self._max_output_tokens
        if request.output_schema is not None:
            payload["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": request.output_schema_name,
                    "strict": True,
                    "schema": request.output_schema,
                }
            }
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")

    @staticmethod
    def _parse_response(response: HttpResponse) -> ProviderAnswer:
        if not 200 <= response.status_code < 300:
            raise OpenAIProviderError(f"OpenAI returned a non-success status (status={response.status_code})")
        if not response.content_type or not response.content_type.lower().startswith("application/json"):
            raise OpenAIProviderError("OpenAI returned an unexpected content type")
        try:
            payload = json.loads(response.body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise OpenAIProviderError("OpenAI returned invalid JSON") from error
        output = payload.get("output") if isinstance(payload, dict) else None
        if not isinstance(output, list):
            raise OpenAIProviderError("OpenAI returned an invalid response shape")
        parts = [
            part.get("text")
            for item in output
            if isinstance(item, dict) and item.get("type") == "message"
            for content in [item.get("content")]
            if isinstance(content, list)
            for part in content
            if isinstance(part, dict) and part.get("type") == "output_text"
        ]
        answer = "".join(part for part in parts if isinstance(part, str)).strip()
        if not answer:
            raise OpenAIProviderError("OpenAI returned an empty or invalid answer")
        return ProviderAnswer(answer)

    @staticmethod
    def _parse_stream(lines: Iterator[bytes]) -> Iterator[str]:
        """Parse OpenAI SSE frames and yield only safe output-text deltas."""
        data_lines: list[str] = []
        completed = False
        yielded_text = False

        def event_data() -> dict[str, object] | None:
            if not data_lines:
                return None
            raw = "\n".join(data_lines)
            data_lines.clear()
            if raw == "[DONE]":
                return {"type": "response.completed"}
            try:
                value = json.loads(raw)
            except json.JSONDecodeError as error:
                raise OpenAIProviderError("OpenAI returned invalid streaming JSON") from error
            if not isinstance(value, dict):
                raise OpenAIProviderError("OpenAI returned an invalid streaming event")
            return value

        try:
            for raw_line in lines:
                try:
                    line = raw_line.decode("utf-8").rstrip("\r\n")
                except UnicodeDecodeError as error:
                    raise OpenAIProviderError("OpenAI returned invalid streaming bytes") from error
                if not line:
                    payload = event_data()
                    if payload is None:
                        continue
                    event_type = payload.get("type")
                    if event_type == "response.output_text.delta":
                        delta = payload.get("delta")
                        if not isinstance(delta, str):
                            raise OpenAIProviderError("OpenAI returned an invalid text delta")
                        if delta:
                            yielded_text = True
                            yield delta
                    elif event_type == "response.completed":
                        completed = True
                        break
                    elif event_type in {"error", "response.failed", "response.incomplete"}:
                        raise OpenAIProviderError("OpenAI streaming response failed")
                elif line.startswith("data:"):
                    data_lines.append(line[5:].lstrip())
            # A provider is allowed to omit the final blank line before close.
            if not completed and data_lines:
                payload = event_data()
                if payload and payload.get("type") == "response.completed":
                    completed = True
        except OpenAIProviderError:
            raise
        except Exception as error:
            raise OpenAIProviderError("OpenAI streaming transport failed") from error
        if not completed or not yielded_text:
            raise OpenAIProviderError("OpenAI returned an incomplete or empty streaming answer")
