"""vLLM 등 OpenAI 호환 Chat Completions API의 서버 측 Adapter.

자체 GPU 서버의 모델을 ``OpenAIResponsesClient``와 같은 ``respond``/``stream`` 경계로
노출한다. 지시문은 ``system`` 메시지, 비식별 입력은 ``user`` 메시지로 보낸다. 구조화
출력은 ``response_format``의 JSON Schema로 요청하며, vLLM은 이를 guided decoding으로
강제한다. Provider 원문 오류와 API Key는 이 모듈 밖으로 노출하지 않는다.
"""

from __future__ import annotations

from collections.abc import Iterator
import json

from ax_g_ai.adapters.openai import (
    HttpResponse,
    HttpTransport,
    OpenAIProviderError,
    OpenAIRequest,
    ProviderAnswer,
    close_response,
    closing_lines,
)


class OpenAICompatibleChatClient:
    """Chat Completions 요청 직렬화와 안전한 텍스트 응답 추출을 담당한다.

    Args:
        endpoint: ``http://vllm:8000/v1/chat/completions``처럼 전체 경로다. 내부망
            추론 서버를 위해 HTTP를 허용한다.
        api_key: vLLM ``--api-key``를 설정한 경우의 Bearer 토큰이다.
        disable_thinking: Qwen3 등 사고 모드가 있는 chat template에
            ``enable_thinking=false``를 전달해 사고 토큰 생성 대기를 없앤다.
    """

    def __init__(
        self,
        transport: HttpTransport,
        endpoint: str,
        model: str,
        timeout_seconds: float,
        *,
        api_key: str | None = None,
        disable_thinking: bool = True,
        max_output_tokens: int | None = None,
    ) -> None:
        if not endpoint.startswith(("https://", "http://")):
            raise ValueError("Chat Completions endpoint must be an HTTP(S) URL")
        if not model.strip():
            raise ValueError("Chat Completions model is required")
        if timeout_seconds <= 0:
            raise ValueError("Chat Completions timeout must be positive")
        self._transport = transport
        self._endpoint = endpoint
        self._model = model
        self._timeout_seconds = timeout_seconds
        self._api_key = api_key.strip() if api_key and api_key.strip() else None
        self._disable_thinking = disable_thinking
        self._max_output_tokens = max_output_tokens

    def respond(self, request: OpenAIRequest) -> ProviderAnswer:
        body = self._request_body(request)
        try:
            response = self._transport.post(self._endpoint, self._headers(), body, self._timeout_seconds)
        except Exception as error:
            raise OpenAIProviderError("Chat Completions transport failed") from error
        return self._parse_response(response)

    def stream(self, request: OpenAIRequest) -> Iterator[str]:
        """Yield ``choices[0].delta.content`` text from a Chat Completions SSE response."""
        body = self._request_body(request, stream=True)
        try:
            response = self._transport.post_stream(self._endpoint, self._headers(), body, self._timeout_seconds)
        except Exception as error:
            raise OpenAIProviderError("Chat Completions transport failed") from error
        if not 200 <= response.status_code < 300:
            close_response(response)
            raise OpenAIProviderError(
                f"Chat Completions returned a non-success status (status={response.status_code})"
            )
        if not response.content_type or not response.content_type.lower().startswith("text/event-stream"):
            close_response(response)
            raise OpenAIProviderError("Chat Completions returned an unexpected streaming content type")
        return closing_lines(response, self._parse_stream(response.lines))

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._api_key is not None:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    def _request_body(self, request: OpenAIRequest, *, stream: bool = False) -> bytes:
        if not request.instructions.strip() or not request.input.strip():
            raise OpenAIProviderError("Chat Completions request requires instructions and input")
        payload: dict[str, object] = {
            "model": self._model,
            # The system message stays byte-identical across calls so the
            # inference server can reuse its prefix cache.
            "messages": [
                {"role": "system", "content": request.instructions},
                {"role": "user", "content": request.input},
            ],
        }
        if stream:
            payload["stream"] = True
        if self._max_output_tokens is not None:
            payload["max_tokens"] = self._max_output_tokens
        if self._disable_thinking:
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        if request.output_schema is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": request.output_schema_name,
                    "strict": True,
                    "schema": request.output_schema,
                },
            }
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")

    @staticmethod
    def _parse_response(response: HttpResponse) -> ProviderAnswer:
        if not 200 <= response.status_code < 300:
            raise OpenAIProviderError(
                f"Chat Completions returned a non-success status (status={response.status_code})"
            )
        if not response.content_type or not response.content_type.lower().startswith("application/json"):
            raise OpenAIProviderError("Chat Completions returned an unexpected content type")
        try:
            payload = json.loads(response.body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise OpenAIProviderError("Chat Completions returned invalid JSON") from error
        choices = payload.get("choices") if isinstance(payload, dict) else None
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise OpenAIProviderError("Chat Completions returned an invalid response shape")
        message = choices[0].get("message")
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str) or not content.strip():
            raise OpenAIProviderError("Chat Completions returned an empty or invalid answer")
        return ProviderAnswer(content.strip())

    @staticmethod
    def _parse_stream(lines: Iterator[bytes]) -> Iterator[str]:
        """Parse Chat Completions SSE frames and yield only content deltas."""
        data_lines: list[str] = []
        completed = False
        yielded_text = False

        def frames() -> Iterator[str]:
            for raw_line in lines:
                try:
                    line = raw_line.decode("utf-8").rstrip("\r\n")
                except UnicodeDecodeError as error:
                    raise OpenAIProviderError("Chat Completions returned invalid streaming bytes") from error
                if line.startswith("data:"):
                    data_lines.append(line[5:].lstrip())
                elif not line and data_lines:
                    yield "\n".join(data_lines)
                    data_lines.clear()
            # A server is allowed to omit the final blank line before close.
            if data_lines:
                yield "\n".join(data_lines)
                data_lines.clear()

        try:
            for raw in frames():
                if raw == "[DONE]":
                    completed = True
                    break
                try:
                    event = json.loads(raw)
                except json.JSONDecodeError as error:
                    raise OpenAIProviderError("Chat Completions returned invalid streaming JSON") from error
                if not isinstance(event, dict) or "error" in event:
                    raise OpenAIProviderError("Chat Completions streaming response failed")
                choices = event.get("choices")
                if not isinstance(choices, list):
                    raise OpenAIProviderError("Chat Completions returned an invalid streaming event")
                for choice in choices:
                    delta = choice.get("delta") if isinstance(choice, dict) else None
                    content = delta.get("content") if isinstance(delta, dict) else None
                    if content is not None and not isinstance(content, str):
                        raise OpenAIProviderError("Chat Completions returned an invalid text delta")
                    if content:
                        yielded_text = True
                        yield content
                    if isinstance(choice, dict) and choice.get("finish_reason") == "length":
                        raise OpenAIProviderError("Chat Completions answer was truncated")
        except OpenAIProviderError:
            raise
        except Exception as error:
            raise OpenAIProviderError("Chat Completions streaming transport failed") from error
        if not completed or not yielded_text:
            raise OpenAIProviderError("Chat Completions returned an incomplete or empty streaming answer")
