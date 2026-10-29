from __future__ import annotations

import json
import unittest

from ax_g_ai.adapters.openai import HttpResponse, HttpStreamResponse, OpenAIProviderError, OpenAIRequest
from ax_g_ai.adapters.openai_compatible import OpenAICompatibleChatClient
from tests.adapters.test_openai import RecordingTransport, StreamingTransport


ENDPOINT = "http://vllm:8000/v1/chat/completions"


def sse(*events: object) -> tuple[bytes, ...]:
    lines: list[bytes] = []
    for event in events:
        data = event if isinstance(event, str) else json.dumps(event, ensure_ascii=False)
        lines.extend((f"data: {data}\n".encode("utf-8"), b"\n"))
    return tuple(lines)


class OpenAICompatibleChatClientTest(unittest.TestCase):
    def test_sends_instructions_as_system_message_and_extracts_content(self) -> None:
        transport = RecordingTransport(HttpResponse(
            200, "application/json", json.dumps({"choices": [{"message": {"content": " 요약 "}}]}).encode("utf-8"),
        ))
        client = OpenAICompatibleChatClient(transport, ENDPOINT, "qwen3-32b-awq", 5)

        answer = client.respond(OpenAIRequest("fixed instructions", "de-identified input"))

        self.assertEqual(answer.answer, "요약")
        url, headers, body, timeout = transport.calls[0]
        self.assertEqual(url, ENDPOINT)
        self.assertNotIn("Authorization", headers)
        self.assertEqual(timeout, 5)
        self.assertEqual(json.loads(body), {
            "model": "qwen3-32b-awq",
            "messages": [
                {"role": "system", "content": "fixed instructions"},
                {"role": "user", "content": "de-identified input"},
            ],
            "chat_template_kwargs": {"enable_thinking": False},
        })

    def test_sends_api_key_json_schema_and_token_limit_when_configured(self) -> None:
        transport = RecordingTransport(HttpResponse(
            200, "application/json", b'{"choices":[{"message":{"content":"{}"}}]}',
        ))
        client = OpenAICompatibleChatClient(
            transport, ENDPOINT, "model", 5, api_key="secret", disable_thinking=False, max_output_tokens=512,
        )

        client.respond(OpenAIRequest("instructions", "input", {"type": "object"}, "summary"))

        self.assertEqual(transport.calls[0][1]["Authorization"], "Bearer secret")
        sent = json.loads(transport.calls[0][2])
        self.assertNotIn("chat_template_kwargs", sent)
        self.assertEqual(sent["max_tokens"], 512)
        self.assertEqual(sent["response_format"], {
            "type": "json_schema",
            "json_schema": {"name": "summary", "strict": True, "schema": {"type": "object"}},
        })

    def test_rejects_failed_or_empty_responses(self) -> None:
        for response in (
            HttpResponse(500, "application/json", b'{"error":{}}'),
            HttpResponse(200, "text/html", b"<html>"),
            HttpResponse(200, "application/json", b'{"choices":[]}'),
            HttpResponse(200, "application/json", b'{"choices":[{"message":{"content":""}}]}'),
        ):
            client = OpenAICompatibleChatClient(RecordingTransport(response), ENDPOINT, "model", 5)
            with self.assertRaises(OpenAIProviderError):
                client.respond(OpenAIRequest("instructions", "input"))

    def test_streams_only_content_deltas_until_done(self) -> None:
        transport = StreamingTransport(HttpStreamResponse(200, "text/event-stream", iter(sse(
            {"choices": [{"delta": {"role": "assistant"}}]},
            {"choices": [{"delta": {"content": "정상 "}}]},
            {"choices": [{"delta": {"content": "응답"}, "finish_reason": "stop"}]},
            "[DONE]",
        ))))
        client = OpenAICompatibleChatClient(transport, ENDPOINT, "model", 5)

        self.assertEqual(list(client.stream(OpenAIRequest("instructions", "input"))), ["정상 ", "응답"])
        self.assertTrue(json.loads(transport.calls[0][2])["stream"])

    def test_stream_failures_are_classified(self) -> None:
        for lines in (
            sse({"choices": [{"delta": {"content": "부분"}}]}),
            sse({"error": {"message": "raw provider error"}}),
            sse({"choices": [{"delta": {"content": "잘린"}, "finish_reason": "length"}]}, "[DONE]"),
            sse("[DONE]"),
        ):
            client = OpenAICompatibleChatClient(
                StreamingTransport(HttpStreamResponse(200, "text/event-stream", iter(lines))), ENDPOINT, "model", 5,
            )
            with self.assertRaises(OpenAIProviderError):
                list(client.stream(OpenAIRequest("instructions", "input")))

    def test_non_success_stream_releases_the_connection(self) -> None:
        closed: list[bool] = []
        transport = StreamingTransport(HttpStreamResponse(
            503, "application/json", iter(()), lambda: closed.append(True),
        ))
        client = OpenAICompatibleChatClient(transport, ENDPOINT, "model", 5)

        with self.assertRaises(OpenAIProviderError):
            client.stream(OpenAIRequest("instructions", "input"))
        self.assertEqual(closed, [True])


if __name__ == "__main__":
    unittest.main()
