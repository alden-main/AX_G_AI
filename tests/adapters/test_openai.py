from __future__ import annotations

import json
import unittest

from ax_g_ai.adapters.openai import (
    HttpResponse,
    OpenAIProviderError,
    OpenAIRequest,
    OpenAIResponsesClient,
)


class RecordingTransport:
    def __init__(self, response: HttpResponse) -> None:
        self.response = response
        self.calls: list[tuple[str, dict[str, str], bytes, float]] = []

    def post(self, url: str, headers: dict[str, str], body: bytes, timeout_seconds: float) -> HttpResponse:
        self.calls.append((url, headers, body, timeout_seconds))
        return self.response


class OpenAIResponsesClientTest(unittest.TestCase):
    def test_sends_a_non_stored_responses_request_and_extracts_output_text(self) -> None:
        transport = RecordingTransport(HttpResponse(
            200,
            "application/json",
            b'{"output":[{"type":"message","content":[{"type":"output_text","text":"\xec\x9a\x94\xec\x95\xbd"}]}]}',
        ))
        client = OpenAIResponsesClient(transport, "secret", "gpt-5-mini", 5)

        answer = client.respond(OpenAIRequest("fixed instructions", "de-identified input"))

        self.assertEqual(answer.answer, "요약")
        url, headers, body, timeout = transport.calls[0]
        self.assertEqual(url, "https://api.openai.com/v1/responses")
        self.assertEqual(headers["Authorization"], "Bearer secret")
        self.assertEqual(timeout, 5)
        self.assertEqual(json.loads(body), {
            "model": "gpt-5-mini",
            "instructions": "fixed instructions",
            "input": "de-identified input",
            "store": False,
        })

    def test_rejects_non_text_or_non_success_responses(self) -> None:
        client = OpenAIResponsesClient(
            RecordingTransport(HttpResponse(429, "application/json", b'{"error":{}}')),
            "secret", "gpt-5-mini", 5,
        )
        with self.assertRaises(OpenAIProviderError):
            client.respond(OpenAIRequest("instructions", "input"))

    def test_sends_json_schema_when_a_structured_output_is_requested(self) -> None:
        transport = RecordingTransport(HttpResponse(
            200, "application/json",
            b'{"output":[{"type":"message","content":[{"type":"output_text","text":"{}"}]}]}',
        ))
        client = OpenAIResponsesClient(transport, "secret", "gpt-5-mini", 5)

        client.respond(OpenAIRequest("instructions", "input", {"type": "object"}))

        sent = json.loads(transport.calls[0][2])
        self.assertEqual(sent["text"], {
            "format": {
                "type": "json_schema",
                "name": "patient_context_summary",
                "strict": True,
                "schema": {"type": "object"},
            }
        })

        client = OpenAIResponsesClient(
            RecordingTransport(HttpResponse(200, "application/json", b'{"output":[]}')),
            "secret", "gpt-5-mini", 5,
        )
        with self.assertRaises(OpenAIProviderError):
            client.respond(OpenAIRequest("instructions", "input"))


if __name__ == "__main__":
    unittest.main()
