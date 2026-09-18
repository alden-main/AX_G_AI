from __future__ import annotations

import json
import unittest

from ax_g_ai.adapters.ai_tank import (
    AiTankClient,
    AiTankProviderError,
    ConsultRequest,
    HttpResponse,
)


class RecordingTransport:
    def __init__(self, response: HttpResponse) -> None:
        self.response = response
        self.calls: list[tuple[str, dict[str, str], bytes, float]] = []

    def post(self, url: str, headers: dict[str, str], body: bytes, timeout_seconds: float) -> HttpResponse:
        self.calls.append((url, headers, body, timeout_seconds))
        return self.response


class AiTankClientTest(unittest.TestCase):
    def test_sends_required_non_streaming_body_and_validates_answer(self) -> None:
        transport = RecordingTransport(HttpResponse(200, "application/json", b'{"answer":"ok"}'))
        client = AiTankClient(transport, "secret", "https://example.test/api/consult", 5)

        answer = client.consult(ConsultRequest((), "질문", "ko"))

        self.assertEqual(answer.answer, "ok")
        _, headers, body, _ = transport.calls[0]
        self.assertEqual(headers["x-api-key"], "secret")
        self.assertEqual(json.loads(body), {
            "history": [], "question": "질문", "language": "ko",
            "streaming": False, "summarization": False,
        })

    def test_rejects_a_non_json_provider_response(self) -> None:
        transport = RecordingTransport(HttpResponse(200, "text/plain", b"ok"))
        client = AiTankClient(transport, "secret", "https://example.test/api/consult", 5)

        with self.assertRaises(AiTankProviderError):
            client.consult(ConsultRequest((), "질문", "ko"))

    def test_accepts_the_provider_sse_answer_shape_when_non_streaming(self) -> None:
        transport = RecordingTransport(HttpResponse(
            201, None, b'data: {"content":"{\\"answer\\":\\"ok\\"}"}\n\ndata: [DONE]\n\n'
        ))
        client = AiTankClient(transport, "secret", "https://example.test/api/consult", 5)

        answer = client.consult(ConsultRequest((), "질문", "ko"))

        self.assertEqual(answer.answer, "ok")


if __name__ == "__main__":
    unittest.main()
