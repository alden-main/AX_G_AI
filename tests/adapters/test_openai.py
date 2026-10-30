from __future__ import annotations

from collections.abc import Callable
import json
import os
import unittest
from unittest.mock import patch

import httpx

from ax_g_ai.adapters.openai import (
    OpenAICompatibleChatClient,
    OpenAIProviderError,
    OpenAIRequest,
    OpenAIResponsesClient,
    ProviderClient,
)
from ax_g_ai.adapters.provider import build_provider_client
from ax_g_ai.config import AISettings


Handler = Callable[[httpx.Request], httpx.Response]


def provider(handler: Handler, **environment: str) -> ProviderClient:
    environment = {"AI_MAX_RETRIES": "0", "AI_TIMEOUT_SECONDS": "5", **environment}
    with patch.dict(os.environ, environment, clear=True):
        settings = AISettings.from_environment()
    return build_provider_client(settings, httpx.AsyncClient(transport=httpx.MockTransport(handler)))


def openai_provider(handler: Handler, **environment: str) -> ProviderClient:
    return provider(handler, AI_API_KEY="secret", AI_MODEL="gpt-5-mini", **environment)


def compatible_provider(handler: Handler, **environment: str) -> ProviderClient:
    return provider(
        handler, AI_PROVIDER="openai_compatible", AI_BASE_URL="http://vllm:8000/v1", AI_MODEL="qwen3", **environment,
    )


class Recorder:
    """Record requests and answer them with fixed or sequential responses."""

    def __init__(self, *responses: httpx.Response) -> None:
        self.responses = list(responses)
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]

    def body(self, index: int = 0) -> dict[str, object]:
        return json.loads(self.requests[index].content)


def json_response(payload: object, status_code: int = 200) -> httpx.Response:
    return httpx.Response(status_code, json=payload)


def sse_response(*events: object) -> httpx.Response:
    body = "".join(
        f"data: {event if isinstance(event, str) else json.dumps(event, ensure_ascii=False)}\n\n" for event in events
    )
    return httpx.Response(200, headers={"Content-Type": "text/event-stream"}, content=body.encode("utf-8"))


def output_text(text: str) -> dict[str, object]:
    return {"output": [{"type": "message", "content": [{"type": "output_text", "text": text}]}]}


def chat_completion(content: str | None) -> dict[str, object]:
    return {"choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}]}


def chat_chunk(content: str | None = None, finish_reason: str | None = None) -> dict[str, object]:
    return {"choices": [{"index": 0, "delta": {"content": content}, "finish_reason": finish_reason}]}


REQUEST = OpenAIRequest("fixed instructions", "de-identified input")


async def collect(client: ProviderClient, request: OpenAIRequest = REQUEST) -> list[str]:
    return [delta async for delta in client.stream(request)]


class OpenAIResponsesClientTest(unittest.IsolatedAsyncioTestCase):
    async def test_sends_a_non_stored_responses_request_and_extracts_output_text(self) -> None:
        recorder = Recorder(json_response(output_text(" 요약 ")))
        client = openai_provider(recorder)

        answer = await client.respond(REQUEST)

        self.assertIsInstance(client, OpenAIResponsesClient)
        self.assertEqual(answer.answer, "요약")
        self.assertEqual(str(recorder.requests[0].url), "https://api.openai.com/v1/responses")
        self.assertEqual(recorder.requests[0].headers["Authorization"], "Bearer secret")
        self.assertEqual(recorder.body(), {
            "model": "gpt-5-mini",
            "instructions": "fixed instructions",
            "input": "de-identified input",
            "store": False,
        })

    async def test_sends_reasoning_output_limit_and_json_schema_when_configured(self) -> None:
        recorder = Recorder(json_response(output_text("{}")))
        client = openai_provider(recorder, AI_REASONING_EFFORT="low", AI_MAX_OUTPUT_TOKENS="800")

        await client.respond(OpenAIRequest("instructions", "input", {"type": "object"}))

        sent = recorder.body()
        self.assertEqual(sent["reasoning"], {"effort": "low"})
        self.assertEqual(sent["max_output_tokens"], 800)
        self.assertEqual(sent["text"], {"format": {
            "type": "json_schema", "name": "patient_context_summary", "strict": True, "schema": {"type": "object"},
        }})

    async def test_rejects_non_success_or_empty_responses_without_provider_details(self) -> None:
        for response in (
            json_response({"error": {"message": "raw provider detail"}}, 429),
            json_response({"output": []}),
        ):
            with self.subTest(status=response.status_code), self.assertRaises(OpenAIProviderError) as raised:
                await openai_provider(Recorder(response)).respond(REQUEST)
            self.assertNotIn("raw provider detail", str(raised.exception))

    async def test_retries_a_transient_failure_before_answering(self) -> None:
        recorder = Recorder(
            httpx.Response(503, headers={"retry-after-ms": "1"}, json={"error": {}}),
            json_response(output_text("재시도 성공")),
        )

        answer = await openai_provider(recorder, AI_MAX_RETRIES="1").respond(REQUEST)

        self.assertEqual(answer.answer, "재시도 성공")
        self.assertEqual(len(recorder.requests), 2)

    async def test_streams_only_output_text_deltas(self) -> None:
        recorder = Recorder(sse_response(
            {"type": "response.created", "response": {}},
            {"type": "response.output_text.delta", "delta": "정상 "},
            {"type": "response.output_text.delta", "delta": "응답"},
            {"type": "response.completed", "response": {}},
        ))

        self.assertEqual(await collect(openai_provider(recorder)), ["정상 ", "응답"])
        self.assertTrue(recorder.body()["stream"])

    async def test_stream_failures_are_classified(self) -> None:
        for events in (
            ({"type": "response.output_text.delta", "delta": "부분"},),
            ({"type": "response.output_text.delta", "delta": "부분"}, {"type": "response.failed", "response": {}}),
            ({"type": "response.completed", "response": {}},),
        ):
            with self.subTest(events=events), self.assertRaises(OpenAIProviderError):
                await collect(openai_provider(Recorder(sse_response(*events))))
        with self.assertRaises(OpenAIProviderError):
            await collect(openai_provider(Recorder(json_response({"error": {}}, 429))))


class OpenAICompatibleChatClientTest(unittest.IsolatedAsyncioTestCase):
    async def test_sends_instructions_as_system_message_and_extracts_content(self) -> None:
        recorder = Recorder(json_response(chat_completion(" 요약 ")))
        client = compatible_provider(recorder, OPENAI_API_KEY="openai-secret")

        answer = await client.respond(REQUEST)

        self.assertIsInstance(client, OpenAICompatibleChatClient)
        self.assertEqual(answer.answer, "요약")
        self.assertEqual(str(recorder.requests[0].url), "http://vllm:8000/v1/chat/completions")
        self.assertNotIn("openai-secret", recorder.requests[0].headers.get("Authorization", ""))
        self.assertEqual(recorder.body(), {
            "model": "qwen3",
            "messages": [
                {"role": "system", "content": "fixed instructions"},
                {"role": "user", "content": "de-identified input"},
            ],
            "chat_template_kwargs": {"enable_thinking": False},
        })

    async def test_sends_api_key_json_schema_and_token_limit_when_configured(self) -> None:
        recorder = Recorder(json_response(chat_completion("{}")))
        client = compatible_provider(
            recorder, AI_API_KEY="secret", AI_DISABLE_THINKING="false", AI_MAX_OUTPUT_TOKENS="512",
        )

        await client.respond(OpenAIRequest("instructions", "input", {"type": "object"}, "summary"))

        self.assertEqual(recorder.requests[0].headers["Authorization"], "Bearer secret")
        sent = recorder.body()
        self.assertNotIn("chat_template_kwargs", sent)
        self.assertEqual(sent["max_tokens"], 512)
        self.assertEqual(sent["response_format"], {
            "type": "json_schema",
            "json_schema": {"name": "summary", "strict": True, "schema": {"type": "object"}},
        })

    async def test_rejects_failed_or_empty_responses(self) -> None:
        for response in (
            json_response({"error": {}}, 500),
            json_response({"choices": []}),
            json_response(chat_completion("")),
        ):
            with self.subTest(response=response.content), self.assertRaises(OpenAIProviderError):
                await compatible_provider(Recorder(response)).respond(REQUEST)

    async def test_streams_only_content_deltas_until_stop(self) -> None:
        recorder = Recorder(sse_response(
            {"choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]},
            chat_chunk("정상 "),
            chat_chunk("응답", "stop"),
            "[DONE]",
        ))

        self.assertEqual(await collect(compatible_provider(recorder)), ["정상 ", "응답"])
        self.assertTrue(recorder.body()["stream"])

    async def test_stream_failures_are_classified(self) -> None:
        for events in (
            (chat_chunk("부분"),),
            (chat_chunk("부분"), {"error": {"message": "raw provider error"}}),
            (chat_chunk("잘린", "length"), "[DONE]"),
            ("[DONE]",),
        ):
            with self.subTest(events=events), self.assertRaises(OpenAIProviderError):
                await collect(compatible_provider(Recorder(sse_response(*events))))


if __name__ == "__main__":
    unittest.main()
