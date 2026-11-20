from __future__ import annotations

import json
import os
import unittest
from collections.abc import Callable
from unittest.mock import patch

import httpx
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pydantic import BaseModel, ConfigDict
from pydantic_ai import InstrumentationSettings

from ax_g_ai.adapters.openai import OpenAIProviderError, OpenAIRequest, ProviderClient, ProviderOutputError
from ax_g_ai.adapters.provider import build_provider_client
from ax_g_ai.config import AISettings

Handler = Callable[[httpx.Request], httpx.Response]


def provider(handler: Handler, instrumentation: InstrumentationSettings | None = None, **environment: str) -> ProviderClient:
    environment = {"AI_MAX_RETRIES": "0", "AI_TIMEOUT_SECONDS": "5", **environment}
    with patch.dict(os.environ, environment, clear=True):
        settings = AISettings.from_environment()
    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return build_provider_client(settings, http_client, instrumentation=instrumentation)


def openai_provider(handler: Handler, **environment: str) -> ProviderClient:
    return provider(handler, AI_API_KEY="secret", AI_MODEL="gpt-5-mini", **environment)


def compatible_provider(handler: Handler, instrumentation: InstrumentationSettings | None = None, **environment: str) -> ProviderClient:
    return provider(
        handler, instrumentation,
        AI_PROVIDER="openai_compatible", AI_BASE_URL="http://vllm:8000/v1", AI_MODEL="qwen3", **environment,
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


def response(status: str = "completed", output: list[object] | None = None) -> dict[str, object]:
    return {
        "id": "resp", "object": "response", "created_at": 0, "model": "gpt-5-mini", "status": status, "output": output or [],
        "parallel_tool_calls": False, "tool_choice": "auto", "tools": [],
    }


def output_text(text: str) -> dict[str, object]:
    return response(output=[{
        "type": "message", "id": "msg", "role": "assistant", "status": "completed",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }])


def response_event(kind: str, status: str) -> dict[str, object]:
    return {"type": f"response.{kind}", "sequence_number": 0, "response": response(status)}


def text_delta(text: str) -> dict[str, object]:
    return {
        "type": "response.output_text.delta", "delta": text, "item_id": "msg", "output_index": 0, "content_index": 0,
        "sequence_number": 1, "logprobs": [],
    }


CREATED = response_event("created", "in_progress")
COMPLETED = response_event("completed", "completed")


def chat_completion(content: str | None, finish_reason: str = "stop") -> dict[str, object]:
    return {
        "id": "chat", "object": "chat.completion", "created": 0, "model": "qwen3",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": finish_reason}],
    }


def chat_chunk(content: str | None = None, finish_reason: str | None = None) -> dict[str, object]:
    return {
        "id": "chat", "object": "chat.completion.chunk", "created": 0, "model": "qwen3",
        "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": finish_reason}],
    }


REQUEST = OpenAIRequest("fixed instructions", "de-identified input")


class Summary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message: str


SCHEMA = {
    "type": "object", "properties": {"message": {"type": "string"}}, "required": ["message"], "additionalProperties": False,
}


async def collect(client: ProviderClient, request: OpenAIRequest = REQUEST) -> list[str]:
    return [delta async for delta in client.stream(request)]


class OpenAIResponsesClientTest(unittest.IsolatedAsyncioTestCase):
    async def test_sends_a_non_stored_responses_request_and_extracts_output_text(self) -> None:
        recorder = Recorder(json_response(output_text(" 요약 ")))
        client = openai_provider(recorder)

        answer = await client.respond(REQUEST)

        self.assertEqual(answer.answer, "요약")
        self.assertEqual(str(recorder.requests[0].url), "https://api.openai.com/v1/responses")
        self.assertEqual(recorder.requests[0].headers["Authorization"], "Bearer secret")
        sent = recorder.body()
        self.assertEqual(sent["model"], "gpt-5-mini")
        self.assertEqual(sent["instructions"], "fixed instructions")
        self.assertEqual(sent["input"], [{"role": "user", "content": "de-identified input"}])
        self.assertIs(sent["store"], False)
        self.assertNotIn("reasoning", sent)
        self.assertNotIn("text", sent)

    async def test_sends_reasoning_output_limit_and_json_schema_when_configured(self) -> None:
        recorder = Recorder(json_response(output_text('{"message": "요약"}')))
        client = openai_provider(recorder, AI_REASONING_EFFORT="low", AI_MAX_OUTPUT_TOKENS="800")

        answer = await client.respond_structured(OpenAIRequest("instructions", "input"), Summary)

        self.assertEqual(answer, Summary(message="요약"))
        sent = recorder.body()
        self.assertEqual(sent["reasoning"], {"effort": "low"})
        self.assertEqual(sent["max_output_tokens"], 800)
        self.assertEqual(sent["text"], {"format": {"type": "json_schema", "name": "Summary", "strict": True, "schema": SCHEMA}})

    async def test_structured_answers_that_do_not_match_the_model_are_rejected(self) -> None:
        for text in ("자유 형식", '{"message": 1}', '{"other": "x"}'):
            with self.subTest(text=text), self.assertRaises(ProviderOutputError):
                await openai_provider(Recorder(json_response(output_text(text)))).respond_structured(REQUEST, Summary)

    async def test_rejects_non_success_or_empty_responses_without_provider_details(self) -> None:
        for reply in (
            json_response({"error": {"message": "raw provider detail"}}, 429),
            json_response(response()),
            json_response(response("incomplete", output_text("잘린")["output"])),  # type: ignore[arg-type]
        ):
            with self.subTest(status=reply.status_code), self.assertRaises(OpenAIProviderError) as raised:
                await openai_provider(Recorder(reply)).respond(REQUEST)
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
        recorder = Recorder(sse_response(CREATED, text_delta("정상 "), text_delta("응답"), COMPLETED))

        self.assertEqual(await collect(openai_provider(recorder)), ["정상 ", "응답"])
        self.assertTrue(recorder.body()["stream"])

    async def test_stream_failures_are_classified(self) -> None:
        for events in (
            (CREATED, text_delta("부분")),
            (CREATED, text_delta("부분"), response_event("failed", "failed")),
            (CREATED, text_delta("잘린"), response_event("incomplete", "incomplete")),
            (CREATED, COMPLETED),
        ):
            recorder = Recorder(sse_response(*events))
            with self.subTest(events=events), self.assertRaises(OpenAIProviderError):
                await collect(openai_provider(recorder))
            self.assertEqual(len(recorder.requests), 1)  # no continuation or output retry
        with self.assertRaises(OpenAIProviderError):
            await collect(openai_provider(Recorder(json_response({"error": {}}, 429))))


class OpenAICompatibleChatClientTest(unittest.IsolatedAsyncioTestCase):
    async def test_sends_instructions_as_system_message_and_extracts_content(self) -> None:
        recorder = Recorder(json_response(chat_completion(" 요약 ")))
        client = compatible_provider(recorder, OPENAI_API_KEY="openai-secret")

        answer = await client.respond(REQUEST)

        self.assertEqual(answer.answer, "요약")
        self.assertEqual(str(recorder.requests[0].url), "http://vllm:8000/v1/chat/completions")
        self.assertNotIn("openai-secret", recorder.requests[0].headers.get("Authorization", ""))
        self.assertEqual(recorder.body(), {
            "model": "qwen3",
            "messages": [
                {"role": "system", "content": "fixed instructions"},
                {"role": "user", "content": "de-identified input"},
            ],
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": False},
        })

    async def test_sends_api_key_json_schema_and_token_limit_when_configured(self) -> None:
        recorder = Recorder(json_response(chat_completion('{"message": "요약"}')))
        client = compatible_provider(
            recorder, AI_API_KEY="secret", AI_DISABLE_THINKING="false", AI_MAX_OUTPUT_TOKENS="512",
        )

        await client.respond_structured(OpenAIRequest("instructions", "input"), Summary)

        self.assertEqual(recorder.requests[0].headers["Authorization"], "Bearer secret")
        sent = recorder.body()
        self.assertNotIn("chat_template_kwargs", sent)
        self.assertEqual(sent["max_completion_tokens"], 512)
        self.assertEqual(sent["response_format"], {
            "type": "json_schema",
            "json_schema": {"name": "Summary", "strict": True, "schema": SCHEMA},
        })

    async def test_rejects_failed_or_empty_responses(self) -> None:
        for response in (
            json_response({"error": {}}, 500),
            json_response({"choices": []}),
            json_response(chat_completion("")),
            json_response(chat_completion("잘린", "length")),
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


class TracingTest(unittest.IsolatedAsyncioTestCase):
    async def trace(self, *, include_content: bool) -> list[ReadableSpan]:
        exporter = InMemorySpanExporter()
        tracer_provider = TracerProvider()
        tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))
        instrumentation = InstrumentationSettings(
            tracer_provider=tracer_provider, include_content=include_content, include_model_request_parameters=include_content,
        )
        client = compatible_provider(Recorder(json_response(chat_completion("model answer"))), instrumentation)

        await client.respond(OpenAIRequest("fixed instructions", "de-identified input", "clinical_recommendation", "abc123"))

        spans = list(exporter.get_finished_spans())
        self.assertTrue(any((span.attributes or {}).get("gen_ai.request.model") == "qwen3" for span in spans))
        return spans

    async def test_spans_carry_no_prompt_or_answer_unless_content_is_enabled(self) -> None:
        hidden = dump(await self.trace(include_content=False))
        shown = dump(await self.trace(include_content=True))

        for text in ("fixed instructions", "de-identified input", "model answer"):
            with self.subTest(text=text):
                self.assertNotIn(text, hidden)
                self.assertIn(text, shown)

    async def test_spans_are_named_after_the_operation_and_carry_the_prompt_version(self) -> None:
        run = next(span for span in await self.trace(include_content=False) if span.name.startswith("invoke_agent"))

        self.assertEqual(run.name, "invoke_agent clinical_recommendation")
        self.assertEqual(json.loads(str((run.attributes or {})["metadata"])),
                         {"operation": "clinical_recommendation", "prompt_version": "abc123"})


def dump(spans: list[ReadableSpan]) -> str:
    return json.dumps([
        {"attributes": dict(span.attributes or {}), "events": [dict(event.attributes or {}) for event in span.events]}
        for span in spans
    ], ensure_ascii=False, default=str)


if __name__ == "__main__":
    unittest.main()
