from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from ax_g_ai.adapters.openai import OpenAIResponsesClient
from ax_g_ai.adapters.openai_compatible import OpenAICompatibleChatClient
from ax_g_ai.config import AISettings, ConfigurationError
from ax_g_ai.adapters.provider import build_provider_client
from tests.adapters.test_openai import RecordingTransport
from ax_g_ai.adapters.openai import HttpResponse


def settings(**environment: str) -> AISettings:
    with patch.dict(os.environ, environment, clear=True):
        return AISettings.from_environment()


class AISettingsTest(unittest.TestCase):
    def test_existing_openai_environment_keeps_working(self) -> None:
        result = settings(OPENAI_API_KEY="secret", OPENAI_MODEL="gpt-5-mini", OPENAI_TIMEOUT_SECONDS="12")

        self.assertEqual(result.provider, "openai")
        self.assertEqual(result.base_url, "https://api.openai.com/v1")
        self.assertEqual((result.api_key, result.model, result.timeout_seconds), ("secret", "gpt-5-mini", 12))
        self.assertIsNone(result.reasoning_effort)

    def test_openai_compatible_provider_allows_internal_http_without_key(self) -> None:
        result = settings(
            AI_PROVIDER="openai_compatible", AI_BASE_URL="http://vllm:8000/v1/", AI_MODEL="qwen3",
            AI_DISABLE_THINKING="false", AI_MAX_OUTPUT_TOKENS="1024",
        )

        self.assertEqual(result.base_url, "http://vllm:8000/v1")
        self.assertIsNone(result.api_key)
        self.assertFalse(result.disable_thinking)
        self.assertEqual(result.max_output_tokens, 1024)

    def test_openai_key_and_model_are_not_sent_to_a_self_hosted_server(self) -> None:
        result = settings(
            AI_PROVIDER="openai_compatible", AI_BASE_URL="http://vllm:8000/v1", AI_MODEL="qwen3",
            OPENAI_API_KEY="openai-secret", OPENAI_MODEL="gpt-5-mini",
        )

        self.assertIsNone(result.api_key)
        self.assertEqual(result.model, "qwen3")

    def test_rejects_unsafe_or_incomplete_settings(self) -> None:
        for environment in (
            {},
            {"AI_PROVIDER": "other", "AI_API_KEY": "k"},
            {"AI_API_KEY": "k", "AI_BASE_URL": "http://proxy/v1"},
            {"AI_PROVIDER": "openai_compatible", "AI_MODEL": "m"},
            {"AI_PROVIDER": "openai_compatible", "AI_BASE_URL": "http://vllm/v1"},
            {"AI_API_KEY": "k", "AI_REASONING_EFFORT": "extreme"},
            {"AI_API_KEY": "k", "AI_MAX_OUTPUT_TOKENS": "0"},
        ):
            with self.subTest(environment=environment), self.assertRaises(ConfigurationError):
                settings(**environment)


class ProviderClientFactoryTest(unittest.TestCase):
    def test_builds_the_configured_provider_endpoint(self) -> None:
        transport = RecordingTransport(HttpResponse(200, "application/json", b"{}"))

        openai = build_provider_client(settings(AI_API_KEY="k", AI_REASONING_EFFORT="low"), transport)
        compatible = build_provider_client(
            settings(AI_PROVIDER="openai_compatible", AI_BASE_URL="http://vllm:8000/v1", AI_MODEL="m"), transport,
        )

        self.assertIsInstance(openai, OpenAIResponsesClient)
        self.assertEqual(openai._endpoint, "https://api.openai.com/v1/responses")
        self.assertIsInstance(compatible, OpenAICompatibleChatClient)
        self.assertEqual(compatible._endpoint, "http://vllm:8000/v1/chat/completions")


if __name__ == "__main__":
    unittest.main()
