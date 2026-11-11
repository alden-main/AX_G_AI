from __future__ import annotations

import os
import unittest
from unittest.mock import patch

import contextlib
import io
from pathlib import Path

from ax_g_ai.config import AISettings, ConfigurationError, LoaderSettings, RuntimeSettings


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


class RuntimeSettingsTest(unittest.TestCase):
    def test_defaults_disable_reference_data_and_bound_the_context_store(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            result = RuntimeSettings.from_environment()

        self.assertIsNone(result.reference_data_db_path)
        self.assertEqual(result.patient_context_ttl_seconds, 12 * 60 * 60)
        self.assertEqual(result.patient_context_max_entries, 1000)

    def test_rejects_a_missing_reference_database_or_invalid_store_limits(self) -> None:
        for environment in (
            {"AX_G_REFERENCE_DATA_DB_PATH": "/nonexistent/reference.sqlite"},
            {"AX_G_PATIENT_CONTEXT_TTL_SECONDS": "0"},
            {"AX_G_PATIENT_CONTEXT_MAX_ENTRIES": "many"},
        ):
            with self.subTest(environment=environment), patch.dict(os.environ, environment, clear=True):
                with self.assertRaises(ConfigurationError):
                    RuntimeSettings.from_environment()

    def test_configuration_errors_do_not_echo_secret_values(self) -> None:
        with self.assertRaises(ConfigurationError) as raised:
            settings(AI_API_KEY="very-secret-key", AI_BASE_URL="http://proxy/v1")

        self.assertNotIn("very-secret-key", str(raised.exception))


class LoaderSettingsTest(unittest.TestCase):
    def test_defaults_and_shared_variable_names(self) -> None:
        with patch.dict(os.environ, {"AX_G_CORPUS_DATABASE_URL": "postgresql://u:secret@db/x"}, clear=True):
            result = LoaderSettings.from_environment()
        self.assertEqual(result.corpus_database_url.get_secret_value(), "postgresql://u:secret@db/x")
        self.assertEqual((result.embedding_url, result.embedding_model, result.embedding_timeout_seconds), (None, "kure-v1", 60))
        self.assertEqual((result.evidence_corpus_dir, result.dur_snapshot_dir), (Path("text_data/corpus"), Path("data/dur")))

    def test_errors_name_fields_without_values(self) -> None:
        environment = {"AX_G_CORPUS_DATABASE_URL": "postgresql://u:very-secret@db/x", "AX_G_EMBEDDING_URL": "ftp://embedding"}
        with patch.dict(os.environ, environment, clear=True), self.assertRaises(ConfigurationError) as raised:
            LoaderSettings.from_environment()
        self.assertIn("AX_G_EMBEDDING_URL", str(raised.exception))
        self.assertNotIn("very-secret", str(raised.exception))

    def test_cli_without_database_url_exits_2_before_connecting(self) -> None:
        from ax_g_ai.corpus_store.loader import main as load_corpus
        from ax_g_ai.dur_store.loader import main as load_dur

        for main in (load_corpus, load_dur):
            with self.subTest(main=main.__module__), patch.dict(os.environ, {}, clear=True), \
                    contextlib.redirect_stderr(io.StringIO()) as err:
                self.assertEqual(main([]), 2)
            self.assertIn("AX_G_CORPUS_DATABASE_URL", err.getvalue())


if __name__ == "__main__":
    unittest.main()
