"""설정에 따라 AI Provider Client를 고르는 조립 지점."""

from __future__ import annotations

from ax_g_ai.adapters.openai import HttpTransport, HttpxHttpTransport, OpenAIResponsesClient, ProviderClient
from ax_g_ai.adapters.openai_compatible import OpenAICompatibleChatClient
from ax_g_ai.config import AI_PROVIDER_OPENAI, AISettings


def build_provider_client(settings: AISettings, transport: HttpTransport | None = None) -> ProviderClient:
    """설정한 Provider 종류에 맞는 ``respond``/``stream`` Client를 만든다."""
    transport = transport or HttpxHttpTransport()
    if settings.provider == AI_PROVIDER_OPENAI:
        return OpenAIResponsesClient(
            transport,
            settings.api_key or "",
            settings.model,
            settings.timeout_seconds,
            f"{settings.base_url}/responses",
            reasoning_effort=settings.reasoning_effort,
            max_output_tokens=settings.max_output_tokens,
        )
    return OpenAICompatibleChatClient(
        transport,
        f"{settings.base_url}/chat/completions",
        settings.model,
        settings.timeout_seconds,
        api_key=settings.api_key,
        disable_thinking=settings.disable_thinking,
        max_output_tokens=settings.max_output_tokens,
    )
