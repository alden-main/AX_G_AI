"""설정에 따라 AI Provider Client를 고르는 조립 지점."""

from __future__ import annotations

import httpx
from openai import AsyncOpenAI

from ax_g_ai.adapters.openai import OpenAICompatibleChatClient, OpenAIResponsesClient, ProviderClient
from ax_g_ai.config import AI_PROVIDER_OPENAI, AISettings

# An unauthenticated vLLM ignores the bearer token.  A placeholder is passed
# because the SDK would otherwise read OPENAI_API_KEY and send the OpenAI key
# to a self-hosted server.
_NO_API_KEY = "not-configured"


def build_provider_client(settings: AISettings, http_client: httpx.AsyncClient | None = None) -> ProviderClient:
    """설정한 Provider 종류에 맞는 ``respond``/``stream`` Client를 만든다.

    SDK Client 하나가 keep-alive 연결 풀을 공유하고, 첫 토큰 전의 연결 오류·429·5xx는
    ``max_retries``만큼 지수 백오프로 재시도한다.
    """
    client = AsyncOpenAI(
        api_key=settings.api_key or _NO_API_KEY,
        base_url=settings.base_url,
        timeout=settings.timeout_seconds,
        max_retries=settings.max_retries,
        http_client=http_client,
    )
    if settings.provider == AI_PROVIDER_OPENAI:
        return OpenAIResponsesClient(
            client,
            settings.model,
            reasoning_effort=settings.reasoning_effort,
            max_output_tokens=settings.max_output_tokens,
        )
    return OpenAICompatibleChatClient(
        client,
        settings.model,
        disable_thinking=settings.disable_thinking,
        max_output_tokens=settings.max_output_tokens,
    )
