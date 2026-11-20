"""설정에 따라 AI Provider Client를 고르는 조립 지점."""

from __future__ import annotations

import httpx
from openai import AsyncOpenAI
from pydantic_ai import InstrumentationSettings
from pydantic_ai.models import Model
from pydantic_ai.models.openai import (
    OpenAIChatModel,
    OpenAIChatModelSettings,
    OpenAIResponsesModel,
    OpenAIResponsesModelSettings,
)
from pydantic_ai.providers.openai import OpenAIProvider

from ax_g_ai.adapters.openai import ProviderClient, PydanticAIProviderClient
from ax_g_ai.config import AI_PROVIDER_OPENAI, AISettings

# An unauthenticated vLLM ignores the bearer token.  A placeholder is passed
# because the SDK would otherwise read OPENAI_API_KEY and send the OpenAI key
# to a self-hosted server.
_NO_API_KEY = "not-configured"


def build_provider_client(
    settings: AISettings,
    http_client: httpx.AsyncClient | None = None,
    *,
    instrumentation: InstrumentationSettings | None = None,
) -> ProviderClient:
    """설정한 Provider 종류에 맞는 ``respond``/``stream`` Client를 만든다.

    SDK Client 하나가 keep-alive 연결 풀을 공유하고, 첫 토큰 전의 연결 오류·429·5xx는
    ``max_retries``만큼 지수 백오프로 재시도한다. ``instrumentation``이 있으면 호출마다
    OpenTelemetry GenAI span을 남긴다.
    """
    client = AsyncOpenAI(
        api_key=settings.api_key or _NO_API_KEY,
        base_url=settings.base_url,
        timeout=settings.timeout_seconds,
        max_retries=settings.max_retries,
        http_client=http_client,
    )
    return PydanticAIProviderClient(_model(settings, OpenAIProvider(openai_client=client)), instrumentation=instrumentation)


def _model(settings: AISettings, provider: OpenAIProvider) -> Model:
    if settings.provider == AI_PROVIDER_OPENAI:
        # store=False: the Responses API would otherwise keep the request on OpenAI's side.
        responses = OpenAIResponsesModelSettings(openai_store=False)
        if settings.reasoning_effort is not None:
            # Reasoning tokens come before any visible text, so effort sets time-to-first-token.
            responses["openai_reasoning_effort"] = settings.reasoning_effort
        if settings.max_output_tokens is not None:
            responses["max_tokens"] = settings.max_output_tokens
        return OpenAIResponsesModel(settings.model, provider=provider, settings=responses)
    chat = OpenAIChatModelSettings()
    if settings.max_output_tokens is not None:
        chat["max_tokens"] = settings.max_output_tokens
    if settings.disable_thinking:
        # Qwen3-style chat templates skip the thinking phase when told so.
        chat["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
    return OpenAIChatModel(settings.model, provider=provider, settings=chat)
