"""설정에 따라 AI Provider Client를 고르는 조립 지점.

설계상 위치: 어댑터 계층의 조립 코드. ``runtime``이 앱 구성 요소를 만들 때 ``build_provider_client``를 호출한다.

역할: ``AISettings``의 Provider 종류에 따라 OpenAI Responses 모델 또는 OpenAI 호환 Chat Completions 모델(vLLM 등)을
설정하고, ``AsyncOpenAI`` SDK Client 위에 ``PydanticAIProviderClient``를 만든다.
"""

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
from pydantic_ai.settings import ModelSettings

from ax_g_ai.adapters.openai import ProviderClient, PydanticAIProviderClient
from ax_g_ai.config import AI_PROVIDER_OPENAI, AISettings

# 인증 없는 vLLM은 bearer 토큰을 무시한다. 그래도 자리표시 값을 넘기는 이유는, 그러지 않으면 SDK가
# ``OPENAI_API_KEY``를 읽어 OpenAI 키를 자체 호스팅 서버로 보내기 때문이다.
_NO_API_KEY = "not-configured"


def build_provider_client(
    settings: AISettings,
    http_client: httpx.AsyncClient | None = None,
    *,
    instrumentation: InstrumentationSettings | None = None,
) -> ProviderClient:
    """설정한 Provider 종류에 맞는 ``respond``/``stream`` Client를 만든다.

    역할: SDK Client 하나가 keep-alive 연결 풀을 공유하고, 첫 토큰 전의 연결 오류·429·5xx는
    ``max_retries``만큼 지수 백오프로 재시도한다. ``instrumentation``이 있으면 호출마다
    OpenTelemetry GenAI span을 남긴다. API Key가 없으면 자리표시 값(``_NO_API_KEY``)을 넘긴다.
    입력:
        settings: Provider 종류·모델·주소·API Key·시간 제한·재시도 횟수·출력 토큰 한도 등 AI 설정.
        http_client: SDK에 넘길 ``httpx.AsyncClient``. ``None``이면 SDK 기본 Client를 쓴다.
        instrumentation: Pydantic AI 계측 설정. ``None``이면 추적하지 않는다.
    출력: ``PydanticAIProviderClient``(``ProviderClient``).
    """
    client = AsyncOpenAI(
        api_key=settings.api_key or _NO_API_KEY,
        base_url=settings.base_url,
        timeout=settings.timeout_seconds,
        max_retries=settings.max_retries,
        http_client=http_client,
    )
    model = _model(settings, OpenAIProvider(openai_client=client))
    return PydanticAIProviderClient(
        model, instrumentation=instrumentation, system_prompt_prefix=settings.system_prompt_prefix,
        operation_settings=_no_thinking_settings(settings, model),
    )


def _no_thinking_settings(settings: AISettings, model: Model) -> dict[str, ModelSettings]:
    """``no_thinking_operations``의 작업마다 사고를 끄는 요청 설정(다른 ``extra_body`` 값은 그대로 둔다).

    요청 설정은 모델 설정 위에 키 단위로 덮어쓰므로 ``extra_body`` 전체(반복 억제 등 포함)를 다시 담는다.
    """
    if settings.provider == AI_PROVIDER_OPENAI or settings.disable_thinking or not settings.no_thinking_operation_names:
        return {}
    base = dict((model.settings or {}).get("extra_body") or {})  # type: ignore[call-overload]
    no_thinking = ModelSettings(extra_body={**base, "chat_template_kwargs": {"enable_thinking": False}})
    return dict.fromkeys(settings.no_thinking_operation_names, no_thinking)


def _model(settings: AISettings, provider: OpenAIProvider) -> Model:
    """Provider 종류에 맞는 Pydantic AI 모델과 모델 설정을 만든다.

    역할: ``AI_PROVIDER_OPENAI``이면 저장하지 않는(``openai_store=False``) Responses 모델에 추론 강도·최대 출력 토큰을
    설정하고, 그 밖에는 Chat Completions 모델에 최대 출력 토큰·샘플링 온도(설정한 경우)와(``disable_thinking``이면) 사고 끄기
    템플릿 인자를 설정한다.
    입력:
        settings: AI 설정.
        provider: SDK Client를 감싼 ``OpenAIProvider``.
    출력: ``OpenAIResponsesModel`` 또는 ``OpenAIChatModel``.
    """
    if settings.provider == AI_PROVIDER_OPENAI:
        # store=False: 그러지 않으면 Responses API가 요청을 OpenAI 쪽에 보관한다.
        responses = OpenAIResponsesModelSettings(openai_store=False)
        if settings.reasoning_effort is not None:
            # 추론 토큰은 보이는 텍스트보다 먼저 나오므로 effort가 첫 토큰까지의 시간을 좌우한다.
            responses["openai_reasoning_effort"] = settings.reasoning_effort
        if settings.max_output_tokens is not None:
            responses["max_tokens"] = settings.max_output_tokens
        return OpenAIResponsesModel(settings.model, provider=provider, settings=responses)
    chat = OpenAIChatModelSettings()
    if settings.max_output_tokens is not None:
        chat["max_tokens"] = settings.max_output_tokens
    if settings.temperature is not None:
        chat["temperature"] = settings.temperature
    extra_body: dict[str, object] = {}
    if settings.disable_thinking:
        # Qwen3 계열 Chat 템플릿은 그렇게 지시하면 사고 단계를 건너뛴다.
        extra_body["chat_template_kwargs"] = {"enable_thinking": False}
    if settings.repetition_penalty is not None:
        # OpenAI 표준 파라미터가 아니므로 extra_body로 보낸다(SGLang·vLLM이 받는다). 그러지 않으면 조용히 빠진다.
        extra_body["repetition_penalty"] = settings.repetition_penalty
    if extra_body:
        chat["extra_body"] = extra_body
    return OpenAIChatModel(settings.model, provider=provider, settings=chat)
