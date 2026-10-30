"""배포 환경별 설정.

AI Provider 주소·API Key·모델과 런타임 경로는 코드에 넣지 않고 환경 변수로 주입한다.
설정값을 로그나 오류 메시지에 기록하지 않는다. 검증 실패는 필드 이름과 사유만 담은
``ConfigurationError``로 바뀐다.
"""

from __future__ import annotations

import os
from typing import Any, Literal

from pydantic import (
    FilePath,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    ValidationError,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict


class ConfigurationError(ValueError):
    """필수 환경 설정이 없거나 허용되지 않은 값일 때 발생한다."""


AI_PROVIDER_OPENAI = "openai"
AI_PROVIDER_OPENAI_COMPATIBLE = "openai_compatible"

_OPENAI_DEFAULTS = {"base_url": "https://api.openai.com/v1", "model": "gpt-5-mini"}
# Legacy ``OPENAI_*`` names are read when the matching ``AI_*`` is unset.  The
# credentials and endpoint belong to OpenAI only: they must not leak to a
# self-hosted server when an old .env still contains the OpenAI key.
_OPENAI_ONLY_LEGACY_FIELDS = ("base_url", "api_key", "model")
_SHARED_LEGACY_FIELDS = ("timeout_seconds", "reasoning_effort")


class _EnvironmentSettings(BaseSettings):
    model_config = SettingsConfigDict(env_ignore_empty=True, str_strip_whitespace=True, frozen=True)

    @classmethod
    def from_environment(cls):
        try:
            return cls()
        except ValidationError as error:
            prefix = cls.model_config.get("env_prefix", "")
            reasons = "; ".join(
                f"{prefix + '.'.join(map(str, item['loc'])).upper() if item['loc'] else cls.__name__}: {item['msg']}"
                for item in error.errors(include_input=False, include_url=False)
            )
            raise ConfigurationError(reasons) from error


class AISettings(_EnvironmentSettings):
    """AI Provider의 서버 전용 접속 설정.

    Attributes:
        provider: ``openai``는 OpenAI Responses API, ``openai_compatible``은 vLLM 등
            OpenAI 호환 Chat Completions API다.
        base_url: ``/responses`` 또는 ``/chat/completions`` 앞의 API 기준 주소다.
        api_key: Bearer 토큰이다. 인증 없는 내부 vLLM에서는 ``None``일 수 있다.
        reasoning_effort: 추론 모델의 사고량이다. 낮을수록 첫 글자가 빨라진다.
        disable_thinking: Qwen3 계열처럼 chat template에서 사고 모드를 끌 수 있는
            모델에 ``enable_thinking=false``를 전달할지 여부다.
        max_retries: 첫 토큰 전 연결 오류·429·5xx의 SDK 재시도 횟수다.
    """

    model_config = SettingsConfigDict(env_prefix="AI_")

    provider: Literal["openai", "openai_compatible"] = AI_PROVIDER_OPENAI
    base_url: str = ""
    api_key: str | None = None
    model: str = ""
    timeout_seconds: PositiveFloat = 30
    reasoning_effort: Literal["minimal", "low", "medium", "high"] | None = None
    disable_thinking: bool = True
    max_output_tokens: PositiveInt | None = None
    max_retries: NonNegativeInt = 2

    @model_validator(mode="before")
    @classmethod
    def _apply_legacy_and_provider_defaults(cls, data: Any) -> Any:
        values = dict(data)
        provider = str(values.get("provider") or AI_PROVIDER_OPENAI).strip().lower()
        values["provider"] = provider
        is_openai = provider == AI_PROVIDER_OPENAI
        legacy_fields = (*_OPENAI_ONLY_LEGACY_FIELDS, *_SHARED_LEGACY_FIELDS) if is_openai else _SHARED_LEGACY_FIELDS
        for name in legacy_fields:
            if not values.get(name) and (legacy := os.getenv(f"OPENAI_{name.upper()}", "").strip()):
                values[name] = legacy
        if is_openai:
            for name, default in _OPENAI_DEFAULTS.items():
                values.setdefault(name, default)
        if isinstance(values.get("reasoning_effort"), str):
            values["reasoning_effort"] = values["reasoning_effort"].lower()
        return values

    @field_validator("base_url")
    @classmethod
    def _http_url(cls, value: str) -> str:
        if not value.startswith(("https://", "http://")):
            raise ValueError("must be an HTTP(S) URL")
        return value.rstrip("/")

    @model_validator(mode="after")
    def _openai_requirements(self) -> "AISettings":
        if not self.model:
            raise ValueError("AI_MODEL is required")
        if self.provider == AI_PROVIDER_OPENAI:
            if not self.base_url.startswith("https://"):
                raise ValueError("AI_BASE_URL must use HTTPS for the openai provider")
            if self.api_key is None:
                raise ValueError("AI_API_KEY or OPENAI_API_KEY is required")
        return self


class RuntimeSettings(_EnvironmentSettings):
    """프로세스 런타임 설정.

    Attributes:
        reference_data_db_path: 승인된 참조 SQLite 경로다. 없으면 참조 보강을 끄고,
            설정했는데 파일이 없으면 앱 조립 시 fail-closed한다.
        patient_context_ttl_seconds: 준비된 환자 문맥을 메모리에 보관하는 시간이다.
        patient_context_max_entries: 메모리에 보관하는 환자 문맥의 최대 개수다.
    """

    model_config = SettingsConfigDict(env_prefix="AX_G_")

    reference_data_db_path: FilePath | None = None
    patient_context_ttl_seconds: PositiveFloat = 12 * 60 * 60
    patient_context_max_entries: PositiveInt = 1000
