"""배포 환경별 외부 연동 설정.

EMR 주소·경로·HTTP 방식·시간 초과·서버 측 인증 토큰은 코드에 넣지 않고 환경 변수로
주입한다. 설정값을 로그나 감사 이벤트에 기록하지 않으며, 실제 비밀값은 배포 환경의
비밀 저장소에서 ``AX_G_BRIDGE_ACCESS_TOKEN``으로 제공해야 한다.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


class ConfigurationError(ValueError):
    """필수 환경 설정이 없거나 허용되지 않은 값일 때 발생한다."""


def reference_data_database_path() -> Path | None:
    """Return the optional approved reference-data SQLite path.

    A missing setting leaves reference enrichment disabled; a configured but
    missing file fails closed during application assembly.
    """
    configured = os.getenv("AX_G_REFERENCE_DATA_DB_PATH", "").strip()
    if not configured:
        return None
    path = Path(configured)
    if not path.is_file():
        raise ConfigurationError("AX_G_REFERENCE_DATA_DB_PATH must reference an existing SQLite file")
    return path


AI_PROVIDER_OPENAI = "openai"
AI_PROVIDER_OPENAI_COMPATIBLE = "openai_compatible"
_REASONING_EFFORTS = frozenset({"minimal", "low", "medium", "high"})


def _env(name: str, legacy_name: str | None = None, default: str = "") -> str:
    """``AI_*`` 값을 우선하고, 기존 배포의 ``OPENAI_*`` 값을 대체로 읽는다."""
    value = os.getenv(name, "").strip()
    if not value and legacy_name is not None:
        value = os.getenv(legacy_name, "").strip()
    return value or default


@dataclass(frozen=True)
class AISettings:
    """AI Provider의 서버 전용 접속 설정.

    Attributes:
        provider: ``openai``는 OpenAI Responses API, ``openai_compatible``은 vLLM 등
            OpenAI 호환 Chat Completions API다.
        base_url: ``/responses`` 또는 ``/chat/completions`` 앞의 API 기준 주소다.
        api_key: Bearer 토큰이다. 인증 없는 내부 vLLM에서는 ``None``일 수 있다.
        reasoning_effort: 추론 모델의 사고량이다. 낮을수록 첫 글자가 빨라진다.
        disable_thinking: Qwen3 계열처럼 chat template에서 사고 모드를 끌 수 있는
            모델에 ``enable_thinking=false``를 전달할지 여부다.
    """

    provider: str
    base_url: str
    api_key: str | None
    model: str
    timeout_seconds: float
    reasoning_effort: str | None = None
    disable_thinking: bool = True
    max_output_tokens: int | None = None

    @classmethod
    def from_environment(cls) -> "AISettings":
        """환경 변수에서 Provider 종류·주소·API Key·모델·시간 초과를 읽는다.

        API Key는 서버에서만 주입한다. ``NEXT_PUBLIC_*`` 변수는 브라우저에 노출될 수
        있으므로 여기서 읽지 않는다. ``AI_*``가 없으면 기존 ``OPENAI_*``를 읽는다.
        """
        provider = _env("AI_PROVIDER", default=AI_PROVIDER_OPENAI).lower()
        if provider not in {AI_PROVIDER_OPENAI, AI_PROVIDER_OPENAI_COMPATIBLE}:
            raise ConfigurationError("AI_PROVIDER must be openai or openai_compatible")
        is_openai = provider == AI_PROVIDER_OPENAI
        # OPENAI_* values belong to OpenAI only.  They must not leak to a
        # self-hosted server when an old .env still contains the OpenAI key.
        legacy = (lambda name: f"OPENAI_{name}") if is_openai else (lambda name: None)
        default_base_url = "https://api.openai.com/v1" if is_openai else ""
        base_url = _env("AI_BASE_URL", legacy("BASE_URL"), default_base_url).rstrip("/")
        if not base_url.startswith(("https://", "http://")):
            raise ConfigurationError("AI_BASE_URL must be an HTTP(S) URL")
        if is_openai and not base_url.startswith("https://"):
            raise ConfigurationError("AI_BASE_URL must use HTTPS for the openai provider")
        api_key = _env("AI_API_KEY", legacy("API_KEY")) or None
        if is_openai and api_key is None:
            raise ConfigurationError("AI_API_KEY or OPENAI_API_KEY is required")
        model = _env("AI_MODEL", legacy("MODEL"), "gpt-5-mini" if is_openai else "")
        if not model:
            raise ConfigurationError("AI_MODEL is required")
        try:
            timeout_seconds = float(_env("AI_TIMEOUT_SECONDS", "OPENAI_TIMEOUT_SECONDS", "30"))
        except ValueError as error:
            raise ConfigurationError("AI_TIMEOUT_SECONDS must be numeric") from error
        if timeout_seconds <= 0:
            raise ConfigurationError("AI_TIMEOUT_SECONDS must be positive")
        reasoning_effort = _env("AI_REASONING_EFFORT", "OPENAI_REASONING_EFFORT").lower() or None
        if reasoning_effort is not None and reasoning_effort not in _REASONING_EFFORTS:
            raise ConfigurationError("AI_REASONING_EFFORT must be minimal, low, medium, or high")
        disable_thinking = _env("AI_DISABLE_THINKING", default="true").lower()
        if disable_thinking not in {"true", "false"}:
            raise ConfigurationError("AI_DISABLE_THINKING must be true or false")
        max_output_tokens: int | None = None
        if raw_max := _env("AI_MAX_OUTPUT_TOKENS"):
            try:
                max_output_tokens = int(raw_max)
            except ValueError as error:
                raise ConfigurationError("AI_MAX_OUTPUT_TOKENS must be an integer") from error
            if max_output_tokens <= 0:
                raise ConfigurationError("AI_MAX_OUTPUT_TOKENS must be positive")
        return cls(
            provider=provider,
            base_url=base_url,
            api_key=api_key,
            model=model,
            timeout_seconds=timeout_seconds,
            reasoning_effort=reasoning_effort,
            disable_thinking=disable_thinking == "true",
            max_output_tokens=max_output_tokens,
        )


# Kept as an import-compatible name for the manual live tests.
OpenAISettings = AISettings


@dataclass(frozen=True)
class BridgeSnapshotSettings:
    """Bridge의 환자 EMR Snapshot API 접속 정보.

    Attributes:
        bridge_base_url: 예를 들어 ``https://bridge.example.org``인 Bridge 기준 주소다.
        snapshot_path: 환자 Snapshot endpoint 경로다.
        method: Bridge Snapshot API의 HTTP 방식이다.
        timeout_seconds: Bridge 응답을 기다리는 양수 초 단위 시간이다.
        access_token: AX-G AI가 Bridge에 보내는 Bearer 토큰이다.
    """

    bridge_base_url: str
    snapshot_path: str
    method: str
    timeout_seconds: float
    access_token: str | None
    send_request_context: bool = True

    @classmethod
    def from_environment(cls) -> "BridgeSnapshotSettings":
        """환경 변수에서 Bridge Snapshot 접속 정보를 읽어 검증한다.

        Required:
            AX_G_BRIDGE_BASE_URL: Bridge API 기준 주소다.

        Optional:
            AX_G_BRIDGE_SNAPSHOT_PATH: 기본 Snapshot endpoint 경로다.
            AX_G_BRIDGE_METHOD: ``GET`` 또는 ``POST``이며 기본은 ``POST``다.
            AX_G_BRIDGE_TIMEOUT_SECONDS: 양수 초 단위이며 기본은 ``10``이다.
            AX_G_BRIDGE_ACCESS_TOKEN: AX-G AI가 Bridge에 보내는 Bearer 토큰이다.
        """
        bridge_base_url = os.getenv("AX_G_BRIDGE_BASE_URL", "").strip()
        if not bridge_base_url.startswith(("https://", "http://")):
            raise ConfigurationError("AX_G_BRIDGE_BASE_URL must be an HTTP(S) URL")
        method = os.getenv("AX_G_BRIDGE_METHOD", "POST").upper().strip()
        if method not in {"GET", "POST"}:
            raise ConfigurationError("AX_G_BRIDGE_METHOD must be GET or POST")
        try:
            timeout_seconds = float(os.getenv("AX_G_BRIDGE_TIMEOUT_SECONDS", "10"))
        except ValueError as error:
            raise ConfigurationError("AX_G_BRIDGE_TIMEOUT_SECONDS must be numeric") from error
        if timeout_seconds <= 0:
            raise ConfigurationError("AX_G_BRIDGE_TIMEOUT_SECONDS must be positive")
        token = os.getenv("AX_G_BRIDGE_ACCESS_TOKEN")
        return cls(
            bridge_base_url=bridge_base_url.rstrip("/"),
            snapshot_path=os.getenv("AX_G_BRIDGE_SNAPSHOT_PATH", "/internal/v1/patient-context/snapshot").strip(),
            method=method,
            timeout_seconds=timeout_seconds,
            access_token=token.strip() if token and token.strip() else None,
            send_request_context=True,
        )
