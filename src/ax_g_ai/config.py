"""배포 환경별 외부 연동 설정.

EMR 주소·경로·HTTP 방식·시간 초과·서버 측 인증 토큰은 코드에 넣지 않고 환경 변수로
주입한다. 설정값을 로그나 감사 이벤트에 기록하지 않으며, 실제 비밀값은 배포 환경의
비밀 저장소에서 ``AX_G_BRIDGE_ACCESS_TOKEN``으로 제공해야 한다.
"""

from __future__ import annotations

from dataclasses import dataclass
import os


class ConfigurationError(ValueError):
    """필수 환경 설정이 없거나 허용되지 않은 값일 때 발생한다."""


@dataclass(frozen=True)
class OpenAISettings:
    """OpenAI Responses API의 서버 전용 접속 설정."""

    api_key: str
    model: str
    timeout_seconds: float

    @classmethod
    def from_environment(cls) -> "OpenAISettings":
        """환경 변수에서 API Key·모델·시간 초과를 읽는다.

        ``OPENAI_API_KEY``는 서버에서만 주입한다. ``NEXT_PUBLIC_*`` 변수는
        브라우저에 노출될 수 있으므로 여기서 읽지 않는다.
        """
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise ConfigurationError("OPENAI_API_KEY is required")
        model = os.getenv("OPENAI_MODEL", "gpt-5-mini").strip()
        if not model:
            raise ConfigurationError("OPENAI_MODEL is required")
        try:
            timeout_seconds = float(os.getenv("OPENAI_TIMEOUT_SECONDS", "30"))
        except ValueError as error:
            raise ConfigurationError("OPENAI_TIMEOUT_SECONDS must be numeric") from error
        if timeout_seconds <= 0:
            raise ConfigurationError("OPENAI_TIMEOUT_SECONDS must be positive")
        return cls(api_key, model, timeout_seconds)


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
