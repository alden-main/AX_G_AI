"""배포 환경별 외부 연동 설정.

EMR 주소·경로·HTTP 방식·시간 초과·서버 측 인증 토큰은 코드에 넣지 않고 환경 변수로
주입한다. 설정값을 로그나 감사 이벤트에 기록하지 않으며, 실제 비밀값은 배포 환경의
비밀 저장소에서 ``AX_G_EMR_ACCESS_TOKEN``으로 제공해야 한다.
"""

from __future__ import annotations

from dataclasses import dataclass
import os


class ConfigurationError(ValueError):
    """필수 환경 설정이 없거나 허용되지 않은 값일 때 발생한다."""


@dataclass(frozen=True)
class AiTankSettings:
    """AI-Tank 상담 API의 서버 전용 접속 설정."""

    api_key: str
    endpoint: str
    timeout_seconds: float

    @classmethod
    def from_environment(cls) -> "AiTankSettings":
        """환경 변수에서 API Key와 endpoint를 읽는다.

        ``AX_G_AI_TANK_API_KEY``는 서버에서만 주입한다. ``NEXT_PUBLIC_*`` 변수는
        브라우저에 노출될 수 있으므로 여기서 읽지 않는다.
        """
        api_key = os.getenv("AX_G_AI_TANK_API_KEY", "").strip()
        if not api_key:
            raise ConfigurationError("AX_G_AI_TANK_API_KEY is required")
        endpoint = os.getenv(
            "AX_G_AI_TANK_ENDPOINT", "https://ahk-api.ai-tank.co.kr/api/consult"
        ).strip()
        if not endpoint.startswith(("https://", "http://")):
            raise ConfigurationError("AX_G_AI_TANK_ENDPOINT must be an HTTP(S) URL")
        try:
            timeout_seconds = float(os.getenv("AX_G_AI_TANK_TIMEOUT_SECONDS", "30"))
        except ValueError as error:
            raise ConfigurationError("AX_G_AI_TANK_TIMEOUT_SECONDS must be numeric") from error
        if timeout_seconds <= 0:
            raise ConfigurationError("AX_G_AI_TANK_TIMEOUT_SECONDS must be positive")
        return cls(api_key, endpoint, timeout_seconds)


@dataclass(frozen=True)
class EmrApiSettings:
    """EMR 환자 문맥 API의 접속 정보.

    Attributes:
        base_url: 예를 들어 ``https://emr.example.org``인 EMR API 기준 주소다.
        context_path: 환자 문맥 endpoint 경로다. ``{patient_id}``, ``{encounter_id}``
            치환자를 사용할 수 있다.
        method: ``GET`` 또는 ``POST``이며, 제공사 계약에 맞게 설정한다.
        timeout_seconds: EMR 응답을 기다리는 양수 초 단위 시간이다.
        access_token: 선택적인 서버 측 Bearer 토큰이다. 코드·UI·감사 로그에 노출하지 않는다.
    """

    base_url: str
    context_path: str
    method: str
    timeout_seconds: float
    access_token: str | None
    send_request_context: bool = True

    @classmethod
    def from_environment(cls) -> "EmrApiSettings":
        """환경 변수에서 EMR 접속 정보를 읽어 검증된 설정 객체로 만든다.

        Required:
            AX_G_EMR_BASE_URL: 제공사가 전달한 API 기준 주소다.

        Optional:
            AX_G_EMR_CONTEXT_PATH: 기본 ``/patient-context`` 경로 또는 치환 경로다.
            AX_G_EMR_METHOD: ``GET`` 또는 ``POST``이며 기본은 ``POST``다.
            AX_G_EMR_TIMEOUT_SECONDS: 양수 초 단위이며 기본은 ``10``이다.
            AX_G_EMR_ACCESS_TOKEN: 선택적인 서버 측 Bearer 토큰이다.
        """
        base_url = os.getenv("AX_G_EMR_BASE_URL", "").strip()
        if not base_url.startswith(("https://", "http://")):
            raise ConfigurationError("AX_G_EMR_BASE_URL must be an HTTP(S) URL")
        method = os.getenv("AX_G_EMR_METHOD", "POST").upper().strip()
        if method not in {"GET", "POST"}:
            raise ConfigurationError("AX_G_EMR_METHOD must be GET or POST")
        try:
            timeout_seconds = float(os.getenv("AX_G_EMR_TIMEOUT_SECONDS", "10"))
        except ValueError as error:
            raise ConfigurationError("AX_G_EMR_TIMEOUT_SECONDS must be numeric") from error
        if timeout_seconds <= 0:
            raise ConfigurationError("AX_G_EMR_TIMEOUT_SECONDS must be positive")
        token = os.getenv("AX_G_EMR_ACCESS_TOKEN")
        return cls(
            base_url=base_url.rstrip("/"),
            context_path=os.getenv("AX_G_EMR_CONTEXT_PATH", "/patient-context").strip(),
            method=method,
            timeout_seconds=timeout_seconds,
            access_token=token.strip() if token and token.strip() else None,
            send_request_context=os.getenv("AX_G_EMR_SEND_REQUEST_CONTEXT", "true").lower() == "true",
        )
