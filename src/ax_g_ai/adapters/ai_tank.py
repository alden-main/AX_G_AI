"""D-11 AI-Tank 비스트리밍 상담 API의 서버 측 Adapter.

이 Adapter는 API Key를 ``x-api-key`` 헤더에만 넣고, UI·감사 이벤트·예외 메시지에
비밀값이나 제공사 원문 오류를 남기지 않는다. 초기 의료진 챗봇은 원문 청크 스트리밍과
Provider 요약을 사용하지 않으므로 두 요청 플래그를 ``False``로 강제한다.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Protocol
from urllib.error import HTTPError
from urllib.request import Request, urlopen


class AiTankProviderError(RuntimeError):
    """안전하게 분류된 AI-Tank 호출 또는 응답 검증 실패다."""


@dataclass(frozen=True)
class ChatHistoryItem:
    """현재 환자 세션에서만 유효한 이전 대화 한 차례.

    ``inputs``와 ``outputs``는 AI-Tank 계약이 요구하는 문자열이다. 환자 전환이나
    권한 해제 후에는 D-03의 세션 상태 폐기 과정에서 이 데이터를 전달하면 안 된다.
    """

    inputs: str
    outputs: str


@dataclass(frozen=True)
class ConsultRequest:
    """AI-Tank ``POST /api/consult`` 비스트리밍 요청의 내부 DTO.

    Attributes:
        history: 검증된 현재 세션 대화만 담은 ``ChatHistoryItem`` 튜플이다.
        question: 의료진의 일반화된 지식 질의 문자열이다. Adapter는 환자 원천값을
            자동으로 삽입하지 않는다.
        language: 제공사에 전달할 응답 언어 문자열이다.
    """

    history: tuple[ChatHistoryItem, ...]
    question: str
    language: str


@dataclass(frozen=True)
class ProviderAnswer:
    """AI-Tank의 검증된 ``answer`` 문자열 응답."""

    answer: str


@dataclass(frozen=True)
class HttpResponse:
    """HTTP transport가 반환하는 상태·Content-Type·본문 바이트."""

    status_code: int
    content_type: str | None
    body: bytes


class HttpTransport(Protocol):
    """실제 HTTP 또는 계약 시험 구현체가 따르는 최소 전송 인터페이스."""

    def post(self, url: str, headers: dict[str, str], body: bytes, timeout_seconds: float) -> HttpResponse: ...


class UrllibHttpTransport:
    """표준 라이브러리로 AI-Tank에 HTTPS POST를 보내는 전송 구현체."""

    def post(self, url: str, headers: dict[str, str], body: bytes, timeout_seconds: float) -> HttpResponse:
        request = Request(url, data=body, headers=headers, method="POST")
        try:
            with urlopen(request, timeout=timeout_seconds) as response:
                return HttpResponse(
                    status_code=response.status,
                    content_type=response.headers.get("Content-Type"),
                    body=response.read(),
                )
        except HTTPError as error:
            return HttpResponse(
                status_code=error.code,
                content_type=error.headers.get("Content-Type"),
                body=error.read(),
            )


class AiTankClient:
    """D-11의 요청 직렬화·응답 검증·안전 오류 변환을 담당한다.

    생성 시 제공된 ``HttpTransport``가 실제 네트워크 처리를 담당한다. 따라서 계약
    시험에서는 가짜 transport를 주입할 수 있고, 호출자는 API Key를 로그나 브라우저에
    전달하지 않는다. endpoint와 timeout은 운영 계약이 확정되면 설정 계층에서 주입한다.
    """

    def __init__(
        self,
        transport: HttpTransport,
        api_key: str,
        endpoint: str,
        timeout_seconds: float,
    ) -> None:
        if not api_key.strip():
            raise ValueError("AI-Tank API key is required")
        if timeout_seconds <= 0:
            raise ValueError("AI-Tank timeout must be positive")
        self._transport = transport
        self._api_key = api_key
        self._endpoint = endpoint
        self._timeout_seconds = timeout_seconds

    def consult(self, request: ConsultRequest) -> ProviderAnswer:
        """비스트리밍 상담 요청을 전송하고 비어 있지 않은 답변만 반환한다.

        Args:
            request: 현재 환자 세션에 한정된 대화 이력, 일반화된 질문, 언어를 담은
                ``ConsultRequest``다. 스트리밍·Provider 요약은 이 DTO에 없다.

        Returns:
            JSON ``{\"answer\": string}``을 검증해 만든 ``ProviderAnswer``다.

        Raises:
            AiTankProviderError: 전송 실패, 비성공 HTTP 상태, JSON 이외 Content-Type,
                잘못된 JSON 또는 빈 답변에 발생한다. 제공사 원문 오류는 노출하지 않는다.
        """
        self._validate_request(request)
        body = json.dumps(
            {
                "history": [
                    {"inputs": item.inputs, "outputs": item.outputs}
                    for item in request.history
                ],
                "question": request.question,
                "language": request.language,
                "streaming": False,
                "summarization": False,
            }
        ).encode("utf-8")
        try:
            response = self._transport.post(
                self._endpoint,
                {"Content-Type": "application/json", "x-api-key": self._api_key},
                body,
                self._timeout_seconds,
            )
        except Exception as error:
            raise AiTankProviderError("AI-Tank transport failed") from error
        return self._parse_response(response)

    @staticmethod
    def _validate_request(request: ConsultRequest) -> None:
        if not request.question.strip() or not request.language.strip():
            raise AiTankProviderError("AI-Tank request requires question and language")
        if any(not item.inputs.strip() or not item.outputs.strip() for item in request.history):
            raise AiTankProviderError("AI-Tank history contains an empty message")

    @staticmethod
    def _parse_response(response: HttpResponse) -> ProviderAnswer:
        if not 200 <= response.status_code < 300:
            # The status is diagnostic metadata, unlike the provider response
            # body which may contain sensitive data and is intentionally never
            # logged or returned to callers.
            raise AiTankProviderError(
                f"AI-Tank returned a non-success status (status={response.status_code})"
            )
        try:
            decoded_body = response.body.decode("utf-8")
        except UnicodeDecodeError as error:
            raise AiTankProviderError("AI-Tank returned invalid text") from error

        # AI-Tank currently returns a ``data: {"content":"{...}"}`` frame even
        # when ``streaming`` is false.  Support that documented-on-the-wire shape
        # without accepting arbitrary plain-text responses as clinical output.
        if decoded_body.lstrip().startswith("data:"):
            return AiTankClient._parse_sse_response(decoded_body)
        if not response.content_type or not response.content_type.lower().startswith("application/json"):
            raise AiTankProviderError("AI-Tank returned an unexpected content type")
        try:
            payload = json.loads(decoded_body)
        except json.JSONDecodeError as error:
            raise AiTankProviderError("AI-Tank returned invalid JSON") from error
        return AiTankClient._answer_from_payload(payload)

    @staticmethod
    def _parse_sse_response(body: str) -> ProviderAnswer:
        for line in body.splitlines():
            if not line.startswith("data:"):
                continue
            event_data = line.removeprefix("data:").strip()
            if not event_data or event_data == "[DONE]":
                continue
            try:
                outer_payload = json.loads(event_data)
                content = outer_payload.get("content") if isinstance(outer_payload, dict) else None
                payload = json.loads(content) if isinstance(content, str) else None
            except json.JSONDecodeError as error:
                raise AiTankProviderError("AI-Tank returned invalid SSE JSON") from error
            return AiTankClient._answer_from_payload(payload)
        raise AiTankProviderError("AI-Tank returned no answer event")

    @staticmethod
    def _answer_from_payload(payload: object) -> ProviderAnswer:
        answer = payload.get("answer") if isinstance(payload, dict) else None
        if not isinstance(answer, str) or not answer.strip():
            raise AiTankProviderError("AI-Tank returned an empty or invalid answer")
        return ProviderAnswer(answer=answer)
