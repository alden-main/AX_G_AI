"""AI 호출의 OpenTelemetry 추적 설정.

설계상 위치: 어댑터 계층의 관측(observability) 설정. ``runtime``이 Provider Client를 만들기 전에 호출해 결과를
``build_provider_client``에 넘긴다.

역할: 표준 ``OTEL_*`` 환경 변수(``OTEL_EXPORTER_OTLP_ENDPOINT``, ``OTEL_EXPORTER_OTLP_HEADERS``,
``OTEL_SERVICE_NAME``, ``OTEL_RESOURCE_ATTRIBUTES`` 등)를 그대로 쓰며, OTLP 주소가 없으면 추적을 켜지 않는다.
기본 수신처는 Arize Phoenix(``compose.observability.yaml``)이며, OTLP/HTTP를 받는 다른 도구(Langfuse·MLflow 등)도 된다.

``include_content``가 거짓이면 span에 프롬프트·지시문·응답 본문을 담지 않고 모델·토큰 수·
소요 시간·종료 사유만 남긴다.
"""

from __future__ import annotations

import os

from openinference.instrumentation.pydantic_ai import OpenInferenceSpanProcessor
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from pydantic_ai import InstrumentationSettings

_ENDPOINT_VARIABLES = ("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", "OTEL_EXPORTER_OTLP_ENDPOINT")


def instrumentation_from_environment(*, include_content: bool) -> InstrumentationSettings | None:
    """OTLP 주소가 설정돼 있으면 내보내기를 켜고 Pydantic AI 계측 설정을 돌려준다.

    역할: ``OTEL_EXPORTER_OTLP_TRACES_ENDPOINT``나 ``OTEL_EXPORTER_OTLP_ENDPOINT``가 있으면 서비스 이름을 붙인
    ``TracerProvider``에 OpenInference 처리기와 OTLP/HTTP 일괄 내보내기를 달아 전역 Tracer Provider로 등록한다.
    입력:
        include_content: 참이면 프롬프트·지시문·응답 본문과 요청 파라미터를 span에 담는다.
    출력: ``InstrumentationSettings``. OTLP 주소가 없으면 ``None``.
    """
    if not any(os.getenv(name, "").strip() for name in _ENDPOINT_VARIABLES):
        return None
    # ``Resource.create``에서는 명시한 속성이 ``OTEL_SERVICE_NAME``보다 우선하므로 여기서 변수를 직접 읽는다.
    provider = TracerProvider(resource=Resource.create({SERVICE_NAME: os.getenv("OTEL_SERVICE_NAME") or "ax-g-ai"}))
    # OpenInference 속성(span에 이미 있는 GenAI 속성에서 파생, 새 정보 없음)을 더해 Phoenix가 호출을 토큰 수와
    # (본문을 켰으면) 메시지가 담긴 LLM span으로 보여 준다. 내보내기보다 먼저 실행돼야 한다.
    provider.add_span_processor(OpenInferenceSpanProcessor())
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    trace.set_tracer_provider(provider)
    return InstrumentationSettings(
        tracer_provider=provider,
        include_content=include_content,
        # 요청 파라미터에 지시문 본문이 들어 있으므로 본문 포함 스위치를 따른다.
        include_model_request_parameters=include_content,
    )
