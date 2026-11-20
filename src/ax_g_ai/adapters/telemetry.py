"""AI 호출의 OpenTelemetry 추적 설정.

표준 ``OTEL_*`` 환경 변수(``OTEL_EXPORTER_OTLP_ENDPOINT``, ``OTEL_EXPORTER_OTLP_HEADERS``,
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
    """OTLP 주소가 설정돼 있으면 내보내기를 켜고 Pydantic AI 계측 설정을 돌려준다."""
    if not any(os.getenv(name, "").strip() for name in _ENDPOINT_VARIABLES):
        return None
    # Explicit attributes win over OTEL_SERVICE_NAME in Resource.create, so the variable is read here.
    provider = TracerProvider(resource=Resource.create({SERVICE_NAME: os.getenv("OTEL_SERVICE_NAME") or "ax-g-ai"}))
    # Adds OpenInference attributes (derived from the GenAI ones already on the span, nothing new) so Phoenix
    # shows calls as LLM spans with tokens and, when content is enabled, the messages. Must run before the export.
    provider.add_span_processor(OpenInferenceSpanProcessor())
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    trace.set_tracer_provider(provider)
    return InstrumentationSettings(
        tracer_provider=provider,
        include_content=include_content,
        # The request parameters carry the instructions text, so they follow the content switch.
        include_model_request_parameters=include_content,
    )
