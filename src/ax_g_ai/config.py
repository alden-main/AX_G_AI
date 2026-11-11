"""배포 환경별 설정.

AI Provider 주소·API Key·모델과 런타임 경로는 코드에 넣지 않고 환경 변수로 주입한다.
설정값을 로그나 오류 메시지에 기록하지 않는다. 검증 실패는 필드 이름과 사유만 담은
``ConfigurationError``로 바뀐다.
"""

from __future__ import annotations

import os
from typing import Any, Literal

from pydantic import (
    DirectoryPath,
    Field,
    FilePath,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    SecretStr,
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
        evidence_corpus_dir: 최신화 말뭉치(``text_data/corpus``) 경로다. 없으면 근거 인용을 끈다.
        evidence_include_unreviewed: 의료진 미검수(draft) 근거·규칙도 쓸지 여부다(경고 문구 없이 출처로 근거를 밝힌다).
            기본값은 끔이며, 켜도 A등급 출처·원문 확인 항목만 대상이다.
        evidence_max_chunks: 질문 하나에 인용할 근거 문단 최대 개수다.
        corpus_database_url: 말뭉치 PostgreSQL(pgvector) 접속 URL이다. 설정하면 DB 하이브리드 검색을 쓰고,
            연결 실패 시 ``evidence_corpus_dir`` 의 파일 검색으로 대체한다. 값은 로그에 쓰지 않는다.
        embedding_url: 내부 TEI 임베딩 서버 주소다(예: ``http://embedding:80``). 없으면 어휘 검색만 한다.
        embedding_model: 저장된 임베딩을 구분하는 모델 ID다(적재기와 같아야 한다).
        evidence_similarity_floor: 의미 검색만으로 근거를 인용할 최소 코사인 유사도다.
        clinical_rules_enabled: 말뭉치 진단·검사·협진·추적 규칙(2단계)을 환자 EMR에 적용해 추천 의견을 낼지 여부다.
            규칙은 근거와 같은 게이트를 지나며, 미검수 규칙은 ``evidence_include_unreviewed`` 가 켜져 있을 때만 쓴다.
        site_code_map_path: 병원 검사·약품 코드 → 말뭉치 개념 매핑 JSON(``text_data/mappings/sites``)이다.
            없으면 LOINC 검사 코드와 처방명 속 성분명으로만 매핑한다.
        recommendation_max_evidence: 추천 의견 하나에 붙일 규칙 근거 문단 최대 개수다.
        clinical_doses_enabled: 말뭉치 용량 규칙(3단계)을 현재 처방·의료진 지정 약에 적용해 체중·신기능 기준 참고 용량,
            용량 조건 규칙, 처방 1일 용량 대조를 낼지 여부다(기본 끔). 규칙은 2단계와 같은 게이트를 지나며 미검수 규칙은
            ``evidence_include_unreviewed`` 가 켜져 있을 때만 쓰고, 항목마다 출처·region(국외 기준 여부)이 붙는다.
        dur_enabled: 심평원 DUR 의약품 목록(PostgreSQL ``dur`` 스키마, ``ax-g-ai-load-dur`` 로 적재)으로 추천 의견·처방 관련
            채팅에서 현재 처방의 병용금기·연령금기·임부금기·노인주의를 점검할지 여부다(공공누리 제1유형 — 출처 표시).
            ``corpus_database_url`` 의 DB를 쓰며, DB나 활성 데이터셋이 없으면 경고 후 점검을 끈다.
        attributed_sources_only: 출처를 밝히면 쓸 수 있는 자료만 AI에 쓴다(기본 true). 켜면 (1) 근거·규칙의 모든 출처에
            라이선스(CC0·CC BY·퍼블릭 도메인)와 제목이 있어야 하고, (2) 출처·라이선스가 확인되지 않은 참조 DB
            (``reference_data_db_path``, 약품 효능·주의사항)를 쓰지 않으며, (3) 추천 의견에서 출처 없는 일반 의학 지식 제안을 막는다.
    """

    model_config = SettingsConfigDict(env_prefix="AX_G_")

    reference_data_db_path: FilePath | None = None
    evidence_corpus_dir: DirectoryPath | None = None
    evidence_include_unreviewed: bool = False
    evidence_max_chunks: int = Field(default=3, ge=1, le=5)
    corpus_database_url: SecretStr | None = None
    embedding_url: str | None = None
    embedding_model: str = "kure-v1"
    embedding_timeout_seconds: PositiveFloat = 10
    evidence_similarity_floor: float = Field(default=0.6, ge=0, le=1)
    clinical_rules_enabled: bool = True
    site_code_map_path: FilePath | None = None
    recommendation_max_evidence: int = Field(default=10, ge=1, le=20)
    attributed_sources_only: bool = True
    clinical_doses_enabled: bool = False
    dur_enabled: bool = True

    @field_validator("embedding_url")
    @classmethod
    def _embedding_http_url(cls, value: str | None) -> str | None:
        if value is not None and not value.startswith(("https://", "http://")):
            raise ValueError("must be an HTTP(S) URL")
        return value
    patient_context_ttl_seconds: PositiveFloat = 12 * 60 * 60
    patient_context_max_entries: PositiveInt = 1000
