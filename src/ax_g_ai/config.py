"""배포 환경별 설정을 환경 변수에서 읽고 검증한다.

설계상 위치: 설정 계층. ``runtime.py``가 ``AISettings``·``RuntimeSettings``로 앱을 조립하고, 적재 CLI
(``ax-g-ai-load-corpus``, ``ax-g-ai-load-dur``)가 ``LoaderSettings``를 쓴다.

역할: AI Provider 주소·API Key·모델과 런타임 경로는 코드에 넣지 않고 환경 변수로 주입한다.
설정값을 로그나 오류 메시지에 기록하지 않는다. 검증 실패는 필드 이름과 사유만 담은
``ConfigurationError``로 바뀐다.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated, Any, Final, Literal

from pydantic import (
    AfterValidator,
    DirectoryPath,
    Field,
    FilePath,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    SecretStr,
    ValidationError,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict

from ax_g_ai.corpus.files import DEFAULT_EMBEDDING_MODEL
from ax_g_ai.prompts.clinical import ModelProfile
from ax_g_ai.prompts.data_labels import PromptLabels


class ConfigurationError(ValueError):
    """필수 환경 설정이 없거나 허용되지 않은 값일 때 발생한다.

    역할: pydantic 검증 오류를 환경 변수 이름과 사유만 담은 메시지로 바꿔 전달한다(설정값은 담지 않는다).
    """


def _require_http_url(value: str) -> str:
    """``http://``·``https://``로 시작하지 않으면 ``ValueError``를 낸다."""
    if not value.startswith(("https://", "http://")):
        raise ValueError("must be an HTTP(S) URL")
    return value


HttpUrlText = Annotated[str, AfterValidator(_require_http_url)]

AI_PROVIDER_OPENAI: Final = "openai"
AI_PROVIDER_OPENAI_COMPATIBLE: Final = "openai_compatible"

_OPENAI_DEFAULTS = {"base_url": "https://api.openai.com/v1", "model": "gpt-5-mini"}
# 대응하는 ``AI_*``가 없으면 예전 ``OPENAI_*`` 이름을 읽는다. 인증 정보와 주소는 OpenAI 전용이므로,
# 예전 .env에 OpenAI 키가 남아 있어도 자체 호스팅 서버로 새어 나가면 안 된다.
_OPENAI_ONLY_LEGACY_FIELDS = ("base_url", "api_key", "model")
_SHARED_LEGACY_FIELDS = ("timeout_seconds", "reasoning_effort")


class _EnvironmentSettings(BaseSettings):
    """환경 변수 설정 클래스들의 공통 기반.

    역할: 빈 값 무시·앞뒤 공백 제거·불변 설정을 공유하고, 검증 실패를 ``ConfigurationError``로 바꾸는
    ``from_environment``를 제공한다.
    """

    model_config = SettingsConfigDict(env_ignore_empty=True, str_strip_whitespace=True, frozen=True)

    @classmethod
    def from_environment(cls):
        """환경 변수로 설정 객체를 만든다.

        역할: 검증 오류마다 ``<접두사><필드 이름 대문자>: 사유``를 이어 붙인 ``ConfigurationError``로 바꾼다.
        입력값(설정값)은 메시지에 넣지 않는다.
        입력: 없음(환경 변수·``.env``를 읽는다).
        출력: 검증된 설정 객체.
        예외: ``ConfigurationError`` — 필수 값 누락 또는 허용되지 않은 값.
        """
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
    """AI Provider의 서버 전용 접속 설정(``AI_`` 환경 변수).

    역할: Provider 종류·주소·인증·모델·시간 제한·재시도·추적 여부를 검증해 ``build_provider_client``에 넘긴다.

    Attributes:
        provider: ``openai``는 OpenAI Responses API, ``openai_compatible``은 vLLM 등
            OpenAI 호환 Chat Completions API다.
        base_url: ``/responses`` 또는 ``/chat/completions`` 앞의 API 기준 주소다.
        api_key: Bearer 토큰이다. 인증 없는 내부 vLLM에서는 ``None``일 수 있다.
        reasoning_effort: 추론 모델의 사고량이다. 낮을수록 첫 글자가 빨라진다.
        disable_thinking: Qwen3 계열처럼 chat template에서 사고 모드를 끌 수 있는
            모델에 ``enable_thinking=false``를 전달할지 여부다. L1처럼 사고 모드로 학습한 모델은 끄면 품질이 떨어진다.
        temperature: Chat Completions 샘플링 온도(0~2)다. ``None``이면 보내지 않아 추론 서버 기본값을 쓰는데,
            모델 ``generation_config``에 값이 없으면 SGLang은 1.0을 써서 같은 입력에도 답이 크게 흔들린다.
            OpenAI 추론 모델(Responses API)은 이 값을 받지 않으므로 ``openai_compatible``에서만 쓴다.
        repetition_penalty: 반복 억제 계수(``openai_compatible`` 전용, ``extra_body``로 보냄)다. ``None``이면 보내지 않는다.
            L1-30B-A5B 모델 카드는 1.05를 권장한다(1.0이면 긴 임상 답변에서 같은 구절을 반복).
        system_prompt_prefix: 모든 지시문 앞에 두는 한 줄(``openai_compatible`` 전용)이다. 모델이 정렬 학습한 신원 문장을 넣는다
            (L1-30B-A5B: ``You are Chain-of-Evidence, a medical AI assistant developed by Lunit.``). ``None``이면 넣지 않는다.
        no_thinking_operations: 사고를 끄고 보낼 작업 이름(쉼표로 구분, ``openai_compatible`` 전용)이다. 사고 토큰은 답변 첫 글자보다
            먼저 생성되므로, 추론이 거의 필요 없는 작업(예: ``one_shot_patient_summary_stream`` 환자 요약)만 끄면 첫 글자가 빨라진다.
            ``disable_thinking``이 참이면 이미 모든 작업에서 꺼져 있다. 비우면 모든 작업이 같은 설정을 쓴다.
        max_retries: 첫 토큰 전 연결 오류·429·5xx의 SDK 재시도 횟수다.
        trace_content: OpenTelemetry 추적(``OTEL_EXPORTER_OTLP_ENDPOINT`` 설정 시)에 프롬프트·지시문·응답 본문을
            담을지 여부다(기본 끔). 끄면 모델·토큰 수·소요 시간·종료 사유만 남는다. 비식별 EMR도 추적 저장소에
            쌓이므로 운영에서 켜려면 별도 승인이 필요하다.
    """

    model_config = SettingsConfigDict(env_prefix="AI_")

    provider: Literal["openai", "openai_compatible"] = AI_PROVIDER_OPENAI
    base_url: str = ""
    api_key: str | None = None
    model: str = ""
    timeout_seconds: PositiveFloat = 30
    reasoning_effort: Literal["minimal", "low", "medium", "high"] | None = None
    disable_thinking: bool = True
    temperature: Annotated[float, Field(ge=0, le=2)] | None = None
    repetition_penalty: Annotated[float, Field(gt=0, le=2)] | None = None
    system_prompt_prefix: Annotated[str, Field(max_length=500)] | None = None
    no_thinking_operations: str = ""
    max_output_tokens: PositiveInt | None = None
    max_retries: NonNegativeInt = 2
    trace_content: bool = False

    @model_validator(mode="before")
    @classmethod
    def _apply_legacy_and_provider_defaults(cls, data: Any) -> Any:
        """검증 전에 예전 ``OPENAI_*`` 값과 OpenAI 기본값을 채운다.

        역할: ``provider``를 소문자로 정규화하고, 비어 있는 필드는 ``OPENAI_<필드>`` 환경 변수에서 채운다. OpenAI가
        아니면 ``timeout_seconds``·``reasoning_effort``만 가져온다. OpenAI면 ``base_url``·``model`` 기본값을 넣는다.
        끝으로 문자열 ``reasoning_effort``는 Provider와 관계없이 소문자로 바꾼다.
        입력:
            data: 검증 전 입력 값 사전.
        출력: 보정한 값 사전.
        """
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

    @property
    def no_thinking_operation_names(self) -> frozenset[str]:
        """``no_thinking_operations``를 작업 이름 집합으로 나눈다(공백·빈 항목 무시)."""
        return frozenset(name.strip() for name in self.no_thinking_operations.split(",") if name.strip())

    @field_validator("base_url")
    @classmethod
    def _http_url(cls, value: str) -> str:
        """HTTP(S) 주소인지 확인하고 끝의 ``/``를 없앤다."""
        return _require_http_url(value).rstrip("/")

    @model_validator(mode="after")
    def _openai_requirements(self) -> AISettings:
        """검증 후 Provider별 필수 조건을 확인한다.

        역할: ``AI_MODEL``은 항상 필수이고, ``openai`` Provider면 HTTPS 주소와 API Key도 필수이며 ``AI_TEMPERATURE``·
            ``AI_REPETITION_PENALTY``·``AI_SYSTEM_PROMPT_PREFIX``는 쓸 수 없다(자체 모델 서버용 설정이므로 기동 시 막는다).
        입력: 없음(검증된 필드).
        출력: 자기 자신.
        예외: ``ValueError`` — 조건 위반(``from_environment``에서 ``ConfigurationError``로 바뀐다).
        """
        if not self.model:
            raise ValueError("AI_MODEL is required")
        if self.provider == AI_PROVIDER_OPENAI:
            if not self.base_url.startswith("https://"):
                raise ValueError("AI_BASE_URL must use HTTPS for the openai provider")
            if self.api_key is None:
                raise ValueError("AI_API_KEY or OPENAI_API_KEY is required")
            for name in ("temperature", "repetition_penalty", "system_prompt_prefix", "no_thinking_operations"):
                if getattr(self, name) not in (None, ""):
                    raise ValueError(f"AI_{name.upper()} is only supported for the openai_compatible provider")
        return self


class RuntimeSettings(_EnvironmentSettings):
    """프로세스 런타임 설정(``AX_G_`` 환경 변수).

    역할: 참조 DB·말뭉치·DUR·임베딩·환자 문맥 저장소 등 런타임 구성 요소를 켜고 끄는 값과 경로를 검증한다.

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
        recommendation_max_possible: 추천 규칙 평가에 보일 ``possible``(조건부)·``reference``(참고) 항목 각각의 최대 개수다(기본 12).
            줄이면 화면의 ``assessment``와 AI 입력이 함께 줄어 자체 모델의 사고 시간이 준다. 값은 벤치마크·게이트로 비교해 정한다.
        clinical_doses_enabled: 말뭉치 용량 규칙(3단계)을 현재 처방·의료진 지정 약에 적용해 체중·신기능 기준 참고 용량,
            용량 조건 규칙, 처방 1일 용량 대조를 낼지 여부다(기본 끔). 규칙은 2단계와 같은 게이트를 지나며 미검수 규칙은
            ``evidence_include_unreviewed`` 가 켜져 있을 때만 쓰고, 항목마다 출처·region(국외 기준 여부)이 붙는다.
        dur_enabled: 심평원 DUR 의약품 목록(PostgreSQL ``dur`` 스키마, ``ax-g-ai-load-dur`` 로 적재)으로 추천 의견·처방 관련
            채팅에서 현재 처방의 병용금기·연령금기·임부금기·노인주의를 점검할지 여부다(공공누리 제1유형 — 출처 표시).
            ``corpus_database_url`` 의 DB를 쓰며, DB나 활성 데이터셋이 없으면 경고 후 점검을 끈다.
        corpus_refresh_seconds: PostgreSQL 활성 말뭉치 릴리스·DUR 데이터셋을 확인하는 주기(초)다. 바뀌면 근거 검색·규칙·DUR을
            한 스냅샷으로 함께 교체한다(요청 경로에서 다시 읽지 않는다).
        patient_context_redis_url: 준비된 환자 문맥을 Redis(``redis://``·``rediss://``)에 둔다. 여러 워커(``WEB_CONCURRENCY``)·인스턴스가
            문맥을 공유할 때 설정한다. 없으면 프로세스 메모리(가장 빠름, 워커 1개)이며, 기동 시 연결할 수 없으면 경고 후 메모리를 쓴다.
            값은 로그에 쓰지 않는다.
        patient_context_store_timeout_seconds: Redis 연결·명령 제한 시간(초)이다.
        dependency_startup_wait_seconds: 기동 시 PostgreSQL·Redis에 연결될 때까지 다시 시도하는 최대 시간(초)이다(기본 0, 한 번만 시도).
            서버 재부팅 때처럼 DB·Redis와 API가 함께 시작되면 연결이 늦어 파일 검색·메모리 저장소로 고정되지 않게 한다.
            시간 안에 연결되지 않으면 지금처럼 경고 후 대체한다.
        attributed_sources_only: 출처를 밝히면 쓸 수 있는 자료만 AI에 쓴다(기본 true). 켜면 (1) 근거·규칙의 모든 출처에
            라이선스(CC0·CC BY·퍼블릭 도메인)와 제목이 있어야 하고, (2) 출처·라이선스가 확인되지 않은 참조 DB
            (``reference_data_db_path``, 약품 효능·주의사항)를 쓰지 않으며, (3) 추천 의견에서 출처 없는 일반 의학 지식 제안을 막는다.
        prompt_data_labels: AI에 보내는 데이터·규칙 문구의 항목 이름 표기. ``korean``(기본)은 한국어 이름('기록상 없음', '판정'),
            ``english``는 영문 키(recorded_none, status)다. 영어로 학습한 자체 모델의 답변 정확도를 비교하는 실험용이며
            API 응답의 키는 어느 쪽이든 바뀌지 않는다.
        model_profile: 모델 성능에 따른 프롬프트·답변 방식. ``compact``(기본, 작은 자체 모델)는 본문의 DUR 판단을 막고 DUR만 묻는
            질문을 서버가 DUR 점검 블록으로 답한다. ``capable``(GPT 등)은 같은 블록에 더해 본문에서 경고의 임상적 의미를 설명할 수 있고
            DUR만 묻는 질문도 모델이 답한다. 데이터 정리·DUR 판정·답변 점검·대체 응답은 같다.
    """

    model_config = SettingsConfigDict(env_prefix="AX_G_")

    reference_data_db_path: FilePath | None = None
    evidence_corpus_dir: DirectoryPath | None = None
    evidence_include_unreviewed: bool = False
    evidence_max_chunks: int = Field(default=3, ge=1, le=5)
    corpus_database_url: SecretStr | None = None
    embedding_url: HttpUrlText | None = None
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    embedding_timeout_seconds: PositiveFloat = 10
    evidence_similarity_floor: float = Field(default=0.6, ge=0, le=1)
    clinical_rules_enabled: bool = True
    site_code_map_path: FilePath | None = None
    recommendation_max_evidence: int = Field(default=10, ge=1, le=20)
    recommendation_max_possible: int = Field(default=12, ge=1, le=50)
    attributed_sources_only: bool = True
    prompt_data_labels: PromptLabels = "korean"
    model_profile: ModelProfile = "compact"
    clinical_doses_enabled: bool = False
    dur_enabled: bool = True
    corpus_refresh_seconds: PositiveFloat = 60
    patient_context_ttl_seconds: PositiveFloat = 12 * 60 * 60
    patient_context_max_entries: PositiveInt = 1000
    patient_context_redis_url: SecretStr | None = None
    patient_context_store_timeout_seconds: PositiveFloat = 1.0
    dependency_startup_wait_seconds: NonNegativeFloat = 0


class LoaderSettings(_EnvironmentSettings):
    """적재 CLI(``ax-g-ai-load-corpus``, ``ax-g-ai-load-dur``) 설정.

    역할: 런타임과 같은 ``AX_G_`` 환경 변수 이름·검증을 쓰되, 적재에 필요한 값만 읽는다
    (``RuntimeSettings``는 참조 DB 경로 등 적재와 무관한 값까지 검증한다).

    Attributes:
        corpus_database_url: 적재 대상 PostgreSQL URL(필수). 값은 출력하지 않는다.
        embedding_url: TEI 임베딩 서버. 없으면 미리 계산한 벡터 파일만 쓴다.
        embedding_timeout_seconds: 문단 묶음 임베딩이라 런타임(질문 1개, 10초)보다 길게 둔다.
        evidence_corpus_dir: 적재할 말뭉치 폴더(``--corpus-dir``가 우선).
        corpus_source_commit: 릴리스에 기록할 말뭉치 빌드 커밋.
        dur_snapshot_dir: 적재할 DUR 스냅샷 폴더(``--snapshot-dir``가 우선).
    """

    model_config = SettingsConfigDict(env_prefix="AX_G_")

    corpus_database_url: SecretStr
    embedding_url: HttpUrlText | None = None
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    embedding_timeout_seconds: PositiveFloat = 60
    evidence_corpus_dir: Path = Path("text_data/corpus")
    corpus_source_commit: str | None = None
    dur_snapshot_dir: Path = Path("data/dur")
