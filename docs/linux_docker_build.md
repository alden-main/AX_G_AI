# Linux 대상 Docker 빌드 및 서버 배포 가이드

이 문서는 로컬 개발 PC에서 Linux용 AX-G AI 이미지를 빌드한 뒤, 이미지 파일을 Linux 서버에 전달하여 실행하는 명령을 정리한다. 로컬에서 수행하는 빌드 명령은 프로젝트 루트(`Dockerfile`이 있는 디렉터리)에서 실행한다.

## 사전 확인

Docker Engine이 실행 중이고 현재 사용자가 Docker 명령을 실행할 수 있는지 확인한다.

```bash
docker --version
docker info
```

`permission denied` 오류가 나면 `sudo docker ...`를 사용하거나, 운영 정책에 따라 사용자를 `docker` 그룹에 추가한 뒤 다시 로그인한다.

## 1. 환경 변수 보관 원칙

운영용 `.env`에는 실제 OpenAI API 키가 들어갈 수 있으므로 배포 대상 서버에서 안전하게 관리한다. `.env`는 Git에 커밋하거나 이미지에 포함하지 않으며, 로컬 PC에서 만든 이미지 tar 파일에도 포함되지 않는다.

## 2. 로컬 PC에서 Linux 대상 이미지 빌드

대상 Linux 서버의 CPU 아키텍처에 맞춰 이미지를 빌드한다. 일반적인 Intel/AMD 64비트 Linux 서버는 `linux/amd64`이며, AWS Graviton 등의 ARM 서버는 `linux/arm64`이다.

아래 명령은 Linux용 이미지를 Docker가 적재할 수 있는 tar 파일로 바로 생성한다. Apple Silicon Mac에서 `linux/amd64` 서버용 이미지를 만들 때도 동일하게 사용한다.

```bash
# x86_64 (Intel/AMD) Linux 서버용
docker buildx build \
  --platform linux/amd64 \
  --tag ax-g-ai:1.0.0 \
  --output type=docker,dest=ax-g-ai_1.0.0_linux_amd64.tar \
  .
```

대상 서버가 ARM64인 경우에는 플랫폼과 tar 파일명을 바꾼다.

```bash
# ARM64 서버 (예: AWS Graviton)
docker buildx build \
  --platform linux/arm64 \
  --tag ax-g-ai:1.0.0 \
  --output type=docker,dest=ax-g-ai_1.0.0_linux_arm64.tar \
  .
```

빌드 산출물의 무결성 검증 파일을 만들고, tar 파일과 함께 Linux 서버에 전달한다.

```bash
sha256sum ax-g-ai_1.0.0_linux_amd64.tar > ax-g-ai_1.0.0_linux_amd64.tar.sha256
```

macOS에서 `sha256sum` 명령이 없으면 다음 명령을 사용한다.

```bash
shasum -a 256 ax-g-ai_1.0.0_linux_amd64.tar > ax-g-ai_1.0.0_linux_amd64.tar.sha256
```

## 3. Linux 서버로 이미지 전달

로컬 PC에서 이미지 tar 파일을 서버의 배포 디렉터리로 전송한다. `deploy`, `SERVER_HOST`, `/opt/ax-g/deploy`는 실제 환경 값으로 바꾼다.

```bash
scp ax-g-ai_1.0.0_linux_amd64.tar ax-g-ai_1.0.0_linux_amd64.tar.sha256 \
  deploy@SERVER_HOST:/opt/ax-g/deploy/
```

`.env`와 API 키는 tar 파일에 포함되지 않는다. 서버에서 별도로 생성·보관하며, 메신저나 소스 저장소에 전달하지 않는다.

## 4. Linux 서버에서 이미지 적재

Linux 서버에 SSH로 접속한 뒤 tar 파일의 SHA-256 값이 로컬에서 확인한 값과 같은지 검증하고 Docker에 적재한다.

```bash
cd /opt/ax-g/deploy
sha256sum --check ax-g-ai_1.0.0_linux_amd64.tar.sha256
docker load --input ax-g-ai_1.0.0_linux_amd64.tar
docker image inspect ax-g-ai:1.0.0 --format '{{.Os}}/{{.Architecture}}'
```

마지막 명령의 결과는 x86_64 서버에서 `linux/amd64`여야 한다. 서버의 CPU 아키텍처와 맞지 않으면 컨테이너가 실행되지 않거나 에뮬레이션으로 매우 느려질 수 있다.

## 5. Linux 서버에서 환경 변수와 컨테이너 준비

서버에서 `.env` 파일을 만들고 권한을 제한한다. 서버에 소스 코드가 없다면 `.env.example` 대신 아래 내용을 직접 생성한다.

```bash
cd /opt/ax-g/deploy
vi .env
chmod 600 .env
```

`.env` 내용:

```dotenv
OPENAI_API_KEY=실제_키
OPENAI_MODEL=gpt-5-mini
OPENAI_TIMEOUT_SECONDS=30
AI_REASONING_EFFORT=low
AX_G_PORT=8100
```

A100 서버에서 vLLM과 함께 실행하는 경우는 아래 vLLM 절을 따른다.

## 6. Linux 서버에서 컨테이너 실행

호스트의 8100 포트로 API를 확인하려면 다음 명령을 사용한다.

```bash
docker run --detach \
  --name ax-g-ai-local \
  --env-file .env \
  --publish 8100:8100 \
  ax-g-ai:1.0.0
```

다른 호스트 포트를 사용하려면 `--publish 호스트포트:8100` 형식으로 바꾼다. 예를 들어 18100 포트로 공개하는 경우:

```bash
docker run --detach --name ax-g-ai-local --env-file .env --publish 18100:8100 ax-g-ai:1.0.0
```

컨테이너 상태와 로그를 확인한다.

```bash
docker ps --filter name=ax-g-ai-local
docker logs --follow ax-g-ai-local
```

## 7. 동작 확인

같은 Linux 서버에서 헬스 체크를 호출한다.

```bash
curl --fail http://127.0.0.1:8100/health
```

정상 응답:

```json
{"status":"ok"}
```

원격 서버에서 접근하는 경우에는 서버 방화벽, 클라우드 보안 그룹 및 리버스 프록시 설정에서 공개한 호스트 포트를 허용해야 한다.

## 8. 재빌드 및 교체

새 버전은 새 태그(예: `1.0.1`)로 로컬에서 빌드·전송·적재한 뒤, Linux 서버에서 기존 컨테이너를 교체한다.

```bash
docker stop ax-g-ai-local
docker rm ax-g-ai-local
docker run --detach --name ax-g-ai-local --env-file .env --publish 8100:8100 ax-g-ai:1.0.1
```

이미지를 적재하기 전에는 컨테이너를 중지·삭제하지 않는다. 새 이미지를 먼저 `docker load`하고 확인하면, 문제 발생 시 기존 컨테이너를 유지할 수 있다.

## 9. 중지 및 정리

컨테이너를 중지하고 제거한다.

```bash
docker stop ax-g-ai-local
docker rm ax-g-ai-local
```

더 이상 필요 없는 이미지까지 제거하려면 이미지 ID를 확인한 후 명시적으로 삭제한다.

```bash
docker image ls ax-g-ai
docker image rm ax-g-ai:1.0.0
```

Bridge와 같은 Docker 네트워크에서 운영할 때는 호스트 포트를 공개하지 말고 `--network`으로 전용 네트워크에 연결한다. 네트워크와 포트는 실제 Bridge 배치에 맞춰 구성한다.


## 참조 데이터와 로컬 Compose

Docker 이미지는 `data/reference_data.sqlite`를 포함하며 기본 경로는 `/app/data/reference_data.sqlite`이다. 빌드 전에 파일이 있어야 한다. 원본이 변경된 경우 DB 생성 방법과 데이터 검증 사항은 [참조 데이터 설계](reference_data_sqlite.md)를 따른다. 별도 DB를 사용할 경우 읽기 전용 볼륨으로 마운트하고 `AX_G_REFERENCE_DATA_DB_PATH`를 설정한다.

소스가 있는 환경에서 `.env`를 준비한 후 로컬 Compose를 실행할 수 있다. `compose.yaml`은 8100 포트를 호스트에 공개하므로 네트워크 접근 범위를 확인한다.

```bash
docker compose up --build -d
docker compose ps
docker compose logs -f ax-g-ai
docker compose down
```

## A100 서버에서 vLLM 실행

```
Bridge ──(bridge-ai, internal)──> AX-G AI ──(ai-inference, internal)──> vLLM(A100)
                                                                       └─(model-download)─> Hugging Face
```

- AX-G AI는 `AI_PROVIDER=openai_compatible`이면 `{AI_BASE_URL}/chat/completions`를 호출한다. 서비스 코드는 OpenAI와 vLLM을 구분하지 않는다.
- EMR 프롬프트는 `ai-inference` 내부 네트워크 밖으로 나가지 않는다. 외부 네트워크에는 vLLM만 연결되며, 모델 다운로드에만 쓴다.

## 서버 사전 조건

```bash
nvidia-smi                                   # A100 인식 확인
docker run --rm --gpus all ubuntu nvidia-smi # NVIDIA Container Toolkit 확인
docker compose version                       # Compose v2
```

## 1. 개발 PC: 이미지 빌드 및 전달

```bash
docker buildx build --platform linux/amd64 --tag ax-g-ai:1.0.0 \
  --output type=docker,dest=ax-g-ai_1.0.0_linux_amd64.tar .
scp ax-g-ai_1.0.0_linux_amd64.tar compose.vllm.yaml deploy@SERVER_HOST:/opt/ax-g/deploy/
```

## 2. A100 서버: 적재와 .env 준비

```bash
cd /opt/ax-g/deploy
docker load --input ax-g-ai_1.0.0_linux_amd64.tar
vi .env && chmod 600 .env
```

`.env` 예시:

```dotenv
AX_G_AI_IMAGE=ax-g-ai:1.0.0
VLLM_MODEL=Qwen/Qwen3-32B-AWQ
AI_MODEL=qwen3-32b-awq
VLLM_MAX_MODEL_LEN=16384
VLLM_GPU_MEMORY_UTILIZATION=0.90
HF_CACHE_DIR=/opt/ax-g/hf-cache
# 비공개(gated) 모델에만 필요
# HF_TOKEN=
AI_TIMEOUT_SECONDS=60
```

`compose.vllm.yaml`이 `AI_PROVIDER`, `AI_BASE_URL`, `AI_MODEL`을 AX-G AI 컨테이너에 설정한다. `openai_compatible`일 때는 `.env`에 `OPENAI_API_KEY`가 남아 있어도 vLLM으로 보내지 않는다.

## 3. 실행

```bash
docker compose -f compose.vllm.yaml up -d
docker compose -f compose.vllm.yaml logs -f vllm   # 모델 다운로드·로딩 진행 확인
```

최초 기동은 모델 다운로드(약 20GB)와 로딩 때문에 수 분 이상 걸린다. vLLM health check가 통과해야 AX-G AI가 시작된다. 모델을 받은 뒤 `.env`에 `HF_HUB_OFFLINE=1`을 설정하면 외부 접속 없이 기동한다.

## 4. 확인

```bash
# vLLM 단독 확인
docker compose -f compose.vllm.yaml exec ax-g-ai python -c \
  "import httpx; print(httpx.get('http://vllm:8000/v1/models').text)"

# AX-G AI 첫 글자 시간 측정 (TTFB = 첫 SSE 이벤트 도착 시각)
curl -sN -X POST http://127.0.0.1:8100/internal/v1/patient-context/stream \
  -H 'Content-Type: application/json' \
  -d '{"patient_id":"pid-4356","emr_payload":{"patient":{"patientId":"pid-4356"},"bloodSugarList":[{"measuredAt":"2026-09-22 10:30:00","glucoseValue":120}]}}' \
  -w '\nTTFB=%{time_starttransfer}s total=%{time_total}s\n'
```

같은 환자에 이어서 `/internal/v1/clinical-chat/stream`을 호출하면 지시문과 EMR 부분은 vLLM prefix cache를 재사용하므로 두 번째 질문부터 첫 글자가 더 빨라진다. vLLM 로그의 `Prefix cache hit rate`로 확인한다.

수동 live 테스트도 같은 설정으로 vLLM에 보낼 수 있다(소스가 있는 환경).

```bash
AI_PROVIDER=openai_compatible AI_BASE_URL=http://127.0.0.1:8000/v1 AI_MODEL=qwen3-32b-awq \
AX_G_RUN_LIVE_SOAP_SCENARIO=true uv run python -m unittest tests.manual.test_live_soap_scenario
```

## 모델 선택 (A100 40GB)

| 모델 | 메모리(가중치) | 비고 |
|---|---|---|
| `Qwen/Qwen3-32B-AWQ` | 약 19GB | 기본값. int4 AWQ, KV cache 여유 충분 |
| `Qwen/Qwen3-14B` (BF16) | 약 28GB | 양자화 없음. `VLLM_MAX_MODEL_LEN=8192` 권장 |
| `Qwen/Qwen3-14B-AWQ` | 약 10GB | 더 빠른 응답, 품질 비교용 |
| EXAONE 계열 | 모델별 상이 | 한국어 강점. 상업 이용 라이선스 확인 필요. `VLLM_REASONING_PARSER` 행 제거 |

- 32B BF16(약 64GB)은 40GB에 들어가지 않는다.
- A100(Ampere)은 FP8 연산을 하드웨어로 지원하지 않으므로 FP8 모델은 속도 이득이 거의 없다. AWQ/GPTQ int4를 권장한다.
- Qwen3는 기본으로 사고(thinking) 출력을 생성해 첫 글자가 늦어진다. AX-G AI는 기본으로 `enable_thinking=false`를 보낸다(`AI_DISABLE_THINKING=true`).
- Qwen3 계열이 아니면 `compose.vllm.yaml`의 `--reasoning-parser` 행을 지운다.
- 모델 교체 후에는 SOAP 형식·근거 반영을 수동 live 테스트와 임상 검토로 다시 확인한다. 프롬프트·모델 변경 시 적용 환경의 평가 및 승인 절차를 따른다.

## 튜닝 항목

| 설정 | 효과 |
|---|---|
| `VLLM_MAX_MODEL_LEN` | 현재 입력은 약 2~3천 토큰이다. 줄일수록 KV cache 동시 처리 여유가 커진다 |
| `VLLM_GPU_MEMORY_UTILIZATION` | 0.90 기본. 같은 GPU에 다른 프로세스가 있으면 낮춘다 |
| `AI_MAX_OUTPUT_TOKENS` | 응답 상한. 긴 응답의 꼬리 지연과 GPU 점유를 제한한다 |
| `VLLM_IMAGE` | 기본 `vllm/vllm-openai:v0.11.0`. 검증된 최신 버전으로 고정해 교체한다 |

## OpenAI로 되돌리기

`compose.vllm.yaml` 대신 기존 방식(`docker run --env-file .env ...`)으로 실행하고 `.env`에 `AI_PROVIDER=openai`(또는 생략)와 `OPENAI_API_KEY`를 둔다. OpenAI의 첫 글자 시간은 `AI_REASONING_EFFORT=low`(또는 `minimal`)로 줄인다.
