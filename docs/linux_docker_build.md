# Linux 대상 Docker 이미지 빌드 및 서버 실행

개발 PC에서 배포 압축파일을 만들어 전달하고, 소스 코드가 없는 Linux 서버에서 Docker Compose로 실행하는 절차다. 기본 방법은 아래의 **배포 스크립트**(`scripts/build-deployment.sh` → 서버의 `start-deployment.sh`)이며, 스크립트를 쓸 수 없을 때만 [수동 절차](#수동-절차-스크립트를-쓰지-않을-때)를 따른다. 명령은 프로젝트 루트에서 실행한다. 이미지 태그 `1.0.0`, 서버 주소와 경로는 배포 환경에 맞게 변경한다.

## 구성

- `compose.yaml`: OpenAI 등 외부 AI 제공자와 함께 API, PostgreSQL/pgvector, TEI 임베딩 서버(KURE-v1) 및 말뭉치·DUR 적재기를 실행한다.
- `compose.vllm.yaml`: `compose.yaml`에 덧붙여 A100 서버의 vLLM으로 실행한다(외부 AI 제공자 네트워크를 빼고 API 포트를 `127.0.0.1`에만 연다).
- `compose.sglang.yaml`: `compose.yaml`에 덧붙여 SGLang으로 실행한다. vLLM이 지원하지 않는 `learning-unit/L1-16B-A3B`(GravityMoE)용이며, 이미지는 `docker/sglang-gravity`(공식 SGLang v0.5.19 + GravityMoE 패치)로 만든다.
- `compose.redis.yaml`: 위 구성 중 하나에 추가해 Redis 환자 문맥 저장소와 다중 워커를 사용한다.
- `compose.observability.yaml`: 위 구성 중 하나에 추가해 AI 호출 추적 화면(Arize Phoenix)을 띄운다(6절).

자체 모델 구성은 `compose.yaml` 위에 덧붙이는 파일이므로 항상 `-f compose.yaml -f compose.<vllm|sglang>.yaml` 순서로 지정한다. 덧붙이는 파일의 `!override`는 Docker Compose 2.24.4 이상이 필요하다.

`.env`에는 API 키와 비밀번호만 두며 서버에서 별도로 관리하고 이미지나 저장소에 포함하지 않는다. 나머지 설정은 Compose 파일과 코드 기본값에 있다. 코드 기본값과 다르게 운영하는 애플리케이션 설정(`AI_REASONING_EFFORT=low`, `AX_G_EVIDENCE_INCLUDE_UNREVIEWED=true`, `AX_G_CLINICAL_DOSES_ENABLED=true`)과 서버 자원·이미지·포트 값은 Compose 파일에 `${이름:-기본값}`으로 적혀 있으며, 서버별로 바꿀 때는 `.env`가 아니라 Compose 파일의 기본값을 고친다. `.env.example`에는 비밀값 이름만 있고, `AX_G_*`·`AI_*` 설정 설명은 `src/ax_g_ai/config.py`에 있다. `.dockerignore`는 `.env`와 이미지 tar를 빌드 컨텍스트에서 제외한다.

## 배포 스크립트로 빌드하고 실행하기

### 개발 PC 준비

- Docker Engine(또는 Docker Desktop)과 Buildx. Apple Silicon에서도 `linux/amd64` 이미지를 빌드할 수 있다.
- 외부 이미지를 내려받을 인터넷 연결(`pgvector/pgvector:0.8.7-pg16`, `ghcr.io/huggingface/text-embeddings-inference:cpu-1.9.4`).
- 임베딩 모델 캐시 `hf-cache/models--nlpai-lab--KURE-v1`(약 2.1GB). 없으면 스크립트가 중단되고 받는 명령을 안내한다.

```bash
uvx --from huggingface_hub hf download nlpai-lab/KURE-v1 --cache-dir hf-cache
```

- 이미지에는 `data/dur` 스냅샷과 `text_data/corpus` 및 사전 계산 임베딩이 들어간다. 임베딩이 현재 말뭉치를 모두 덮지 못하면 빌드가 실패한다. 말뭉치를 바꿨다면 `text_data/tools/build_corpus.py`와 `text_data/tools/build_embeddings.py`를 순서대로 실행해 갱신한다.

### 빌드

```bash
./scripts/build-deployment.sh
```

현재 소스로 API 이미지를 빌드하고 아래 파일을 `ax-g-ai-deploy-amd64.tar.gz`(약 2.3GB)와 체크섬 파일 `ax-g-ai-deploy-amd64.tar.gz.sha256`로 만든다. 빌드는 수 분이 걸린다.

| 압축 안의 파일 | 내용 |
|---|---|
| `image.tar`, `image-tag.txt` | API 이미지(`ax-g-ai:1.0.0`)와 태그 |
| `postgres-image.tar`, `postgres-image.txt` | PostgreSQL(pgvector) 이미지와 이름 |
| `embedding-image.tar`, `embedding-image.txt` | TEI 임베딩 서버 이미지와 이름 |
| `hf-cache/` | KURE-v1 임베딩 모델 |
| `compose.yaml`, `compose.vllm.yaml`, `compose.sglang.yaml`, `compose.redis.yaml`, `compose.observability.yaml` | Compose 파일 |
| `phoenix-image.tar`, `phoenix-image.txt` | 추적 화면(Arize Phoenix) 이미지와 이름(`AX_G_BUNDLE_PHOENIX=0`이면 빠짐) |
| `docker/sglang-gravity/` | SGLang 이미지 빌드 파일(Dockerfile, GravityMoE 패치) |
| `sglang-image.tar`, `sglang-image.txt` | SGLang 이미지와 이름(`AX_G_BUNDLE_SGLANG=1`일 때만) |
| `.env.example` | 비밀값 예제(값 없음) |
| `start-deployment.sh` | 서버 기동 스크립트 |

빌드 옵션은 환경 변수로 지정한다.

| 변수 | 기본값 | 용도 |
|---|---|---|
| `AX_G_PLATFORM` | `linux/amd64` | ARM 서버는 `linux/arm64`(파일명 `ax-g-ai-deploy-arm64.tar.gz`, TEI는 `cpu-arm64-1.9.4` 이미지) |
| `AX_G_IMAGE_TAG` | `ax-g-ai:1.0.0` | API 이미지 태그. 새 버전 배포 시 올린다 |
| `AX_G_PACKAGE_PATH` | `ax-g-ai-deploy-<arch>.tar.gz` | 출력 경로 |
| `AX_G_BUNDLE_POSTGRES` / `AX_G_BUNDLE_EMBEDDING` | `1` | `0`이면 PostgreSQL 이미지 / TEI 이미지와 모델을 넣지 않는다(서버가 최초 기동 때 내려받는다) |
| `AX_G_POSTGRES_IMAGE` / `AX_G_EMBEDDING_IMAGE` | 위 이미지 | 넣을 외부 이미지 |
| `AX_G_EMBEDDING_MODEL_CACHE` | `hf-cache/models--nlpai-lab--KURE-v1` | 넣을 모델 캐시 디렉터리 |
| `AX_G_BUNDLE_SGLANG` | `0` | `1`이면 SGLang 이미지를 빌드해 넣는다(수십 GB). 폐쇄망 GPU 서버에서 SGLang을 쓸 때 필요하다 |
| `AX_G_SGLANG_IMAGE` | `ax-g-sglang-gravity:0.5.19` | 넣을 SGLang 이미지 태그 |
| `AX_G_BUNDLE_PHOENIX` | `1` | `0`이면 추적 화면 이미지(약 330MB)를 넣지 않는다 |
| `AX_G_PHOENIX_IMAGE` | `arizephoenix/phoenix:version-21.3.0` | 넣을 Phoenix 이미지 |

```bash
# 예: ARM 서버용, 새 태그
AX_G_PLATFORM=linux/arm64 AX_G_IMAGE_TAG=ax-g-ai:1.0.1 ./scripts/build-deployment.sh
```

전달 전에 압축 파일을 검사할 수 있다. 체크섬, 필수 파일, 함께 넣은 이미지·모델의 짝을 확인하고, API 이미지를 불러와
단독으로 띄워 `/health`·`/ready`(AI 설정 인식, 파일 말뭉치의 근거·규칙 로드)를 확인한 뒤 컨테이너를 지운다. AI는 호출하지 않는다.

```bash
./scripts/verify-deployment.sh ax-g-ai-deploy-amd64.tar.gz
```

GitHub Actions(`.github/workflows/deployment-bundle.yml`)는 `main` 푸시·PR(이미지에 들어가는 파일이 바뀐 경우)과 수동 실행 때
amd64 압축 파일을 같은 두 스크립트로 만들고 검사한다. 결과물은 크기 때문에 올리지 않으며(실행 요약에 크기·SHA-256만 남김),
실제 전달 파일은 위처럼 로컬에서 만든다.

### 서버 준비

- Linux 서버(`uname -m`이 `x86_64`면 amd64 압축, `aarch64`면 arm64 압축)에 Docker Engine과 Compose v2(`docker compose`).
- 메모리: 임베딩 서버가 최대 6GB를 쓴다(4GB 안팎에서는 모델 적재 중 OOM). 디스크: 압축파일과 압축 해제분, 적재한 이미지를 합쳐 10GB 이상.
- API 포트는 호스트 8500(`AX_G_PORT`, 컨테이너 안은 8100). OpenAI 구성은 OpenAI API로 나가는 연결이 필요하다. PostgreSQL·임베딩 이미지와 모델은 압축에 들어 있어 내려받지 않는다.
- 재부팅 후 자동 기동: `sudo systemctl enable docker`. API·DB·임베딩·Redis는 `restart: unless-stopped`다.

### 전달과 실행

```bash
scp ax-g-ai-deploy-amd64.tar.gz ax-g-ai-deploy-amd64.tar.gz.sha256 deploy@SERVER_HOST:/opt/ax-g/deploy/
```

서버에서 검증하고 압축을 푼 뒤 실행한다. 첫 실행은 `.env.example`을 `.env`로 복사하고(권한 600) 종료한다. `OPENAI_API_KEY`와 `CORPUS_DB_PASSWORD`를 입력한 후 다시 실행하면 이미지 세 개를 적재하고 Compose 서비스를 시작한다. 비밀번호는 DB 접속 URL에 들어가므로 영숫자만 쓴다(예: `openssl rand -hex 24`).

```bash
cd /opt/ax-g/deploy
sha256sum --check ax-g-ai-deploy-amd64.tar.gz.sha256
tar -xzf ax-g-ai-deploy-amd64.tar.gz
cd ax-g-ai-deploy
./start-deployment.sh          # .env 생성 후 종료
vi .env                        # OPENAI_API_KEY, CORPUS_DB_PASSWORD
./start-deployment.sh          # 이미지 적재 + 기동
```

| 실행 형태 | 명령 | 추가로 필요한 것 |
|---|---|---|
| OpenAI(기본) | `./start-deployment.sh` | — |
| OpenAI + Redis | `./start-deployment.sh openai --redis` | `.env`에 `REDIS_PASSWORD`, 서버에서 Redis 이미지 |
| 자체 모델(권장) | `./start-deployment.sh local` | GPU 서버, 추론 서버 이미지·모델(기본 L1·SGLang)([자체 모델 구성](#4-a100과-자체-모델-구성vllm-sglang)) |
| vLLM 지정 | `./start-deployment.sh vllm` | 위와 같음 |
| SGLang 지정 | `./start-deployment.sh sglang` | 위와 같음 |
| 위 구성 + Redis | 위 명령 뒤에 `--redis` | `.env`에 `REDIS_PASSWORD`, 서버에서 Redis 이미지 |

`local`은 `LLM_MODEL`(Hugging Face 모델 ID, 기본 `learning-unit/L1-16B-A3B`)을 보고 추론 서버를 고른다. `learning-unit/L1-*`는 SGLang, 그 밖의 모델은 vLLM이다. 다른 모델은 셸에서 `LLM_MODEL=Qwen/Qwen3-32B-AWQ ./start-deployment.sh local`처럼 지정한다. SGLang으로 띄울 모델 목록은 `start-deployment.sh`의 `SGLANG_MODEL_PATTERNS`에 있다. `vllm`·`sglang`은 엔진을 직접 지정하며, 모델과 맞지 않으면 경고한다.

스크립트는 Compose 파일 조합을 검증한 뒤 이미지를 적재하고 `up -d --no-build`로 기동한다. 기존 `.env`는 덮어쓰지 않는다. 압축 안의 이미지 이름(`image-tag.txt`, `*-image.txt`)을 Compose에 넘기고, 임베딩 서버는 압축 안의 `hf-cache/`를 오프라인(`HF_HUB_OFFLINE=1`)으로 쓴다. `.env`에 `EMBEDDING_IMAGE`, `EMBEDDING_CACHE_DIR`, `EMBEDDING_HF_HUB_OFFLINE`, `POSTGRES_IMAGE`를 직접 적으면 그 값이 우선한다.

### 기동 확인

```bash
curl --fail http://127.0.0.1:8500/health          # {"status":"ok"}
curl http://127.0.0.1:8500/ready                   # 준비되면 200 "status":"ready", 아니면 503 (corpus.backend가 postgres인지 확인)
docker compose -f compose.yaml ps -a               # 적재기 두 개는 Exited (0)이 정상, ax-g-ai는 (healthy)
docker compose -f compose.yaml logs --tail=100 ax-g-ai corpus-loader dur-loader
```

로그에서 `clinical_evidence_loaded backend=postgres`, `dur_loaded`, `clinical_rules_loaded`를 확인한다. `fallback=files`나 `dur_unavailable`이 보이면 DB 적재를 확인한다. `/ready`는 AI 설정·문맥 저장소·말뭉치 DB 상태를 보여 주지만 AI 제공자 연결까지 확인하지는 않으므로 실제 질문을 한 번 보낸다. Redis·vLLM·SGLang을 쓰면 `-f` 뒤에 실행한 파일 조합을 같은 순서로 지정한다(스크립트가 마지막에 해당 로그 명령을 출력한다).

재부팅 때는 서비스가 동시에 시작되므로 API가 PostgreSQL·Redis 연결을 최대 `AX_G_DEPENDENCY_STARTUP_WAIT_SECONDS`(기본 120초) 동안 다시 시도하고, 그 안에 연결되지 않으면 경고 후 파일 검색·메모리 저장소로 대체한다.

### 새 버전으로 갱신

개발 PC에서 새 태그로 다시 빌드해 전달하고, 서버의 **같은 위치**에 압축을 풀어 덮어쓴 뒤 스크립트를 다시 실행한다. 압축에는 `.env`가 없으므로 기존 `.env`가 유지된다. Compose 프로젝트 이름이 디렉터리 이름(`ax-g-ai-deploy`)이므로 DB 볼륨(`ax-g-ai-deploy_corpus-db`)도 그대로 쓰며, 적재기가 새 이미지의 말뭉치·DUR을 새 릴리스로 적재·활성화한다.

```bash
# 개발 PC
AX_G_IMAGE_TAG=ax-g-ai:1.0.1 ./scripts/build-deployment.sh
# 서버
cd /opt/ax-g/deploy
sha256sum --check ax-g-ai-deploy-amd64.tar.gz.sha256
tar -xzf ax-g-ai-deploy-amd64.tar.gz
cd ax-g-ai-deploy && ./start-deployment.sh
```

압축 해제 후에는 이미지 tar와 압축파일을 모두 보관할 디스크 공간이 필요하다. 적재가 끝난 tar 파일과 이전 이미지는 복구 정책에 따라 정리한다.

## 수동 절차 (스크립트를 쓰지 않을 때)

아래는 API 이미지만 tar로 만들어 전달하고 서버에서 Compose를 직접 실행하는 절차다. 이 경우 서버가 PostgreSQL·TEI 이미지와 KURE-v1 모델(약 2.3GB)을 최초 기동 때 내려받으므로 인터넷 연결이 필요하다. 폐쇄망이면 배포 스크립트를 쓴다.

### 1. 이미지 빌드와 전달

예시는 x86_64 서버용이다.

```bash
docker buildx build --platform linux/amd64 \
  --tag ax-g-ai:1.0.0 \
  --output type=docker,dest=ax-g-ai_1.0.0_linux_amd64.tar .

# macOS; Linux에서는 sha256sum을 사용한다.
shasum -a 256 ax-g-ai_1.0.0_linux_amd64.tar > ax-g-ai_1.0.0_linux_amd64.tar.sha256

scp ax-g-ai_1.0.0_linux_amd64.tar ax-g-ai_1.0.0_linux_amd64.tar.sha256 \
  compose.yaml compose.vllm.yaml compose.sglang.yaml compose.redis.yaml compose.observability.yaml .env.example \
  deploy@SERVER_HOST:/opt/ax-g/deploy/
```

SGLang 구성을 쓰면 `docker/sglang-gravity/` 디렉터리도 같은 상대 경로로 전송한다. ARM 서버는 플랫폼과 파일명을 `linux/arm64`, `ax-g-ai_1.0.0_linux_arm64.tar`로 바꾼다. 서버의 배포 디렉터리는 미리 생성한다. `.env.example`에는 비밀 값이 없으며 실제 `.env`는 전송하지 않는다.

임베딩 검사를 의도적으로 건너뛸 때만 `--build-arg ALLOW_STALE_EMBEDDINGS=1`을 지정한다(말뭉치·임베딩 갱신은 [개발 PC 준비](#개발-pc-준비) 참고).

### 2. 서버에서 이미지와 환경 설정 준비

```bash
cd /opt/ax-g/deploy
sha256sum --check ax-g-ai_1.0.0_linux_amd64.tar.sha256
docker load --input ax-g-ai_1.0.0_linux_amd64.tar
docker image inspect ax-g-ai:1.0.0 --format '{{.Os}}/{{.Architecture}}'
cp .env.example .env
chmod 600 .env
vi .env
```

이미지 아키텍처가 서버와 일치하는지 확인한다. `.env`에는 비밀값만 설정한다. 비밀번호는 URL에도 삽입되므로 아래처럼 영숫자로 된 임의 값을 사용한다(예: `openssl rand -hex 24`). 적재한 이미지 태그는 비밀이 아니므로 Compose를 실행하는 셸에서 `export AX_G_AI_IMAGE=ax-g-ai:1.0.0`으로 지정한다.

```dotenv
CORPUS_DB_PASSWORD=생성한_비밀번호
OPENAI_API_KEY=실제_API_키
```

Compose 파일에는 `AX_G_EVIDENCE_INCLUDE_UNREVIEWED=true`와 `AX_G_CLINICAL_DOSES_ENABLED=true`가 설정되어 있다(Provider는 코드 기본값 `openai`, 모델 `gpt-5-mini`). 현재 말뭉치와 용량 규칙은 미검수이므로 배포 전에 해당 설정을 검토한다. 미검수 사용을 끄면 현재 말뭉치의 근거 인용 및 추천 진단·검사가 비활성화된다. 끄려면 `compose.yaml`의 값을 `false`로 고친다. 나머지 설정은 `src/ax_g_ai/config.py`를 참고한다.

PostgreSQL의 사용자·DB 이름과 비밀번호는 기존 `corpus-db` 볼륨의 초기 설정과 일치해야 한다. `.env`만 바꿔도 기존 DB 계정의 비밀번호가 자동 변경되지는 않는다. ARM 서버(Apple Silicon 개발 PC 포함)에서 Compose를 직접 실행할 때는 셸에서 `EMBEDDING_IMAGE=ghcr.io/huggingface/text-embeddings-inference:cpu-arm64-1.9.4`를 지정한다(배포 스크립트는 자동 설정). amd64 이미지를 에뮬레이션으로 돌리면 메모리 한도를 넘겨 재시작을 반복한다.

### 3. OpenAI 구성 실행

```bash
docker compose -f compose.yaml up -d --no-build
docker compose -f compose.yaml ps -a
docker compose -f compose.yaml logs corpus-loader dur-loader
docker compose -f compose.yaml logs --tail=100 ax-g-ai
curl --fail http://127.0.0.1:8500/health
```

`--no-build`는 서버에 소스가 없어도 적재한 이미지를 사용하도록 한다. 최초 실행 시 PostgreSQL과 임베딩 서버가 정상 상태가 된 후 적재기 두 개가 성공 종료되어야 API가 시작된다. 적재기는 성공 시 종료 상태로 남는 것이 정상이다. `/health`의 정상 응답은 `{"status":"ok"}`다. 이 응답만으로 AI 제공자, Redis 또는 데이터 기능까지 정상이라고 판단할 수 없으므로 로그와 실제 연동 요청도 확인한다. API 설정 오류나 데이터 적재 실패가 있으면 해당 서비스 로그를 확인한다.

`compose.yaml`의 포트 매핑은 `${AX_G_BIND:-0.0.0.0}:${AX_G_PORT:-8500}:8100`으로 기본은 모든 호스트 인터페이스의 8500에 공개된다. 서버 내부에서만 접근하려면 Compose 파일의 `AX_G_BIND` 기본값을 `127.0.0.1`로 고친다. Bridge를 같은 `bridge-ai` 네트워크에 연결할 때는 포트 매핑과 네트워크 접근을 배포 환경에 맞게 구성한다.

소스가 있는 개발 환경에서는 `.env`를 준비한 뒤 `docker compose up -d --build`를 사용할 수 있다.

### 4. A100과 자체 모델 구성(vLLM, SGLang)

배포 스크립트(`./start-deployment.sh local`)를 쓸 때도 아래 GPU 확인과 추론 서버 이미지·모델 준비가 필요하다. NVIDIA 드라이버, NVIDIA Container Toolkit과 GPU 접근, Compose 버전(2.24.4 이상)을 확인한다.

```bash
nvidia-smi
docker run --rm --gpus all ubuntu nvidia-smi
docker compose version
```

서버 `.env`에 `CORPUS_DB_PASSWORD`를 설정한다(수동 절차면 `AX_G_AI_IMAGE`는 셸에서 지정). 모델·GPU 번호·문맥 길이는 Compose 파일 기본값을 쓴다. 자체 모델 구성은 API 컨테이너의 `AI_PROVIDER`, `AI_BASE_URL`, `AI_MODEL`을 직접 설정하므로 `.env`의 동일 항목보다 우선한다. `OPENAI_API_KEY`는 쓰지 않으며, API 컨테이너는 외부 AI 제공자 네트워크에 연결되지 않는다.

| 모델(`LLM_MODEL`) | 추론 서버 | Compose 파일 | 비고 |
|---|---|---|---|
| `Qwen/Qwen3-32B-AWQ`(기본) 등 | vLLM `vllm/vllm-openai:v0.11.0` | `compose.vllm.yaml` | A100 40GB 대상 |
| `learning-unit/L1-16B-A3B` | SGLang `ax-g-sglang-gravity:0.5.19` | `compose.sglang.yaml` | GravityMoE, bf16 가중치 약 32.5GB, 최대 문맥 32768 |

`AI_MODEL`(응답 모델 이름)을 비우면 배포 스크립트가 모델 ID의 끝부분을 소문자로 쓴다(`qwen3-32b-awq`, `l1-16b-a3b`). Compose를 직접 실행할 때는 `LLM_MODEL`과 `AI_MODEL`을 함께 적는다. 예전 이름 `VLLM_MODEL`도 vLLM 모델로 읽는다.

모델·GPU 번호·문맥 길이·바인딩은 Compose 파일에 기본값으로 있으며 `.env`에 적지 않는다. 서버가 바뀌면 Compose 파일을 고친다.

| 값 | 기본값 | 위치 |
|---|---|---|
| 모델 | `learning-unit/L1-16B-A3B`(SGLang), `Qwen/Qwen3-32B-AWQ`(vLLM) | `compose.sglang.yaml`, `compose.vllm.yaml`의 `LLM_MODEL` |
| GPU 번호(nvidia-smi 기준, 1장) | `1` | 같은 파일의 `GPU_DEVICE` |
| 문맥 길이 | SGLang `32768`, vLLM `16384` | `SGLANG_CONTEXT_LENGTH`, `VLLM_MAX_MODEL_LEN` |
| GPU 메모리 비율 | SGLang `0.88`, vLLM `0.90` | `SGLANG_MEM_FRACTION_STATIC`, `VLLM_GPU_MEMORY_UTILIZATION` |
| API 바인딩·포트 | `0.0.0.0:8500` | `AX_G_BIND`, `AX_G_PORT` |
| 모델 캐시 | `./hf-cache` | `HF_CACHE_DIR` |

gated 모델이면 `.env`에 `HF_TOKEN`을 설정한다.

```bash
# ENGINE은 vllm 또는 sglang
docker compose -f compose.yaml -f compose.ENGINE.yaml up -d --no-build
docker compose -f compose.yaml -f compose.ENGINE.yaml ps -a
docker compose -f compose.yaml -f compose.ENGINE.yaml logs --tail=100 ENGINE corpus-loader dur-loader ax-g-ai
curl --fail http://127.0.0.1:8500/health
```

추론 서버는 최초 기동 시 모델을 다운로드하므로 시간이 걸린다(수동 절차면 TEI 모델도). 모델이 캐시에 준비된 뒤 `HF_HUB_OFFLINE=1`로 실행할 수 있다. 임베딩 서버의 오프라인 여부는 `EMBEDDING_HF_HUB_OFFLINE`으로 따로 정해지며, 배포 스크립트는 압축 안의 모델로 오프라인 기동한다. 다른 모델 계열을 쓰면 reasoning parser(`VLLM_REASONING_PARSER`, `SGLANG_REASONING_PARSER`)와 모델 호환성을 함께 확인한다. API 포트는 기본적으로 `0.0.0.0:${AX_G_PORT:-8500}`에 바인딩된다. 인증이 없으므로 서버 방화벽으로 접근을 제한한다.

#### SGLang 이미지와 L1 모델

- **이미지**: `learning-unit/L1-16B-A3B`는 공식 SGLang에 없는 `GravityMoEForCausalLM` 구조다. 모델 카드가 안내하는 `trillion-labs/sglang-gravity`는 SGLang v0.5.19에 커밋 하나(`cac38fc`, 모델 등록)를 더한 fork이므로, `docker/sglang-gravity/Dockerfile`은 공식 `lmsysorg/sglang:v0.5.19-cu129-runtime`에 그 패치(`gravity.patch`)만 적용한다. CUDA 12 이미지이므로 드라이버는 CUDA 12.9 이상을 지원해야 한다. 모델 카드 예시의 `lmsysorg/sglang:latest`는 이 모델을 등록하지 않아 쓸 수 없다.
- **빌드**: 서버에 이미지가 없으면 `start-deployment.sh`가 처음 한 번 `docker build --tag ax-g-sglang-gravity:0.5.19 docker/sglang-gravity`로 빌드한다(베이스 이미지 다운로드 필요). 폐쇄망이면 개발 PC에서 `AX_G_BUNDLE_SGLANG=1 ./scripts/build-deployment.sh`로 이미지를 압축에 넣는다. 수동으로는 `docker build -t ax-g-sglang-gravity:0.5.19 docker/sglang-gravity`.
- **실행 옵션**: 모델 카드 권장값(`--trust-remote-code`, `--dtype=bfloat16`, `--attention-backend=triton`, `--moe-runner-backend=triton`)을 쓴다. `--mem-fraction-static=0.88`은 A100 40GB 기준값으로, 가중치를 뺀 나머지가 KV 캐시다(MLA라 토큰당 약 32KB). 기동 중 OOM이면 `SGLANG_MEM_FRACTION_STATIC`이나 `SGLANG_CONTEXT_LENGTH`를 낮춘다.
- **사고 모드·샘플링**: L1은 `<think>` 사고 모드로 학습·평가된 모델이다(모델 카드). 사고를 끄면(`enable_thinking=false`) 같은 입력에도 답이 크게 흔들리고 입력의 DUR 경고와 반대로 답하는 사례가 있었으므로, `compose.sglang.yaml`은 `AI_DISABLE_THINKING=false`(코드 기본값은 끔), `AI_TEMPERATURE=0.3`(모델 `generation_config`에 온도가 없어 지정하지 않으면 1.0), 사고 토큰을 포함한 `AI_MAX_OUTPUT_TOKENS=6144`, `AI_TIMEOUT_SECONDS=120`, `--context-length=32768`을 쓴다. `--reasoning-parser=glm45`가 사고 내용을 `reasoning_content`로 분리하므로 답변에는 남지 않는다. 사고 시간만큼 첫 글자가 늦어진다.
- **L1-30B-A5B**: `LLM_MODEL=learning-unit/L1-30B-A5B ./start-deployment.sh local`로 띄운다(같은 GravityMoE 구조라 SGLang). 스크립트가 모델 카드 권장값을 채운다: 지시문 앞 신원 문장(`AI_SYSTEM_PROMPT_PREFIX="You are Chain-of-Evidence, a medical AI assistant developed by Lunit."`), `AI_REPETITION_PENALTY=1.05`(OpenAI 표준 파라미터가 아니어서 `extra_body`로 보낸다), `AI_TEMPERATURE=0`. 모델 프로필도 `capable`로 둔다(본문에서 DUR 경고의 임상적 의미와, 근거를 붙인 검토 선택지를 쓸 수 있다). 첫 글자가 늦지 않도록 환자 요약과 DUR 해설은 사고 없이 보낸다(`AI_NO_THINKING_OPERATIONS=one_shot_patient_summary_stream,dur_commentary_stream`; 다른 작업은 사고 사용). 셸에서 지정한 값이 있으면 그 값을 쓴다. `docker compose`로 직접 띄우면 이 값이 채워지지 않으므로 셸에서 지정한다. 16B는 셋 다 비어 있어 지금처럼 보낸다(온도 0.3). 가중치가 약 59GB(bf16)라 A100 80GB 1장에서 KV 캐시가 약 11GB(약 18만 토큰)이고, 기동 OOM이면 `SGLANG_MEM_FRACTION_STATIC=0.85`로 낮춘다. 모델 파일 약 60GB를 `hf-cache`에 받아 두어야 한다.
- **답변 품질 모니터링(로그)**: 답변마다 `answer_verified operation=… sentences=… dropped=… regenerated=… problems=…`가 남는다(본문 없음). `dropped`는 자료와 대조되지 않아 뺀 문장 수, `problems`는 위반 종류다. `provider_usage operation=… input_tokens=… output_tokens=… reasoning_tokens=…`는 작업별 토큰 사용량이다(사고 토큰은 Provider가 알려 줄 때만). 그 밖에 `provider_answer_rejected`(다시 요청), `provider_answer_fallback`(고정 안내로 답함), `dur_commentary_skipped`(DUR 해설 생략). 집계 예: `docker compose ... logs --since 24h ax-g-ai | grep -o 'answer_verified operation=[a-z_]* .*problems=[a-z_,-]*' | awk '{print $2, $NF}' | sort | uniq -c`. 위반이 꾸준히 늘면 모델·프롬프트 변경을 의심하고 벤치마크 출시 게이트로 확인한다.
- **모델 프로필**: `AX_G_MODEL_PROFILE`이 모델 성능에 따른 프롬프트·답변 방식을 정한다. `compose.yaml`(OpenAI) 기본은 `capable`, `compose.sglang.yaml`·`compose.vllm.yaml`(자체 모델) 기본은 `compact`다. `compact`는 본문의 DUR 판단을 막고 DUR만 묻는 질문을 서버가 DUR 점검 블록으로 답한다. `capable`은 같은 블록에 더해 본문에서 경고의 임상적 의미를 설명할 수 있고, DUR만 묻는 질문도 모델이 답한다. 데이터 정리(처방일 `기록 없음` 표시 등)·DUR 판정 블록·답변 점검·대체 응답은 둘이 같다. 추적의 `prompt_version`에 `-capable`이 붙는다. OpenAI 사고량(`AI_REASONING_EFFORT`) 기본은 `low`다(사고 토큰이 답변보다 먼저 나오므로 `medium`이면 첫 글자가 약 5초에서 약 20초로 늦어졌다).
- **프롬프트 데이터 표기**: AI에 보내는 EMR·규칙 평가 데이터의 항목 이름은 `compose.yaml`의 `AX_G_PROMPT_DATA_LABELS`로 정한다. `korean`(기본)은 한국어 이름(`기록상 없음`, `판정`), `english`는 영문 키(`recorded_none`, `status`)다. 영어로 학습한 L1이 어느 쪽을 더 정확히 읽는지 비교하는 실험용 설정이며 API 응답의 키는 바뀌지 않는다. 바꾼 뒤 `docker compose ... up -d ax-g-ai`로 API만 다시 시작한다(추적의 `prompt_version`에 `-en`이 붙는다).
- **기동 후 확인**: 요약 요청이 JSON 스키마대로 나오는지, 스트리밍이 `finish_reason: stop`으로 끝나는지, 응답에 `<think>`가 섞이지 않는지 실제 질문으로 확인한다. 3B 활성 모델이므로 한국어 임상 응답 품질은 기존 모델과 통합테스트 시나리오로 비교한 뒤 운영에 쓴다.

### 5. 선택 사항: Redis와 다중 워커

실제 사용하는 파일 조합 마지막에 `compose.redis.yaml`을 추가한다. `.env`에 `REDIS_PASSWORD`를 설정한다. URL에 삽입되므로 영숫자 비밀번호를 사용한다. 기본 워커 수는 2개이며 `AX_G_WORKERS`로 조절한다.

```bash
# OpenAI 구성 예시. 자체 모델이면 compose.yaml 뒤에 compose.vllm.yaml 또는 compose.sglang.yaml을 넣는다.
docker compose -f compose.yaml -f compose.redis.yaml up -d --no-build
docker compose -f compose.yaml -f compose.redis.yaml ps -a
docker compose -f compose.yaml -f compose.redis.yaml logs --tail=100 redis ax-g-ai
```

이후 상태 확인, 갱신 및 종료 명령에도 같은 `-f` 파일 조합을 사용한다. Redis는 내부 네트워크에서만 사용하고 디스크에 저장하지 않는다. 재시작 시 환자 문맥을 다시 준비해야 한다. Redis 없이 기본 메모리 저장소를 쓰면 문맥은 프로세스 사이에 공유되지 않으므로 단일 워커로 운영한다. 기동 시 Redis 연결에 실패하면 메모리 저장소로 대체되므로 로그에서 `patient_context_store backend=redis`를 확인한다.

### 6. 선택 사항: AI 호출 추적 화면(Arize Phoenix)

AI 호출마다 작업 이름(`invoke_agent clinical_recommendation` 등), 모델, 토큰 수, 소요 시간, 종료 사유, 프롬프트 판(`metadata.prompt_version`)을 웹 화면에서 본다. 배포 스크립트는 `--observability`로 켠다(`./start-deployment.sh local --observability`, `--redis`와 함께 써도 된다). 수동이면 파일 조합 마지막에 `compose.observability.yaml`을 추가한다.

- `.env`: `PHOENIX_SECRET`, `PHOENIX_ADMIN_SECRET`(각 32자 이상, 숫자·소문자 포함, 서로 다르게. 예: `openssl rand -hex 24`), `PHOENIX_ADMIN_PASSWORD`(화면 로그인 `admin@localhost`의 최초 비밀번호).
- 화면: `http://<서버>:6006`. 기본은 서버 안(`127.0.0.1`)에서만 열리며, 사내 PC에서 보려면 `.env`에 `PHOENIX_BIND=0.0.0.0`을 둔다(로그인 필요). 추적 수신(OTLP)도 `PHOENIX_ADMIN_SECRET` 토큰이 있어야 받는다.
- 저장: 말뭉치 PostgreSQL의 `phoenix` 스키마(최초 기동 때 Phoenix가 만든다). 보관 기간 기본 30일(`PHOENIX_RETENTION_DAYS`).
- 자원: GPU를 쓰지 않는다. 확인한 유휴 메모리는 약 470MB이며 상한은 `PHOENIX_MEM_LIMIT`(기본 `1g`)·`PHOENIX_CPUS`(기본 `1`)다.
- 폐쇄망·PHI: 외부 접속(폰트·텔레메트리·가격표 갱신), 화면의 AI 비서, 외부 LLM Playground를 끈다.
- 본문: 기본은 프롬프트·응답 본문을 남기지 않는다. 개발·벤치마크에서 셸 값 `AI_TRACE_CONTENT=true`로 켜면(예: `AI_TRACE_CONTENT=true ./start-deployment.sh local --observability`, `.env`에 둬도 동작) 화면에서 지시문·입력·응답을 대화 형태로 본다. 배포 스크립트가 기동 시 `추적 본문 기록: 켬/끔`을 출력한다. 운영에서 켜려면 개인정보·보안 승인이 필요하다. 본문을 꺼도 Provider 오류 응답 본문은 오류 이벤트에 남는다.
- 끄기: `docker compose <같은 -f 조합> stop phoenix`. API는 그대로 동작하며 추적만 버려지고 경고 로그가 남는다. 완전히 빼려면 `compose.observability.yaml` 없이 다시 `up -d`한다.

```bash
docker compose -f compose.yaml -f compose.observability.yaml up -d --no-build
docker compose -f compose.yaml -f compose.observability.yaml logs --tail=100 phoenix
```

## 운영과 갱신

- PostgreSQL의 `corpus-db` 볼륨은 말뭉치 및 DUR 데이터를 보관한다. `docker compose down`은 기본적으로 볼륨을 유지하며 `down -v`는 데이터를 삭제한다.
- 새 버전 갱신은 배포 스크립트면 [새 버전으로 갱신](#새-버전으로-갱신)을 따른다. 수동 절차면 새 이미지를 새 태그로 빌드·전달·적재한 후 셸의 `AX_G_AI_IMAGE`를 새 태그로 지정하고 사용 중인 Compose 파일 조합으로 `up -d --no-build`를 실행한다. 적재기 로그와 API 상태를 다시 확인한다. 이전 이미지와 DB 백업은 복구 정책에 따라 유지한다.
- 말뭉치와 DUR 적재기는 이미지의 데이터를 DB에 적재하고 활성화한다. 이전 데이터로 전환할 때는 실제 Compose 파일 조합에 대해 각각 `docker compose ... run --rm corpus-loader --activate <release_id>`, `docker compose ... run --rm dur-loader --activate <dataset_id>`를 사용한다. 활성 데이터는 기본 60초 주기로 갱신된다. 자세한 내용은 [말뭉치 DB 문서](clinical_corpus_postgres.md)를 참고한다.
- DB를 사용할 수 없으면 말뭉치 파일 기반 검색으로 대체될 수 있지만 DUR은 DB의 활성 데이터셋이 없으면 비활성화된다. 로그의 `clinical_evidence_loaded`, `dur_loaded`, `dur_unavailable`을 확인한다.
- 이미지에는 `data/reference_data.sqlite`가 포함되지 않는다. 사용 시 파일을 읽기 전용으로 마운트하고 `AX_G_REFERENCE_DATA_DB_PATH`와 `AX_G_ATTRIBUTED_SOURCES_ONLY=false`를 설정해야 한다. 출처·라이선스 확인과 생성 방법은 [참조 데이터 문서](reference_data_sqlite.md)를 따른다.
- 중지하려면 사용 중인 파일 조합으로 `docker compose -f compose.yaml down`을 실행한다. vLLM·SGLang 또는 Redis를 사용했다면 해당 파일을 같은 순서로 지정한다.
