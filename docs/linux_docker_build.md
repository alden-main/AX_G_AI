# Linux 대상 Docker 이미지 빌드 및 서버 실행

개발 PC에서 이미지를 빌드해 tar로 전달하고, 소스 코드가 없는 Linux 서버에서 Docker Compose로 실행하는 절차다. 빌드 명령은 프로젝트 루트에서 실행한다. 이미지 태그 `1.0.0`, 서버 주소와 경로는 배포 환경에 맞게 변경한다.

## 구성과 사전 준비

- `compose.yaml`: OpenAI 등 외부 AI 제공자와 함께 API, PostgreSQL/pgvector, TEI 임베딩 서버 및 말뭉치·DUR 적재기를 실행한다.
- `compose.vllm.yaml`: 위 데이터 서비스와 API를 A100 서버의 vLLM과 함께 실행한다.
- `compose.redis.yaml`: 위 두 구성 중 하나에 추가해 Redis 환자 문맥 저장소와 다중 워커를 사용한다.

개발 PC에는 Docker Engine과 Buildx가, 서버에는 Docker Engine과 Compose가 필요하다. 서버의 `uname -m`으로 아키텍처를 확인한다. `x86_64`는 `linux/amd64`, `aarch64`는 `linux/arm64`다. 서버에는 API 이미지 외에 PostgreSQL, TEI 및 선택한 Redis/vLLM 이미지를 내려받을 수 있는 연결이 필요하다. TEI 모델 최초 다운로드에는 약 2.3GB의 저장 공간과 네트워크가 필요하며 기본 메모리 한도는 6GB다. 폐쇄망에서는 필요한 이미지와 모델 캐시를 별도 준비해야 한다.

`.env`에는 API 키와 비밀번호만 두며 서버에서 별도로 관리하고 이미지나 저장소에 포함하지 않는다. 나머지 설정은 Compose 파일과 코드 기본값에 있다. 코드 기본값과 다르게 운영하는 애플리케이션 설정(`AI_REASONING_EFFORT=low`, `AX_G_EVIDENCE_INCLUDE_UNREVIEWED=true`, `AX_G_CLINICAL_DOSES_ENABLED=true`)과 서버 자원·이미지·포트 값은 Compose 파일에 `${이름:-기본값}`으로 적혀 있으며, 서버별로 바꿀 때만 같은 이름을 `.env`에 적는다. 그 밖의 `AX_G_*`·`AI_*` 설정 설명은 `src/ax_g_ai/config.py`에 있다. `.dockerignore`는 `.env`와 이미지 tar를 빌드 컨텍스트에서 제외한다.

## 배포 압축파일로 빌드하고 실행하기

개발 PC에서 다음 스크립트를 실행하면 현재 소스로 이미지를 빌드하고 이미지, PostgreSQL(pgvector) 이미지, 임베딩 서버(TEI) 이미지와 KURE-v1 모델, Compose 파일, 환경 변수 예제, 서버 기동 스크립트를 `ax-g-ai-deploy-amd64.tar.gz`(약 3GB)에 담는다. 체크섬 파일도 함께 생성한다. 기본 대상은 `linux/amd64`이며 ARM 서버는 `AX_G_PLATFORM=linux/arm64`를 지정한다. 이미지 태그와 출력 경로는 각각 `AX_G_IMAGE_TAG`, `AX_G_PACKAGE_PATH`로 변경할 수 있다.

```bash
./scripts/build-deployment.sh
scp ax-g-ai-deploy-amd64.tar.gz ax-g-ai-deploy-amd64.tar.gz.sha256 deploy@SERVER_HOST:/opt/ax-g/deploy/
```

서버에서 검증하고 압축을 푼 뒤 실행한다. 첫 실행은 `.env` 예제를 생성하고 종료한다. 비밀번호와 API 키 등 서버별 값을 입력한 후 다시 실행하면 이미지를 적재하고 Compose 서비스를 시작한다.

```bash
cd /opt/ax-g/deploy
sha256sum --check ax-g-ai-deploy-amd64.tar.gz.sha256
tar -xzf ax-g-ai-deploy-amd64.tar.gz
cd ax-g-ai-deploy
./start-deployment.sh
vi .env
./start-deployment.sh
```

vLLM은 `./start-deployment.sh vllm`, Redis를 추가하려면 `./start-deployment.sh openai --redis` 또는 `./start-deployment.sh vllm --redis`를 사용한다. 동일 디렉터리의 기존 `.env`는 덮어쓰지 않는다. 임베딩 서버는 압축파일의 이미지와 `hf-cache/`의 모델로 외부 접속 없이 기동한다(`EMBEDDING_CACHE_DIR`·`EMBEDDING_HF_HUB_OFFLINE`을 `.env`에 직접 지정하면 그 값을 쓴다). Redis, vLLM 이미지와 vLLM 모델은 포함되지 않으므로 사용할 경우 서버에서 별도로 확보해야 한다. 빌드 PC에 모델이 없으면 `uvx --from huggingface_hub hf download nlpai-lab/KURE-v1 --cache-dir hf-cache`로 먼저 받는다. 임베딩이나 PostgreSQL 이미지를 빼고 만들려면 `AX_G_BUNDLE_EMBEDDING=0`, `AX_G_BUNDLE_POSTGRES=0`을 지정한다. API와 DB·임베딩·Redis는 `restart: unless-stopped`라 서버 재부팅 후 Docker가 다시 띄운다(Docker 서비스가 부팅 시 시작되도록 `sudo systemctl enable docker`). 재부팅 때는 서비스가 동시에 시작되므로 API가 PostgreSQL·Redis 연결을 최대 `AX_G_DEPENDENCY_STARTUP_WAIT_SECONDS`(기본 120초) 동안 다시 시도한다. 압축 해제 후 이미지 tar와 압축파일을 모두 보관할 충분한 디스크 공간이 필요하다. 아래의 수동 절차에도 구성 및 운영 세부사항이 있다.

## 1. 이미지 빌드와 전달

예시는 x86_64 서버용이다. Apple Silicon에서도 Buildx를 사용해 빌드할 수 있다.

```bash
docker buildx build --platform linux/amd64 \
  --tag ax-g-ai:1.0.0 \
  --output type=docker,dest=ax-g-ai_1.0.0_linux_amd64.tar .

# macOS; Linux에서는 sha256sum을 사용한다.
shasum -a 256 ax-g-ai_1.0.0_linux_amd64.tar > ax-g-ai_1.0.0_linux_amd64.tar.sha256

scp ax-g-ai_1.0.0_linux_amd64.tar ax-g-ai_1.0.0_linux_amd64.tar.sha256 \
  compose.yaml compose.vllm.yaml compose.redis.yaml .env.example \
  deploy@SERVER_HOST:/opt/ax-g/deploy/
```

ARM 서버는 플랫폼과 파일명을 `linux/arm64`, `ax-g-ai_1.0.0_linux_arm64.tar`로 바꾼다. 서버의 배포 디렉터리는 미리 생성한다. `.env.example`에는 비밀 값이 없으며 실제 `.env`는 전송하지 않는다.

이미지에는 `data/dur` 스냅샷과 `text_data/corpus` 및 사전 계산 임베딩이 포함된다. 임베딩이 현재 말뭉치를 모두 덮지 못하면 빌드가 실패한다. 말뭉치 변경 시 `text_data/tools/build_corpus.py`와 `text_data/tools/build_embeddings.py`를 순서대로 실행해 갱신한다. 의도적으로 검사를 건너뛸 때만 `--build-arg ALLOW_STALE_EMBEDDINGS=1`을 지정한다.

## 2. 서버에서 이미지와 환경 설정 준비

```bash
cd /opt/ax-g/deploy
sha256sum --check ax-g-ai_1.0.0_linux_amd64.tar.sha256
docker load --input ax-g-ai_1.0.0_linux_amd64.tar
docker image inspect ax-g-ai:1.0.0 --format '{{.Os}}/{{.Architecture}}'
cp .env.example .env
chmod 600 .env
vi .env
```

이미지 아키텍처가 서버와 일치하는지 확인한다. `.env`에서 적어도 다음 항목을 설정한다. 비밀번호는 URL에도 삽입되므로 아래처럼 영숫자로 된 임의 값을 사용한다(예: `openssl rand -hex 24`).

```dotenv
AX_G_AI_IMAGE=ax-g-ai:1.0.0
CORPUS_DB_PASSWORD=생성한_비밀번호
OPENAI_API_KEY=실제_API_키
```

Compose 파일에는 `AX_G_EVIDENCE_INCLUDE_UNREVIEWED=true`와 `AX_G_CLINICAL_DOSES_ENABLED=true`가 설정되어 있다(Provider는 코드 기본값 `openai`, 모델 `gpt-5-mini`). 현재 말뭉치와 용량 규칙은 미검수이므로 배포 전에 해당 설정을 검토한다. 미검수 사용을 끄면 현재 말뭉치의 근거 인용 및 추천 진단·검사가 비활성화된다. 끄려면 `.env`에 `false`로 적는다. 나머지 선택 설정은 `.env.example`과 `src/ax_g_ai/config.py`를 참고한다.

PostgreSQL의 사용자·DB 이름과 비밀번호는 기존 `corpus-db` 볼륨의 초기 설정과 일치해야 한다. `.env`만 바꿔도 기존 DB 계정의 비밀번호가 자동 변경되지는 않는다. ARM 서버에서는 `.env`에 `EMBEDDING_IMAGE=ghcr.io/huggingface/text-embeddings-inference:cpu-arm64-1.9.4`를 추가한다.

## 3. OpenAI 구성 실행

```bash
docker compose -f compose.yaml up -d --no-build
docker compose -f compose.yaml ps -a
docker compose -f compose.yaml logs corpus-loader dur-loader
docker compose -f compose.yaml logs --tail=100 ax-g-ai
curl --fail http://127.0.0.1:8100/health
```

`--no-build`는 서버에 소스가 없어도 적재한 이미지를 사용하도록 한다. 최초 실행 시 PostgreSQL과 임베딩 서버가 정상 상태가 된 후 적재기 두 개가 성공 종료되어야 API가 시작된다. 적재기는 성공 시 종료 상태로 남는 것이 정상이다. `/health`의 정상 응답은 `{"status":"ok"}`다. 이 응답만으로 AI 제공자, Redis 또는 데이터 기능까지 정상이라고 판단할 수 없으므로 로그와 실제 연동 요청도 확인한다. API 설정 오류나 데이터 적재 실패가 있으면 해당 서비스 로그를 확인한다.

`compose.yaml`의 포트 매핑은 `8100:8100`으로 모든 호스트 인터페이스에 공개되며 `.env`의 `AX_G_PORT`는 여기에 적용되지 않는다. 서버 내부에서만 접근하려면 서버의 Compose 파일에서 `127.0.0.1:8100:8100`으로 변경한다. Bridge를 같은 `bridge-ai` 네트워크에 연결할 때는 포트 매핑과 네트워크 접근을 배포 환경에 맞게 구성한다.

소스가 있는 개발 환경에서는 `.env`를 준비한 뒤 `docker compose up -d --build`를 사용할 수 있다.

## 4. A100과 vLLM 구성

NVIDIA 드라이버, NVIDIA Container Toolkit과 GPU 접근을 확인한다.

```bash
nvidia-smi
docker run --rm --gpus all ubuntu nvidia-smi
docker compose version
```

서버 `.env`에 `AX_G_AI_IMAGE`, `CORPUS_DB_PASSWORD`를 설정하고 필요하면 아래 값을 추가한다. vLLM 구성은 API 컨테이너의 `AI_PROVIDER`, `AI_BASE_URL`, `AI_MODEL`을 직접 설정하므로 `.env`의 동일 항목보다 우선한다. `OPENAI_API_KEY`는 vLLM 인증에 사용되지 않는다.

```dotenv
VLLM_MODEL=Qwen/Qwen3-32B-AWQ
AI_MODEL=qwen3-32b-awq
VLLM_MAX_MODEL_LEN=16384
VLLM_GPU_MEMORY_UTILIZATION=0.90
HF_CACHE_DIR=/opt/ax-g/hf-cache
# gated 모델에 필요한 경우 설정
# HF_TOKEN=
```

```bash
docker compose -f compose.vllm.yaml up -d
docker compose -f compose.vllm.yaml ps -a
docker compose -f compose.vllm.yaml logs --tail=100 vllm corpus-loader dur-loader ax-g-ai
curl --fail http://127.0.0.1:8100/health
```

vLLM과 TEI는 최초 기동 시 모델을 다운로드하므로 시간이 걸린다. 두 모델이 캐시에 준비된 뒤 `HF_HUB_OFFLINE=1`로 실행할 수 있다. 기본 vLLM 모델은 Qwen3 32B AWQ이며 A100 40GB를 대상으로 한다. 다른 모델 계열을 사용하면 `compose.vllm.yaml`의 `--reasoning-parser` 설정과 모델 호환성을 함께 확인한다. vLLM 구성의 API 포트는 기본적으로 `127.0.0.1:${AX_G_PORT:-8100}`에만 바인딩된다.

## 5. 선택 사항: Redis와 다중 워커

두 구성 중 실제 사용하는 파일에 `compose.redis.yaml`을 추가한다. `.env`에 `REDIS_PASSWORD`를 설정한다. URL에 삽입되므로 영숫자 비밀번호를 사용한다. 기본 워커 수는 2개이며 `AX_G_WORKERS`로 조절한다.

```bash
# OpenAI 구성 예시. vLLM이면 첫 번째 파일을 compose.vllm.yaml로 바꾼다.
docker compose -f compose.yaml -f compose.redis.yaml up -d --no-build
docker compose -f compose.yaml -f compose.redis.yaml ps -a
docker compose -f compose.yaml -f compose.redis.yaml logs --tail=100 redis ax-g-ai
```

이후 상태 확인, 갱신 및 종료 명령에도 같은 `-f` 파일 조합을 사용한다. Redis는 내부 네트워크에서만 사용하고 디스크에 저장하지 않는다. 재시작 시 환자 문맥을 다시 준비해야 한다. Redis 없이 기본 메모리 저장소를 쓰면 문맥은 프로세스 사이에 공유되지 않으므로 단일 워커로 운영한다. 기동 시 Redis 연결에 실패하면 메모리 저장소로 대체되므로 로그에서 `patient_context_store backend=redis`를 확인한다.

## 운영과 갱신

- PostgreSQL의 `corpus-db` 볼륨은 말뭉치 및 DUR 데이터를 보관한다. `docker compose down`은 기본적으로 볼륨을 유지하며 `down -v`는 데이터를 삭제한다.
- 새 이미지는 새 태그로 빌드·전달·적재한 후 `.env`의 `AX_G_AI_IMAGE`를 변경하고 사용 중인 Compose 파일 조합으로 `up -d --no-build`를 실행한다. 적재기 로그와 API 상태를 다시 확인한다. 이전 이미지와 DB 백업은 복구 정책에 따라 유지한다.
- 말뭉치와 DUR 적재기는 이미지의 데이터를 DB에 적재하고 활성화한다. 이전 데이터로 전환할 때는 실제 Compose 파일 조합에 대해 각각 `docker compose ... run --rm corpus-loader --activate <release_id>`, `docker compose ... run --rm dur-loader --activate <dataset_id>`를 사용한다. 활성 데이터는 기본 60초 주기로 갱신된다. 자세한 내용은 [말뭉치 DB 문서](clinical_corpus_postgres.md)를 참고한다.
- DB를 사용할 수 없으면 말뭉치 파일 기반 검색으로 대체될 수 있지만 DUR은 DB의 활성 데이터셋이 없으면 비활성화된다. 로그의 `clinical_evidence_loaded`, `dur_loaded`, `dur_unavailable`을 확인한다.
- 이미지에는 `data/reference_data.sqlite`가 포함되지 않는다. 사용 시 파일을 읽기 전용으로 마운트하고 `AX_G_REFERENCE_DATA_DB_PATH`와 `AX_G_ATTRIBUTED_SOURCES_ONLY=false`를 설정해야 한다. 출처·라이선스 확인과 생성 방법은 [참조 데이터 문서](reference_data_sqlite.md)를 따른다.
- 중지하려면 사용 중인 파일 조합으로 `docker compose -f compose.yaml down`을 실행한다. vLLM 또는 Redis를 사용했다면 해당 파일을 같은 순서로 지정한다.
