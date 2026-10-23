# Docker 로컬 실행 및 Bridge 연동 확인

이 문서는 새 Bridge ↔ AX-G AI 내부 API 계약으로 컨테이너를 실행하고 통합을 확인하는 방법을 설명한다. AX-G AI는 환자 EMR Snapshot을 자체 영속 저장하지 않으며, Bridge가 먼저 환자 준비 API를 호출한 뒤에만 채팅을 요청한다.

## 사전 준비

- Docker Desktop 또는 Docker Engine과 Docker Compose v2가 실행 중이어야 한다.
- Bridge가 EMR Snapshot을 조회할 수 있어야 하며, AX-G AI의 준비 API를 호출할 수 있어야 한다.
- 프로젝트 루트에 `.env`를 둔다. 이 파일은 Git과 Docker 이미지에 포함하지 않는다.

`.env.example`을 복사해 실제 서버별 값으로 채운다.

```bash
cp .env.example .env
```

```dotenv
# OpenAI Responses API 설정
OPENAI_API_KEY=실제_키
OPENAI_MODEL=gpt-5-mini
OPENAI_TIMEOUT_SECONDS=30

AX_G_PORT=8100
```

실제 Provider API Key를 저장소, 문서, 로그, 브라우저 코드에 넣지 않는다. Bridge의
인증·인가를 통과한 요청은 Bridge와 AX-G AI가 공유하는 전용 Docker 내부 네트워크로만
전달한다. AX-G AI는 별도의 Bearer token을 요구하지 않는다.

## 참조 SQLite 데이터

이미지에는 `data/reference_data.sqlite`가 포함된다. 이 파일은 의약품·상병 코드의 표시명,
효능·효과, 주의사항을 보강하는 읽기 전용 참조 데이터이며, 컨테이너는 기본적으로
`AX_G_REFERENCE_DATA_DB_PATH=/app/data/reference_data.sqlite`를 사용한다.

이미지를 빌드하기 전, 원본 Pickle이 변경됐을 때만 아래 명령으로 DB를 다시 생성한다.

```bash
mkdir -p data
uv run ax-g-ai-reference-db \
  --medicine medicine_dict.pkl \
  --disease disease_dict.pkl \
  --output data/reference_data.sqlite \
  --replace
```

참조 DB에는 약품 설명 원문이 포함되어 있어 이미지 크기가 증가한다. 원본 데이터의
출처·최신성·이용 권한을 확인한 버전만 빌드에 포함한다. 운영에서 별도 최신 DB를 쓸 경우에만
읽기 전용 volume으로 `/app/data`를 마운트하고 같은 환경 변수를 그 경로로 변경한다.

## Docker Compose 실행

프로젝트 루트에서 빌드와 기동을 실행한다.

```bash
docker compose up --build -d
```

상태·로그·종료 명령은 다음과 같다.

```bash
docker compose ps
docker compose logs -f ax-g-ai
docker compose down
```

실행 중인 컨테이너의 오류는 다음처럼 확인한다.

```bash
docker logs -f ax-g-ai-local
```

내부 API가 실패하면 로그에는 본문·EMR 원문·질문·API Key를 제외하고 `request_id`,
HTTP 상태, 메서드, 경로와 실패 지점이 출력된다. Bridge가 `X-Request-ID`를 보내면 그
값이 그대로 응답 헤더와 로그에 사용되고, 없으면 AX-G AI가 UUID를 만든다. 예를 들어
누락 필드 문제는 `api_validation_failed ... failures=body.emr_payload:missing`, Provider의
HTTP 오류는 `ai_provider_failed ... status=400`으로 확인한다.

직접 이미지를 실행할 때도 동일하게 `.env`를 주입한다.

```bash
docker build -t ax-g-ai:local .
docker run --rm --name ax-g-ai-local --env-file .env -p 8100:8100 ax-g-ai:local
```

## Linux 서버용 이미지 빌드 및 전달

다른 서버에서 소스 코드 없이 AX-G AI를 실행하려면 Dockerfile이 아니라 **빌드된 Docker
image**를 전달한다. 대상 Linux 서버의 CPU 아키텍처에 맞추어 빌드해야 한다. 일반적인
x86_64 Linux 서버는 `linux/amd64`를 사용한다. ARM 서버(예: AWS Graviton)는
`linux/arm64`를 사용한다.

### 1. 배포용 이미지 빌드

개발 PC가 macOS Apple Silicon이거나 대상 서버와 CPU가 다를 때에도 아래처럼 플랫폼을
명시해 빌드한다. `--load`는 빌드 결과를 현재 Docker 엔진에 적재하므로, 다음 단계의
`docker save`에서 사용할 수 있다.

```bash
# x86_64 Linux 서버용
docker buildx build --platform linux/amd64 --load -t ax-g-ai:1.0.0 .
```

대상 서버가 ARM64 Linux인 경우에는 플랫폼과 파일명만 다음처럼 바꾼다.

```bash
docker buildx build --platform linux/arm64 --load -t ax-g-ai:1.0.0 .
```

빌드 이미지의 플랫폼을 확인한다.

```bash
docker image inspect ax-g-ai:1.0.0 --format '{{.Os}}/{{.Architecture}}'
```

### 2. 이미지 파일 생성 및 서버로 전달

빌드한 이미지를 tar 파일로 내보낸다. 아래 예시는 `linux/amd64` 이미지 기준이다.

```bash
docker save --output ax-g-ai_1.0.0_linux_amd64.tar ax-g-ai:1.0.0
shasum -a 256 ax-g-ai_1.0.0_linux_amd64.tar
```

tar 파일과 출력된 SHA-256 값을 전달한다. 예를 들어 SSH로 전송할 수 있다.

```bash
scp ax-g-ai_1.0.0_linux_amd64.tar deploy@SERVER_HOST:/opt/ax-g/deploy/
```

이미지 tar 파일에는 애플리케이션과 런타임 의존성만 들어 있다. `OPENAI_API_KEY` 같은
운영 비밀값과 `.env` 파일은 포함되지 않으므로, 서버에서 별도로 생성·관리한다.

### 3. 대상 Linux 서버에서 이미지 적재

대상 서버에서 전달받은 파일의 무결성을 확인한 뒤 Docker에 적재한다.

```bash
cd /opt/ax-g/deploy
sha256sum ax-g-ai_1.0.0_linux_amd64.tar
docker load --input ax-g-ai_1.0.0_linux_amd64.tar
docker image inspect ax-g-ai:1.0.0 --format '{{.Os}}/{{.Architecture}}'
```

`sha256sum`의 결과가 전달받은 값과 같아야 한다. 마지막 명령의 결과가 서버 아키텍처와
맞지 않으면 컨테이너가 기동하지 않거나 에뮬레이션으로 매우 느리게 실행될 수 있다.

### 4. Compose에서 AX-G AI 이미지 참조

배포 서버의 Compose 파일에서는 `ax-g-ai` 서비스의 `build: .`을 제거하고, 적재한
이미지 태그를 지정한다. Bridge 서비스는 AI 서비스와 같은 `bridge-ai` 네트워크에
연결하고 `http://ax-g-ai:8100`으로 호출한다.

```yaml
services:
  ax-g-ai:
    image: ax-g-ai:1.0.0
    env_file:
      - .env
    expose:
      - "8100"
    networks:
      - bridge-ai
      - ai-provider-egress

  bridge:
    image: your-registry/bridge:1.0.0
    depends_on:
      - ax-g-ai
    networks:
      - bridge-ai

  frontend:
    image: your-registry/frontend:1.0.0
    # frontend의 공개 포트와 네트워크는 해당 프로젝트 설정을 따른다.

networks:
  bridge-ai:
    internal: true
  ai-provider-egress: {}
```

배포 디렉터리에 위 Compose 파일(프론트엔드·Bridge의 실제 설정을 반영한 파일)과 AX-G
AI용 `.env`를 둔 다음 기동한다.

```bash
docker compose up -d
docker compose ps
docker compose logs -f ax-g-ai
```

`.env` 파일은 서버에서만 생성하고 권한을 제한한다.

```bash
cp .env.example .env
chmod 600 .env
```

이미지를 새 버전으로 교체할 때는 새 태그(예: `ax-g-ai:1.0.1`)로 빌드·전달·`docker
load`한 뒤 Compose의 `image:` 태그를 변경하고 다시 기동한다.

```bash
docker compose up -d
```

Docker Registry(Docker Hub, GHCR 또는 사설 Registry)를 운영할 수 있다면 tar 파일 전달
대신 이미지에 태그를 붙여 `docker push`하고, 서버에서 `docker compose pull && docker
compose up -d`로 배포하는 방식이 버전 관리와 반복 배포에 더 적합하다.

## 기동 및 계약 확인

기동 확인은 인증 없이 가능하다.

```bash
curl http://127.0.0.1:8100/health
```

정상 응답:

```json
{"status":"ok"}
```

Bridge는 환자 선택 직후 아래 준비 요청을 먼저 보낸다.

```http
POST /internal/v1/patient-context
Content-Type: application/json

{
  "patient_id":"pid-4356",
  "emr_payload": {
    "patient":{"patientId":"pid-4356"},
    "bloodPressureList":[], "bloodSugarList":[], "oxygenSaturationList":[],
    "labResultList":[], "medicineDtailList":[]
  }
}
```

준비 성공 응답에는 `patient_summary`가 포함된다. Bridge는 이를 환자 상태 영역 또는 첫 AI 메시지로 표시하고, 성공 후에만 채팅을 요청한다.

```json
{"patient_id":"pid-4356","status":"ready","patient_summary":{"message":"EMR 데이터가 로드되었습니다.","sections":[{"label":"최근 검사","content":"검사 결과를 확인하세요."}],"guidance":"현재 기록을 함께 확인하세요."},"emr_updated_at":"2026-09-22T10:30:00+09:00"}
```

```http
POST /internal/v1/clinical-chat
Content-Type: application/json

{"patient_id":"pid-4356","question":"최근 혈당과 HbA1c를 고려할 때 확인할 관리 원칙은 무엇인가요?"}
```

성공 응답은 다음 세 필드만 포함한다.

```json
{
  "answer":"최근 혈당과 HbA1c 추이를 함께 확인하세요.",
  "source":"EMR 데이터베이스",
  "emr_updated_at":"2026-09-22T10:30:00+09:00"
}
```

`POST /internal/v1/clinical-chat`을 준비 전에 호출하면 `404 PATIENT_CONTEXT_NOT_READY`가 정상이다.

## Bridge 원문 확인 사항

Bridge는 환자 선택 시 EMR 원문 객체를 준비 request의 `emr_payload`로 전달한다. AX-G AI는 `patient.patientId`를 바깥 `patient_id`와 대조하고, 혈압·혈당·SpO2·검사·처방 목록만 변환한다. 목록 누락/null은 빈 목록으로 처리하며, 원문 환자 ID 불일치는 `403 PATIENT_MISMATCH`, 원문 형식 오류는 `502 EMR_UNAVAILABLE`로 준비를 실패 처리한다.

## 네트워크 유의사항

- 배포 Compose에서는 AI service에 `ports`를 지정하지 않고 `expose: 8100`만 사용한다.
- Bridge와 AX-G AI만 `bridge-ai` 내부 네트워크에 연결한다. Bridge는 `http://ax-g-ai:8100`으로 호출한다.
- AI service만 Provider egress 네트워크에 추가하고, Bridge·디버그 컨테이너·외부 서비스는 `bridge-ai`에 연결하지 않는다.
- `docker run -p 8100:8100`은 로컬 수동 확인에만 사용하고 배포 Compose에는 사용하지 않는다.
