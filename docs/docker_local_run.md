# Docker 로컬 실행 및 AI-Tank 연결 시험

이 문서는 의료진 챗봇 FastAPI 서버를 Docker로 로컬에서 실행하고, 같은 네트워크의
다른 PC에서 호출하는 방법을 설명한다. 여기의 `AX_G_CHAT_TEST_MODE=true`는 AI-Tank
연결 시험 전용 모드다. 실제 환자 데이터, EMR 또는 운영 인증을 사용하지 않는다.

## 사전 준비

- Docker Desktop 또는 Docker Engine과 Docker Compose v2가 실행 중이어야 한다.
- 프로젝트 루트에 `.env` 파일을 둔다. `.env`는 Git과 Docker 이미지에서 제외된다.

`.env.example`을 복사해 `.env`를 만들고 실제 AI-Tank 키를 입력한다.

```bash
cp .env.example .env
```

```dotenv
AX_G_AI_TANK_API_KEY=실제_키
AX_G_AI_TANK_ENDPOINT=https://ahk-api.ai-tank.co.kr/api/consult
AX_G_AI_TANK_TIMEOUT_SECONDS=30
AX_G_CHAT_TEST_MODE=true
AX_G_PORT=8100
```

`NEXT_PUBLIC_` 접두사는 브라우저 번들에 노출될 수 있으므로 API 키에 사용하지 않는다.

## Docker build 후 직접 실행

프로젝트 루트에서 이미지를 빌드한다.

```bash
docker build -t ax-g-ai:local .
```

다음 명령은 컨테이너를 포그라운드에서 실행한다. `--env-file`로만 키를 주입하므로
키가 이미지 레이어나 명령 출력에 포함되지 않는다.

```bash
docker run --rm \
  --name ax-g-ai-local \
  --env-file .env \
  -e AX_G_CHAT_TEST_MODE=true \
  -p 8100:8100 \
  ax-g-ai:local
```

백그라운드 실행이 필요하면 `-d`를 추가한다.

```bash
docker run -d \
  --name ax-g-ai-local \
  --env-file .env \
  -e AX_G_CHAT_TEST_MODE=true \
  -p 8100:8100 \
  ax-g-ai:local
```

로그와 종료 명령은 다음과 같다.

```bash
docker logs -f ax-g-ai-local
docker stop ax-g-ai-local
```

## Docker Compose로 build 및 실행

`compose.yaml`은 `.env`의 서버 전용 환경 변수를 컨테이너에 주입한다. 빌드와 기동을
한 번에 실행하려면 다음을 사용한다.

```bash
AX_G_CHAT_TEST_MODE=true docker compose up --build
```

백그라운드 실행:

```bash
AX_G_CHAT_TEST_MODE=true docker compose up --build -d
```

상태·로그·종료:

```bash
docker compose ps
docker compose logs -f
docker compose down
```

`.env`에 `AX_G_CHAT_TEST_MODE=true`를 설정했다면 명령 앞의 환경 변수 지정은 생략할 수
있다. 운영에서는 이 값을 `false`로 유지한다. 현재 운영 인증 문맥 연동은 구현 중이므로
`false`에서는 챗봇 요청이 `503`으로 거부되는 것이 정상이다.

## 서버 기동 확인

컨테이너가 실행된 PC에서 다음을 확인한다.

```bash
curl http://127.0.0.1:8100/health
```

정상 응답:

```json
{"status":"ok"}
```

Swagger UI는 브라우저에서 다음 주소로 연다.

```text
http://127.0.0.1:8100/docs
```

로컬 연결 시험 모드에서는 다음 요청으로 AI-Tank까지의 전체 경로를 확인할 수 있다.

```bash
curl -X POST http://127.0.0.1:8100/api/clinical-chat \
  -H 'Content-Type: application/json' \
  -d '{
    "question": "고혈압 환자의 일반적인 생활습관 관리 원칙을 알려주세요.",
    "language": "ko",
    "knowledge_ids": ["local-test-guide"],
    "request_id": "local-test-1",
    "history": []
  }'
```

성공하면 `answer`, `evidence`, `limitations` 필드를 포함한 `200` 응답이 반환된다.
`knowledge_ids`에는 로컬 시험 모드에서 제공하는 `local-test-guide`를 사용해야 한다.

## 같은 네트워크의 다른 PC에서 호출

컨테이너는 `0.0.0.0:8100`에 바인딩되므로, 실행 PC의 LAN IPv4 주소를 사용하면 같은
네트워크의 다른 PC에서도 호출할 수 있다. macOS에서는 다음 명령으로 Wi-Fi 주소를
확인한다.

```bash
ipconfig getifaddr en0
```

예를 들어 결과가 `192.168.0.10`이면 다른 PC에서 다음 주소를 사용한다.

```text
http://192.168.0.10:8100/docs
http://192.168.0.10:8100/health
```

다른 PC에서 접속되지 않으면 다음을 확인한다.

- 두 PC가 동일한 LAN/VPN에 연결되어 있는지
- 실행 PC의 macOS/보안 제품 방화벽이 TCP 8100 인바운드를 허용하는지
- Docker 컨테이너가 실행 중인지 (`docker ps` 또는 `docker compose ps`)
- 포트 충돌이 없는지. 필요하면 `.env`의 `AX_G_PORT`를 변경하고 `-p` 또는 Compose를
  다시 실행한다.

## 안전 유의사항

- `.env`와 실제 API 키는 Git에 커밋하거나 메시지·로그에 붙여 넣지 않는다.
- 로컬 시험 모드는 인증되지 않은 요청을 가상 문맥으로 처리한다. 공용 네트워크 또는
  실제 환자 데이터가 있는 환경에서 사용하지 않는다.
- 운영 전환에는 메인 서버의 인증 세션·환자/에피소드 문맥 공급자와 승인 지식·평가
  저장소를 연결해야 한다.
