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

`.env`에는 실제 OpenAI API 키가 들어가므로 Linux 서버에서만 생성한다. `.env`는 Git에 커밋하거나 이미지에 포함하지 않으며, 로컬 PC에서 만든 이미지 tar 파일에도 포함되지 않는다.

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
AX_G_PORT=8100
```

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

Bridge와 같은 Docker 네트워크에서 운영할 때는 호스트 포트를 공개하지 말고 `--network`으로 전용 네트워크에 연결한다. 전체 Compose 배포와 이미지 전달 절차는 [docker_local_run.md](docker_local_run.md)를 참고한다.
