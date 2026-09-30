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
