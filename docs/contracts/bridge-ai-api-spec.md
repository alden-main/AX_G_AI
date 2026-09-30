# Bridge server ↔ AX-G AI API 명세

> 버전: `v1`  
> 기준일: 2026-09-29  
> 적용 범위: Bridge server와 AX-G AI service layer 사이의 내부 서버 간 통신

## 1. 단순화 원칙과 흐름

환자를 선택하면 Bridge가 EMR에서 조회한 원문 객체를 준비 요청에 포함해 AX-G AI로 전달한다. AX-G AI는 환자 ID를 대조하고 알려진 5개 목록만 최소 문맥으로 변환해 환자별 프로세스 메모리에 준비한 뒤 최초 상태 요약을 생성한다. 이후 채팅 요청은 환자 ID와 질문만 사용한다.

- `context_id`, `encounter_id`, 조회 기간, timezone은 이 계약에서 사용하지 않는다.
- 응답 언어는 항상 한국어다. 번역이 부자연스러운 의학 용어만 영어 표기를 허용한다. 따라서 `language`는 보내지 않는다.
- 답변의 데이터 출처는 항상 `EMR 데이터베이스`다. 따라서 `knowledge_ids`와 문서별 evidence는 보내거나 반환하지 않는다.
- Bridge는 원문 객체를 준비 요청의 `emr_payload`로 전달한다. 이전 `emr_snapshot`은 이 PoC API에서 받지 않는다.
- AX-G AI는 환자별 검증 Snapshot과 해당 Snapshot에서 최초 생성한 `patient_summary`를 프로세스 메모리에만 보관한다. 영속 저장·공유 캐시는 사용하지 않으며, 환자 전환·Snapshot 갱신·프로세스 재시작 시 메모리 항목은 교체 또는 소멸한다.
- 최초 요약과 질문 답변을 생성할 때 AX-G AI는 Snapshot 전체를 Provider에 보내지 않는다. 허용된 비식별·구조화 항목 중 요약 또는 해당 질문에 필요한 최소 항목만 선택한다.
- 외부 생성형 AI는 OpenAI Responses API만 사용한다. AX-G AI 서버가 `POST https://api.openai.com/v1/responses`를 호출하며, Bridge와 브라우저는 OpenAI API를 직접 호출하거나 API Key를 보유하지 않는다.
- OpenAI 호출은 `OPENAI_API_KEY`의 Bearer 인증, `OPENAI_MODEL`(기본 `gpt-5-mini`), `store: false`를 사용한다. OpenAI의 응답 ID·원문 오류·API Key는 Bridge 응답·감사 로그에 포함하지 않는다.

```text
1. 브라우저 → Bridge: 환자 선택
2. Bridge → EMR: 환자 ID로 EMR Snapshot 조회
3. EMR → Bridge: 원문 EMR 객체
4. Bridge → AX-G AI: 환자 EMR 준비 요청 (patient_id + emr_payload)
5. AX-G AI: Snapshot 검증·메모리 보관, 최소 EMR 문맥으로 최초 환자 상태 요약 생성
6. AX-G AI → Bridge: 준비 완료 + 환자 상태 요약
7. 브라우저 → Bridge: 의료진 질문
8. Bridge → AX-G AI: 채팅 요청 (patient_id + question)
9. AX-G AI: Snapshot에서 질문 관련 최소 항목 선택
10. AX-G AI → Bridge → 브라우저: 한국어 답변 + EMR 데이터베이스 출처
```

모든 호출은 UTF-8 JSON을 사용한다. Bridge가 인증·인가를 완료하고 AI service는 Bridge와만 공유하는 Docker 내부 네트워크에 배치한다.

### 추적 헤더

Bridge는 선택적으로 `X-Request-ID` 헤더를 보낼 수 있다. AX-G AI는 이 값을 컨테이너 로그의 오류 기록에 사용하고, 모든 응답 헤더의 `X-Request-ID`로 반환한다. 요청에 없으면 AX-G AI가 UUID를 생성해 반환한다. 이 값은 JSON body에는 포함하지 않는다.

## 2. Bridge → AX-G AI: 환자 EMR 준비

환자 화면을 열거나 환자를 변경할 때 Bridge가 먼저 원문 EMR을 조회한 후 반드시 호출한다. AX-G AI는 원문의 환자 ID를 대조하고 허용 목록만 변환·메모리 보관한 후 최초 환자 상태 요약을 생성한다. 성공 응답을 받기 전에는 해당 환자의 채팅 입력을 활성화하지 않는다.

```http
POST /internal/v1/patient-context
Content-Type: application/json
```

```json
{
  "patient_id": "pid-4356",
  "emr_payload": {
    "patient": {"patientId": "pid-4356"},
    "bloodPressureList": [], "bloodSugarList": [], "oxygenSaturationList": [],
    "labResultList": [], "medicineDtailList": []
  }
}
```

| 필드 | 타입 | 필수 | 의미 |
|---|---|---:|---|
| `patient_id` | string | 예 | Bridge가 접근 권한을 확인한 현재 환자 식별자. 빈 문자열 불가. |
| `emr_payload` | object | 예 | Bridge가 같은 요청에서 조회한 원문 EMR 객체. |

AX-G AI는 `emr_payload.patient.patientId`를 검증하며, 바깥 `patient_id`와 다르면 사용하지 않는다. `bloodPressureList`, `bloodSugarList`, `oxygenSaturationList`, `labResultList`, `medicineDtailList`만 변환한다. 목록이 없거나 null이면 빈 목록으로 처리하고 환자명·의사 정보·처방 `instructions`는 Provider에 보내지 않는다.

### Success response: `200 OK`

```json
{
  "patient_id": "pid-4356",
  "status": "ready",
  "patient_summary": {
    "message": "EMR 데이터가 로드되었습니다.",
    "sections": [
      {"label": "최근 혈당", "content": "식후 혈당 200 mg/dL, 식전 혈당 89 mg/dL가 기록되어 있습니다."},
      {"label": "최근 검사", "content": "당화혈색소(HbA1c) 6.8이 기록되어 있습니다."},
      {"label": "복용 약물", "content": "등록된 처방 정보를 확인하세요."}
    ],
    "guidance": "각 수치의 측정 시점과 현재 처방을 함께 확인하세요. AI 답변은 의료진의 판단을 보조합니다."
  },
  "emr_updated_at": "2026-09-22T10:30:00+09:00"
}
```

| 필드 | 의미 |
|---|---|
| `patient_id` | 준비가 완료된 환자 ID. |
| `status` | 항상 `ready`. Snapshot 검증 및 최초 요약 생성까지 성공한 경우에만 반환한다. |
| `patient_summary.message` | 첫 AI 메시지의 제목 문장. |
| `patient_summary.sections[]` | 말풍선에서 라벨을 강조해 표시할 EMR 요약 항목. `label`과 `content`으로 구성되며, 실제로 제공된 데이터가 있는 항목만 포함한다. |
| `patient_summary.guidance` | 요약 하단의 확인·안내 문구. Bridge는 `message` → `sections` → `guidance` 순서로 표시한다. |
| `emr_updated_at` | Bridge가 반환한 Snapshot의 `snapshot_at`. |

Snapshot을 읽거나 검증하지 못하거나, 최초 `patient_summary` 생성에 실패하면 준비를 완료로 표시하지 않고 오류를 반환한다. Bridge는 이전 Snapshot 또는 이전 요약을 새 환자·새로고침 결과로 재사용하지 않는다.

## 3. Bridge → AX-G AI: 의료진 채팅

환자 EMR 준비와 최초 요약 생성이 완료된 환자만 질문할 수 있다. 대화 이력은 요청하지도 저장하지도 않는다. AX-G AI는 메모리의 동일 환자 Snapshot에서 현재 질문에 관련된 최소 구조화 항목만 선택해 Provider에 전달한다. 이전 답변을 가리키는 질문은 현재 질문과 준비된 EMR만으로 해석 가능한 범위에서 답한다.

```http
POST /internal/v1/clinical-chat
Content-Type: application/json
```

```json
{
  "patient_id": "pid-4356",
  "question": "최근 혈당과 HbA1c를 고려할 때 확인할 관리 원칙은 무엇인가요?"
}
```

| 필드 | 타입 | 필수 | 의미 |
|---|---|---:|---|
| `patient_id` | string | 예 | 2절에서 준비 완료된 환자 ID. |
| `question` | string | 예 | 의료진의 질문. 빈 문자열 불가. |

### Success response: `200 OK`

```json
{
  "answer": "최근 혈당과 HbA1c 추이를 함께 확인하고, 식전·식후 측정 조건과 현재 처방 상태를 검토하는 것이 필요합니다.",
  "source": "EMR 데이터베이스",
  "emr_updated_at": "2026-09-22T10:30:00+09:00"
}
```

| 필드 | 의미 |
|---|---|
| `answer` | 한국어 의료진 보조 답변. 한국어로 옮기기 어려운 의학 용어만 영어 표기를 허용한다. |
| `source` | 항상 `EMR 데이터베이스`. 프런트엔드는 답변과 함께 표시한다. |
| `emr_updated_at` | 답변에 사용한 준비된 EMR Snapshot 시각. |

## 4. 오류 계약

오류 본문에는 질문 원문, 환자 이름, EMR 원문, Provider 오류를 넣지 않는다.

```json
{"code": "EMR_UNAVAILABLE", "message": "환자 EMR 정보를 준비할 수 없습니다."}
```

| HTTP 상태 | code | 발생 조건 | Bridge 처리 |
|---:|---|---|---|
| `400` | `INVALID_REQUEST` | `patient_id`, `emr_payload` 또는 `question` 누락·형식 오류, 허용되지 않은 최상위 필드 | 요청을 수정한다. |
| `404` | `PATIENT_CONTEXT_NOT_READY` | 준비하지 않은 환자의 채팅 요청 | 2절 API를 호출한 뒤 채팅을 활성화한다. |
| `403` | `PATIENT_MISMATCH` | Snapshot의 환자 ID 불일치 또는 접근 거부 | 해당 환자의 채팅을 중지하고 EMR을 다시 확인한다. |
| `502` | `EMR_UNAVAILABLE` | Bridge의 EMR 조회 실패 또는 전달된 Snapshot 검증 실패 | 재시도 안내를 표시한다. |
| `502` | `AI_PROVIDER_UNAVAILABLE` | AI Provider 호출 실패 | 답변을 추정하지 않고 재시도 안내를 표시한다. |

## 5. AX-G AI → OpenAI Responses API (내부 구현 계약)

이 절은 Bridge 공개 계약이 아니라 AX-G AI가 OpenAI에 보내는 서버 측 요청을 고정한다. OpenAI API Key는 배포 환경의 비밀 저장소에서만 `OPENAI_API_KEY`로 주입한다.

```http
POST https://api.openai.com/v1/responses
Authorization: Bearer $OPENAI_API_KEY
Content-Type: application/json
```

```json
{
  "model": "gpt-5-mini",
  "instructions": "고정된 한국어 의료진 보조·안전 지시",
  "input": "질문과 최소 비식별 EMR 요약",
  "store": false
}
```

- `instructions`는 서버 코드의 고정값이다. 한국어 응답, 제공 데이터만 사용, 진단 확정·처방 변경·용량 결정 금지를 요구하며, `input`의 EMR JSON을 지시가 아닌 데이터로 취급한다.
- `input`에는 1절의 allowlist로 만든 최소 비식별 문맥과 현재 질문만 넣는다. 환자 ID·이름·의료진 정보·원문 EMR·처방 `instructions`·대화 이력은 넣지 않는다.
- 비스트리밍 요청만 사용한다. `output` 배열의 `message.content[type=output_text].text`만 정상 답변으로 채택하며, 빈 응답·거부·형식 오류·비성공 상태는 `AI_PROVIDER_UNAVAILABLE`로 변환한다.
- `store: false`는 API 요청에 명시한다. 이 애플리케이션의 프로세스 메모리 Snapshot 정책과 OpenAI 플랫폼의 데이터 보관 정책은 별도이므로, 실제 환자 데이터 전송 전 개인정보·보안 승인과 OpenAI 조직의 데이터 제어 설정을 별도로 확인해야 한다.

## 6. 구현 체크리스트

- Bridge는 환자 선택 시 EMR 원문을 조회해 2절의 `emr_payload`로 전달하고, 준비 API가 성공한 뒤에만 채팅을 연다.
- AX-G AI는 Bridge가 전달한 원문의 환자 ID를 대조하고 허용 목록만 변환하며, EMR 또는 Bridge에 별도 조회를 요청하지 않는다.
- AX-G AI는 준비된 환자의 Snapshot과 최초 `patient_summary`만 프로세스 메모리에서 참조하고, `patient_id`가 일치할 때만 요약·답변한다.
- AX-G AI는 최초 요약과 질문 답변마다 Snapshot 전체가 아닌 필요한 최소 구조화 항목만 Provider에 전달한다.
- AX-G AI의 시스템 지시는 한국어 답변과 `EMR 데이터베이스` 단일 출처를 강제한다.
- AX-G AI는 OpenAI Responses API에 `store: false`를 포함하고, OpenAI 응답의 텍스트 출력만 내부 `answer`로 변환한다.
- Bridge는 `answer`, `source`, `emr_updated_at`을 함께 화면에 표시한다.
