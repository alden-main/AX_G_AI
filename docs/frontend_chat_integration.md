# 의료진 챗봇 프론트엔드·Bridge 연동 안내

> 대상: 의료진 채팅 화면 및 Bridge server 담당자  
> 기준: 간소화된 Bridge ↔ AX-G AI 내부 API

## 1. 화면 흐름

브라우저는 AX-G AI를 직접 호출하지 않는다. 환자를 선택하면 Bridge가 먼저 EMR 원문을 조회한다. Bridge는 이를 AX-G AI의 EMR 준비 요청에 한 번에 전달하고, AX-G AI는 환자 ID 대조와 허용 목록 변환 후 프로세스 메모리에 보관해 최초 환자 상태 요약을 생성한다.

```text
환자 선택 → Bridge가 EMR 조회 → Bridge가 원문을 준비 요청으로 전달 → 최소 문맥 메모리 준비·AI 환자 상태 요약 생성 → 준비 완료·요약 표시 → 채팅 활성화
질문 입력 → Bridge가 patient_id + question 전달 → 답변 표시
```

환자를 변경하면 현재 답변을 숨기고 새 환자의 준비가 성공할 때까지 채팅을 비활성화한다. `context_id`, 진료 에피소드, 조회 기간, 언어, 지식문서 ID, 대화 이력은 프런트 또는 Bridge에서 AX-G AI로 보내지 않는다.

## 2. 환자 EMR 준비 API

```http
POST /internal/v1/patient-context
Content-Type: application/json
```

```json
{
  "patient_id": "pid-4356",
  "emr_payload": {"patient": {"patientId": "pid-4356"}, "bloodPressureList": []}
}
```

성공 응답:

```json
{
  "patient_id": "pid-4356",
  "status": "ready",
  "patient_summary": {
    "message": "EMR 데이터가 로드되었습니다.",
    "sections": [
      {"label": "최근 혈당", "content": "식후 혈당 200 mg/dL가 기록되어 있습니다."},
      {"label": "최근 검사", "content": "당화혈색소(HbA1c) 6.8이 기록되어 있습니다."}
    ],
    "guidance": "각 수치의 측정 시점과 현재 처방을 함께 확인하세요."
  },
  "emr_updated_at": "2026-09-22T10:30:00+09:00"
}
```

성공 전에는 입력창과 전송 버튼을 비활성화한다. 성공 시 `patient_summary.message`를 첫 줄로, `sections[]`의 `label`을 강조한 뒤 `content`를 이어서, `guidance`를 마지막 안내 문구로 표시한다. 같은 환자의 새로고침도 이 API를 다시 호출하며, 응답으로 받은 새 요약으로 이전 요약을 교체한다.

## 3. 채팅 API

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

성공 응답:

```json
{
  "answer": "최근 혈당과 HbA1c 추이를 함께 확인하고, 식전·식후 측정 조건과 현재 처방 상태를 검토하는 것이 필요합니다.",
  "source": "EMR 데이터베이스",
  "emr_updated_at": "2026-09-22T10:30:00+09:00"
}
```

- 답변은 한국어로 표시한다. 한국어로 표현하기 어려운 의학 용어만 영어를 허용한다.
- `source`는 항상 `EMR 데이터베이스`이며 답변 하단에 표시한다.
- `emr_updated_at`은 “EMR 기준 시각”으로 표시한다.
- Bridge는 현재 선택 환자와 응답을 요청한 `patient_id`가 같을 때만 응답을 표시한다.

## 4. EMR 수신과 오류 처리

Bridge는 환자 선택 시 EMR 원문 객체를 준비 요청의 `emr_payload`로 AX-G AI에 전달한다. AX-G AI는 원문을 별도로 조회하지 않으며, 환자 ID 대조 뒤 허용된 목록만 프로세스 메모리에 보관한다. 이후 질문에서는 이 최소 문맥만 AI에 전달한다.

| code | 화면 처리 |
|---|---|
| `PATIENT_CONTEXT_NOT_READY` | EMR 준비 API를 호출하고 준비 전 안내를 표시한다. |
| `PATIENT_MISMATCH` | 채팅을 중지하고 환자 정보를 다시 확인한다. |
| `EMR_UNAVAILABLE` | 이전 환자의 답변을 재사용하지 않고 재시도 안내를 표시한다. |
| `AI_PROVIDER_UNAVAILABLE` | 답변을 추정하지 않고 재시도 안내를 표시한다. |

Bridge와 브라우저는 원천 EMR 전체나 AI Provider API Key를 노출하거나 저장하지 않는다. 전체 계약은 [Bridge API 계약](contracts/bridge-ai-api-spec.md)을 따른다.
