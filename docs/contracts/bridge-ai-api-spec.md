# Bridge ↔ AX-G AI API 명세

내부 서버 간 API. UTF-8 JSON을 사용하며, Bridge에서 인증·인가 및 EMR 조회를 수행한다. AI 서비스는 Bridge와 공유하는 내부 네트워크에 배치한다. 응답은 한국어다.

선택적 요청 헤더 `X-Request-ID`는 모든 응답 헤더에 반환된다. 생략하면 서버가 생성한다.

| 메서드 | 경로 | 용도 |
|---|---|---|
| GET | `/health` | 상태 확인 |
| POST | `/internal/v1/patient-context` | EMR 준비 및 최초 요약 |
| POST | `/internal/v1/patient-context/stream` | EMR 즉시 준비 및 최초 요약 SSE 스트리밍 |
| POST | `/internal/v1/clinical-chat` | 준비된 EMR로 채팅 답변 |
| POST | `/internal/v1/clinical-chat/stream` | 채팅 답변 SSE 스트리밍 |
| POST | `/internal/v1/emr-summary` | EMR을 한 번 요약 |
| POST | `/internal/v1/emr-summary/stream` | EMR 일회성 요약 SSE 스트리밍 |
| POST | `/internal/v1/clinical-recommendations` | 준비된 EMR로 의심 진단·추천 검사·협진 의견(말뭉치 규칙 + 근거) |
| POST | `/internal/v1/clinical-recommendations/stream` | 추천 의견 SSE 스트리밍 |
| GET | `/internal/v1/clinical-recommendations/findings` | `findings` 입력에 쓸 수 있는 소견·사건 키 목록 |

## EMR 입력과 환자 문맥

Bridge가 EMR을 조회해 `emr_payload`로 전달한다. `patient_id`는 빈 문자열일 수 없으며, 원문 `patient.patientId`와 일치해야 한다. 원문 객체 또는 `{"data": 원문객체}` 형식을 받는다.

기존 주요 목록과 진단·계산 입력을 변환하고, 추가 임상 필드도 식별자 제거 후 반영한다. 따라서 실제 요약 내용은 전달된 데이터와 변환 가능한 필드에 따라 달라진다. `emr_updated_at`은 변환된 측정·검사 기록에서 확인된 가장 최근 시각이며, 해당 시각이 없으면 준비 시각이다. EMR 전체의 최종 수정 시각을 보장하지 않는다.

채팅은 먼저 `patient-context`를 호출해야 한다. 준비된 문맥은 프로세스 메모리에 환자별로 저장되므로 재시작 후 또는 다른 프로세스에서 조회할 때 다시 준비해야 한다. 동일 환자 준비를 다시 시도하면 기존 문맥은 삭제되며 준비 실패 시에도 복구되지 않는다. 대화 이력은 저장하지 않는다.

## 환자 EMR 준비

`POST /internal/v1/patient-context`

요청:

```json
{
  "patient_id": "pid-4356",
  "emr_payload": {
    "patient": {"patientId": "pid-4356"},
    "bloodPressureList": [],
    "bloodSugarList": [],
    "oxygenSaturationList": [],
    "labResultList": [],
    "medicineDtailList": []
  }
}
```

성공: `200 OK`

```json
{
  "patient_id": "pid-4356",
  "status": "ready",
  "patient_summary": {
    "message": "EMR에 혈당 및 검사 기록이 있습니다.",
    "sections": [{"label": "최근 혈당", "content": "혈당이 기록되어 있습니다."}],
    "guidance": "검사 결과가 기록되어 있습니다."
  },
  "emr_updated_at": "2026-09-22T10:30:00+09:00"
}
```

`patient_summary.sections`는 `label`, `content` 객체의 배열이다. `message`는 한 줄 요약이며 `sections`와 `guidance`는 합쳐서 추가 요약 4~5줄을 목표로 한다. 기록이 부족하면 줄 수가 줄어들 수 있다. EMR에 기록된 사실을 요약하고 새로운 추정 진단은 제외하도록 요청한다. 출력은 모델에 따라 달라진다. 준비와 요약 생성이 모두 성공한 경우에만 `status: ready`를 반환한다.

## 채팅

`POST /internal/v1/clinical-chat`

요청:

```json
{
  "patient_id": "pid-4356",
  "question": "최근 혈당을 요약해 주세요.",
  "reference_record": "추가 참고 기록"
}
```

`patient_id`와 `question`은 필수이며 공백만 입력할 수 없다. `reference_record`는 선택 사항이고 제공할 경우 공백만 입력할 수 없으며 최대 12,000자다. 허용되지 않은 최상위 필드는 오류다. `question`과 `reference_record`는 EMR 매퍼의 식별자 제거를 거치지 않고 Provider에 전달되므로 Bridge에서 내용을 관리한다.

성공: `200 OK`

```json
{
  "answer": "최근 혈당을 확인하세요.",
  "source": "EMR 데이터베이스",
  "emr_updated_at": "2026-09-22T10:30:00+09:00"
}
```

### 선택 필드 `evidence` (최신화 근거 인용)

AI 서버에 근거 인용(`AX_G_EVIDENCE_CORPUS_DIR`)이 켜져 있고 질문에 맞는 근거가 있을 때만 들어간다. 없으면 필드 자체가 생략되므로 기존 응답 처리와 호환된다. 답변 본문의 `[근거 n: 출처명]`(다시 인용하면 `[근거 n]`) 번호가 `n`과 같다. 본문은 검수 여부를 쓰지 않고 출처명으로 근거를 밝히며, 검수 상태는 이 목록의 `review_label`로 화면에 표시한다. 출처 제목·링크는 본문에 반복하지 않으므로 화면에서 이 목록으로 표시한다.

```json
"evidence": [
  {
    "n": 1,
    "chunk_id": "369-C03-chunk",
    "claim_id": "369-C03",
    "title": "담낭 및 담도질환 > 급성 담낭염 수술 시기와 수술 불가 환자",
    "text": "급성 담낭염 수술 시기와 수술 불가 환자\n복강경 담낭절제가 1차 치료이고 …",
    "review_status": "draft",
    "sources": [{"id": "PAPER-WSES-CHOLECYSTITIS-2020", "title": "…", "citation": "Pisano M et al. World journal of emergency surgery : WJES 2020. doi:…", "locator": "Statements 3.1–3.5; 4.1–4.2; 6.1–6.9", "url": "https://…", "license": "CC BY"}],
    "conflicts": ["조기 수술 시점: 원문 진단 후 48–72시간 vs WSES 입원 7일·증상 10일 이내 가능한 빨리."],
    "updated_at": "2026-11-10"
  }
]
```

- 경고 문구는 응답에 넣지 않는다(`review_label`·`review_notice`·`notice`·`dose_notice`·`unreviewed` 없음). 서비스 자체가 의료진 참고용이므로 화면에 미검수·참고용 경고를 반복하지 않고, 각 내용은 출처(`sources`)로 근거를 밝힌다. `review_status`(`draft`/`approved`)는 데이터로만 남는다. 미검수(`draft`) 근거를 쓸지는 `AX_G_EVIDENCE_INCLUDE_UNREVIEWED`로 정한다(코드 기본 false, 배포 `.env.example`은 true).
- `text`는 지침·허가사항의 일반 근거이며 환자 기록이 아니다. 근거 문단의 용량 수치를 이 환자 용량으로 쓰지 않는다(환자 값에 적용한 용량은 `dose_assessments`).
- 출처 표시: 화면에는 `sources[]`의 `title`(있으면 `citation`), `url`, `license`를 함께 보여준다. CC BY 출처는 저자·제목·라이선스 표시가 이용 조건이다. `license`는 `CC0 1.0`, `CC BY`(버전이 확인되면 `CC BY 4.0` 등), `Public domain (U.S. Government work)`, `Copyright-free (U.S. Government work)` 중 하나다.
- 출처 우선순위: **국내 > 미국 > 유럽 > 그 밖**. 각 근거에 `region`(`KR`·`US`·`EU`·`INTL`)과 `region_label`(국내·미국·유럽·국제·기타)이 있고 `sources[]`도 국내 출처부터 정렬한다. 같은 주제에 국내 기준이 있으면 국외 근거는 `priority: "reference"`와 `superseded_by`(대체한 국내 주장 ID)가 붙고 목록 뒤로 간다 — 화면에서는 "참고(국외 기준)"로 표시한다. 국내 기준이 없는 주제의 국외 근거는 `priority: "primary"`.
- 출처 표시 가능 자료만 사용(`AX_G_ATTRIBUTED_SOURCES_ONLY`, 기본 true): 라이선스·제목이 없는 출처의 근거·규칙은 인용하지 않는다. 같은 형식의 `sources`가 추천 응답의 `assessment.items[].sources`에도 들어간다.

## 스트리밍 채팅

`POST /internal/v1/clinical-chat/stream`은 채팅과 동일한 요청 본문을 받는다. `Accept: text/event-stream`을 사용한다. 성공 시 `Content-Type: text/event-stream`으로 다음 SSE 이벤트를 전송한다.

```text
event: meta
data: {"source":"EMR 데이터베이스","emr_updated_at":"2026-09-22T10:30:00+09:00"}

event: delta
data: {"text":"최근 혈당을 "}

event: delta
data: {"text":"확인하세요."}

event: done
data: {}
```

근거 인용이 있으면 `meta`에 채팅 응답과 같은 형식의 `evidence` 목록이 함께 들어간다(없으면 생략).

`meta`는 첫 이벤트이며 `delta.text`를 수신 순서대로 이어 붙인다. 정상 완료는 `done`으로 표시한다. 첫 텍스트 전 오류는 아래 JSON 오류를 반환한다. 전송 시작 후 실패하면 HTTP 200 상태에서 `error` 이벤트(`code`, `message`)를 보내고 `done`은 보내지 않는다. 부분 답변을 완료된 답변으로 취급하지 않는다.

## 스트리밍 EMR 준비

`POST /internal/v1/patient-context/stream`은 `patient-context`와 같은 요청 본문을 받는다. 구조화(JSON) 요약을 끝까지 기다리지 않고 요약 문장을 생성되는 대로 전송하므로 첫 글자가 빨리 표시된다. 요약은 `sections` 구조가 아닌 텍스트이며, 한 줄 요약 다음에 추가 요약 4~5줄을 요청한다. 기록이 부족하면 줄 수가 줄어들 수 있다. `delta.text`를 이어 붙여 표시한다.

```text
event: meta
data: {"patient_id":"pid-4356","status":"ready","source":"EMR 데이터베이스","emr_updated_at":"2026-09-22T10:30:00+09:00"}

event: delta
data: {"text":"최근 혈당은 "}

event: done
data: {}
```

- 채팅용 문맥은 `meta` 전송 시점에 이미 준비되어 있으므로, 요약 스트리밍 중에도 채팅을 요청할 수 있다.
- EMR 검증 실패(`PATIENT_MISMATCH`, `EMR_UNAVAILABLE`)와 첫 텍스트 전 AI 실패(`AI_PROVIDER_UNAVAILABLE`)는 JSON 오류로 반환하며 문맥을 준비하지 않는다.
- 전송 시작 후 실패하면 `error` 이벤트를 보내고 준비된 문맥을 삭제한다. 이후 채팅은 `PATIENT_CONTEXT_NOT_READY`가 되므로 다시 준비해야 한다.

## 독립 EMR 요약

`POST /internal/v1/emr-summary`는 환자 EMR 준비와 같은 요청 본문을 받아 한 번 요약한다. 사전 준비가 필요 없고 채팅용 문맥을 저장하거나 변경하지 않는다. 성공 응답은 채팅과 동일한 `answer`, `source`, `emr_updated_at` 구조다.

`POST /internal/v1/emr-summary/stream`은 같은 요약을 SSE로 전송한다. 이벤트 형식과 오류 처리는 스트리밍 채팅과 같고(`meta`는 `source`, `emr_updated_at`), 채팅용 문맥은 저장하지 않는다.

## 추천 진단·검사

`POST /internal/v1/clinical-recommendations`는 `patient-context`로 준비된 환자에 대해 최신화 말뭉치의 진단·검사·협진·추적 규칙을 EMR 사실에 적용하고, 그 결과와 근거 문단으로 의심 진단·추천 검사(혈액·소변, 영상, 내시경, 조직, 기능 검사)·협진 의견을 만든다. `AX_G_CLINICAL_DOSES_ENABLED=true`이면 용량 규칙(3단계)도 적용해 `assessment.dose_assessments`를 함께 반환한다(아래 「약물 용량 참고」).

요청:

```json
{
  "patient_id": "pid-4356",
  "clinical_note": "2달 전부터 복통과 배변습관 변화, 혈변, 체중 4kg 감소",
  "findings": {"hematochezia": true, "prior_hbv_triple_panel": false},
  "question": "대장 쪽 검사가 필요한지 정리해줘",
  "dose_drugs": ["레보티록신"]
}
```

- 네 필드 모두 선택이다. 빈 요청이면 EMR 사실만으로 평가한다.
- `dose_drugs`(최대 20개, 3단계 사용 시): 용량을 계산할 약. 성분 ID(`levothyroxine`)·약효군 ID(`basal_insulin`)·한글/영문 성분명(`레보티록신`)을 받는다. 현재 처방 약은 넣지 않아도 자동 포함된다. 용량 규칙이 있는 약 목록은 `GET /internal/v1/clinical-recommendations/findings`의 `dose_drugs`.
- `clinical_note`(최대 4,000자): 의료진이 입력한 증상·소견. AI가 소견 키로 추출해 규칙에 반영하고(`assessment.findings_used`로 반환), 추천 문장 생성에도 쓴다. **식별정보를 넣지 않는다** — 매퍼를 거치지 않고 Provider로 전달된다.
- `findings`: 소견 키 → `true`(있음)/`false`(없음). 메모 추출 결과보다 우선한다. 목록에 없는 키는 무시하고 `assessment.unknown_findings`로 돌려준다. 키 목록은 `GET /internal/v1/clinical-recommendations/findings`.

응답:

```json
{
  "answer": "67세 남성 2형 당뇨 환자로, 혈변·체중 감소를 동반한 배변습관 변화가 있어 대장 병변 평가가 우선입니다.\n의심 진단\n• …\n추천 검사\n• 대장 출혈·종양 의심 → 대장내시경(내시경검사): … [근거 1: 일본소화기병학회 IBS 지침 2020]\n…",
  "source": "EMR 데이터베이스",
  "emr_updated_at": "2026-09-04T09:00:00+09:00",
  "assessment": {
    "items": [
      {
        "record_id": "352-C01-K2", "claim_id": "352-C01", "kind": "referral", "status": "met",
        "topic": "진단 기준과 경고 증상에 따른 대장내시경", "chapter": "352",
        "referral_to": "소화기내과(대장내시경)",
        "matched": ["finding:chronic_abdominal_pain_with_bowel_change", "finding:hematochezia"],
        "missing": [],
        "review_status": "draft",
        "sources": [{"id": "PAPER-JSGE-IBS-2020", "title": "…", "locator": "…", "url": "https://…"}],
        "evidence_refs": [1]
      },
      {
        "record_id": "335-C03-K1", "kind": "workup", "status": "met", "topic": "CKD 검사 대상과 검사 항목",
        "tests": [{"id": "UACR", "name": "소변 알부민/크레아티닌 비", "category": "검체검사(혈액·소변 등)", "reason": "알부민뇨"}],
        "matched": ["나이 > 60 (환자 값 67)", "진단 기록 E11 (E11.9)"], "strength": "VA/DoD Rec 1 (Weak for)",
        "review_status": "draft", "sources": ["…"], "evidence_refs": [2]
      }
    ],
    "lab_interpretations": [
      {"lab": "Na", "name": "나트륨", "value": 131.0, "unit": "mmol/L", "band": "경도", "context": "저나트륨혈증 생화학적 중증도", "review_status": "draft", "evidence_refs": [3]}
    ],
    "findings_used": {"chronic_abdominal_pain_with_bowel_change": true, "hematochezia": true},
    "unknown_findings": [],
    "mapping_issues": [{"kind": "lab_unmapped", "code": "HbA1c", "name": "당화혈색소", "detail": "사이트 매핑에 없는 검사 코드", "suggestion": "HbA1c"}]
  },
  "evidence": [{"n": 1, "chunk_id": "352-C01-chunk", "review_status": "draft", "sources": ["…"], "…": "채팅 evidence와 같은 형식"}]
}
```

`assessment.medication_alerts`(DUR 사용 시): 현재 처방(`medicineDtailList[].medCode` = 9자리 보험 제품코드)을 심평원 DUR 의약품 목록(PostgreSQL `dur` 스키마의 활성 데이터셋)과 대조한 결과. DB나 활성 데이터셋이 없으면 필드를 생략한다.

```json
"medication_alerts": {
  "alerts": [
    {"type": "병용금기", "status": "met", "products": [{"med_code": "111111111", "med_name": "클래리정", "ingredient_name": "clarithromycin"}, {"med_code": "222222222", "med_name": "심바정", "ingredient_name": "simvastatin"}],
     "detail": "근육병증의 위험 증가 가능", "notice": {"no": "2020", "date": "2020-01-01"}},
    {"type": "임부금기", "status": "possible", "grade": "1등급", "missing": [{"key": "pregnancy", "label": "임신 여부(가임기 여성)"}], "…": "…"}
  ],
  "unmapped": [{"med_code": "999999999", "med_name": "…", "detail": "DUR 목록에 없는 제품코드 — DUR 점검 제외"}],
  "source": {"id": "KR-HIRA-DUR-2026-06", "title": "건강보험심사평가원 의약품안전사용서비스(DUR) 의약품 목록(2026.6) — 공공데이터포털", "license": "KOGL Type 1 (공공누리 제1유형 — 출처표시)", "region": "KR", "url": "https://www.data.go.kr/data/15127983/fileData.do"}
}
```

- `type`: `병용금기`(현재 처방 두 약의 성분 조합), `연령금기`(나이 조건 충족, 나이 모르면 판정 안 함), `임부금기`(임신 소견 입력 시 `met`, 임신 여부를 모르는 15–49세 여성은 `possible`), `노인주의`(65세 이상 — 기준 나이는 공고 원문 대조 필요).
- `condition_note`: 목록의 비고(예: "48시간 이내 병용금기", "1주에 MTX 15mg 이상 투여시 병용금기")를 그대로 보여준다 — 비고의 용량 조건은 평가하지 않는다. DUR 용량주의·투여기간주의 목록은 아직 원본 파일이 없어 쓰지 않는다.
- 화면에는 출처(`source.title`, 공공누리 제1유형)를 함께 표시한다. 답변 본문에서는 `[출처: 심평원 DUR 목록]`으로 표시된다. 처방 변경을 지시하지 않는다.
- 채팅에서도 질문에 `병용`, `금기`, `상호작용`, `DUR`, `처방 점검`, `복용 약` 등이 있으면 같은 점검 결과를 답변에 반영한다.

### 약물 용량 참고 (3단계, `AX_G_CLINICAL_DOSES_ENABLED=true`)

`assessment.dose_assessments`: 말뭉치 용량 규칙(`dose_rules`)을 현재 처방 약(성분 또는 약효군이 일치)과 `dose_drugs`에 적용한 결과. 3단계를 끄면 필드가 없고, 켰는데 해당 약이 없으면 빈 목록이다. 확인할 수 없는 `dose_drugs`는 `unmatched_dose_drugs`로 돌려준다.

```json
"dose_assessments": [
  {
    "record_id": "DR-PREGABALIN-DPN", "kind": "dose", "topic": "당뇨병성 말초신경병증 통증 약물",
    "drug": {"id": "pregabalin", "name": "프레가발린", "type": "ingredient"},
    "anchor": ["prescribed"], "indication": "당뇨병성 말초신경병증 통증",
    "prescriptions": [{
      "med_code": "…", "med_name": "리리카캡슐150밀리그램(프레가발린)", "dose": {"dosage": 150, "unit": "mg", "frequency": 2},
      "daily_mg": 300.0, "daily_basis": "1회 150 mg × 2회(frequency를 1일 횟수로 봄, interval 미기재)",
      "check": {"limit_daily_mg": 75.0, "limit_basis": "CrCl 15 이상 30 미만 mL/min 구간 1일 최대", "status": "exceeds"}
    }],
    "renal": {
      "param": "CrCl", "label": "크레아티닌 청소율(Cockcroft-Gault)", "value": 29.1, "unit": "mL/min",
      "test_date": "2026-09-04", "basis": "Cockcroft-Gault(실제 체중)",
      "band": {"range": "15 이상 30 미만 mL/min", "dose": [{"key": "daily_mg", "label": "1일 용량(mg)", "value": {"start": "25–50", "max": 75}}]}
    },
    "missing": [], "alert": "high",
    "alert_note": "미국 허가 기준 참고 범위 — 처방 용량으로 표시하지 않음",
    "review_status": "draft", "region": "US", "region_label": "미국", "priority": "primary",
    "sources": ["…"], "evidence_refs": [4]
  },
  {
    "record_id": "DR-LEVOTHYROXINE", "drug": {"id": "levothyroxine", "name": "레보티록신", "type": "ingredient"}, "anchor": ["requested"],
    "calculations": [{"name": "full_replacement_mcg_day", "formula": "weight_kg × 1.6", "unit": "mcg/day", "inputs": {"weight_kg": 58.0}, "value": 92.8}],
    "rules": [{"status": "met", "action": "1.6 mcg/kg/일보다 낮게 시작, 6–8주마다 조정", "alert": "caution", "matched": ["나이 >= 65 (환자 값 72)"]}],
    "reference": [{"key": "administration", "label": "투여 방법", "value": ["공복, 아침식사 30–60분 전", "…"]}],
    "…": "…"
  }
]
```

| 필드 | 의미 |
|---|---|
| `anchor` | `prescribed`(현재 처방 약), `requested`(`dose_drugs`로 요청) |
| `calculations` | 체중 등 이 환자 값(`inputs`)을 규칙 공식(`formula`)에 넣은 참고 계산값 `value` 또는 범위 `range`. 입력이 없으면 값 대신 `missing` |
| `renal` | 규칙이 쓰는 신기능 지표(`eGFR`·`CrCl`)의 환자 값·근거(`basis`: 검사실 보고값 또는 계산식)·검사일, 그 값이 속한 구간 `band`. CrCl을 eGFR로 대신하지 않는다. 이상체중 기준 규칙에 실제 체중 CrCl을 쓰면 `method_note` |
| `prescriptions[].daily_mg` | 처방 `dosage`×`frequency`(단위가 mg·g·mcg이고 `interval`이 없거나 1일일 때만). 주 단위·정·U 등은 계산하지 않음 |
| `prescriptions[].check` | 구간에 숫자 1일 최대가 있을 때 처방 1일 용량과 대조(`exceeds`/`within`) |
| `rules` | 용량 조건(시작 금지·감량·금기 등). `met` 충족, `possible`은 `missing` 확인 시 해당 |
| `reference` | 규칙의 표준·시작·최대 용량, 증량, 감시, 투여 방법 등 원문 참고 |
| `not_applied` | 18세 미만 등 성인 기준 계산을 적용하지 않은 이유. 나이를 모르면 `missing`에 `age` |
| `alert` | 충족된 경고 중 가장 높은 단계(`contraindicated` > `high` > `caution`/`medium` > `info`), 1일 최대 초과는 `high` |
| `region` / `region_label` | 기준이 된 출처의 나라(현재 용량 규칙은 전부 국외 — 미국 FDA 라벨·국제 지침) |
| `alert_note` | 말뭉치 규칙의 원문 주석(병용 약 조정, 병원 프로토콜 우선 등) |

- 값은 출처(허가사항·지침)의 공식·구간을 이 환자 입력값에 적용한 결과이며 처방 지시가 아니다. 답변 본문은 `용량 참고` 항목에 출처명(`region_label`·출처 제목, 예: "미국 FDA 허가사항 기준")과 입력값을 함께 쓰고, 처방하라는 표현을 쓰지 않는다.
- 현재 말뭉치의 용량 규칙은 전부 `review_status: draft`이고 출처가 국외(미국 FDA 라벨·국제 지침)다. `AX_G_EVIDENCE_INCLUDE_UNREVIEWED=true`일 때만 쓰인다.
- 채팅에서도 질문에 `용량`, `투여량`, `감량`, `증량`, `mg/kg`, `신기능 조정` 등이 있으면 현재 처방 약의 용량 결과를 답변에 반영한다.

`assessment.items` 항목:

| 필드 | 의미 |
|---|---|
| `kind` | `diagnosis`(진단 기준), `workup`(검사), `referral`(협진·의뢰), `monitoring`(추적 검사) |
| `status` | `met`: 규칙 조건 충족. `possible`: 일부 조건(진단 기록·처방·검사값·입력 소견)이 맞고 `missing` 정보가 확인되지 않음 — 조건부 제안 |
| `matched` / `missing` | 이 환자에게 맞은 조건 / 확인하면 판정이 바뀌는 정보(`kind`: `value`·`finding`·`event` 등) |
| `tests` | 검사 개념 `id`, 한국어 `name`, `category`(검체검사·영상검사·내시경검사·조직검사·기능·진찰 검사·점수·위험도 평가 등), `reason` |
| `diagnosis` / `already_recorded` | 진단 기준 충족 시 진단명·KCD, 이미 EMR에 기록된 진단인지 |
| `review_status`, `sources`, `conflicts`, `strength` | 검수 상태(데이터), 출처(제목·위치·링크·라이선스·지역, 국내 우선 정렬), 지침 간 차이, 권고 강도 |
| `region`, `region_label` | 규칙 출처의 지역(국내 > 미국 > 유럽 > 그 밖) |
| `priority`, `superseded_by` | `primary`: 권고로 쓰는 항목. `reference`: 같은 주제에 국내 기준(`superseded_by`)이 있는 국외 기준 — 권고가 아니라 참고로만 표시 |
| `evidence_refs` | `evidence[].n` 번호. 답변의 `[근거 n: 출처명]`과 같다 |

- 정렬: `primary` → `reference`, 그 안에서 충족(`met`) → 조건부(`possible`), 같은 상태 안에서는 진단 기록·처방·검사값에 근거한 항목이 나이만 맞는 선별 항목보다 앞선다. `possible`은 최대 12개.
- 답변 본문 표시: `[근거 n: 출처명]`(근거 문단), `[출처: 출처명]`(근거 문단 없이 규칙 출처만 있는 항목), `[말뭉치 규칙]`(출처 제목도 없는 항목). 검수 여부(`미검수`)나 참고용 경고는 본문에도 응답 필드에도 넣지 않는다. 기본 설정(`AX_G_ATTRIBUTED_SOURCES_ONLY=true`)에서는 말뭉치 규칙·근거로 뒷받침되는 제안만 쓴다. 이 설정을 끄면 말뭉치에 없는 AI 제안이 `[AI 일반 지식·출처 없음]` 표시와 함께 들어갈 수 있다.
- 규칙은 근거 인용과 같은 게이트를 지난다(A등급 출처 전부·원문 확인·폐기 제외). 미검수 규칙은 `AX_G_EVIDENCE_INCLUDE_UNREVIEWED=true`일 때만 쓴다. 쓸 규칙이 없으면 `503 RECOMMENDATION_UNAVAILABLE`.

`POST /internal/v1/clinical-recommendations/stream`은 같은 요청을 받아 SSE로 보낸다. `meta`에 `source`, `emr_updated_at`, `assessment`, `evidence`가 먼저 오고 이후 `delta`/`done`은 스트리밍 채팅과 같다. 소견 추출과 규칙 평가는 첫 이벤트 전에 끝난다.

채팅에서도 질문에 `추천`, `감별`, `의심`, `어떤 검사`, `필요한 검사`, `추가 검사`, `협진`, `의뢰` 등이 있으면 같은 규칙 평가 결과(소견 입력 없이 EMR 사실만)를 답변에 반영하고, 관련 근거를 `evidence` 앞쪽에 최대 4개 넣는다.

## 오류

오류 응답은 JSON 형식이다.

```json
{"code": "EMR_UNAVAILABLE", "message": "환자 EMR 정보를 준비할 수 없습니다."}
```

| HTTP 상태 | code | 조건 |
|---:|---|---|
| 400 | `INVALID_REQUEST` | 필수 필드 누락, 형식 오류 또는 허용되지 않은 필드 |
| 403 | `PATIENT_MISMATCH` | 요청과 EMR의 환자 ID 불일치 |
| 404 | `PATIENT_CONTEXT_NOT_READY` | 준비되지 않았거나 보관 기간(기본 12시간)이 지난 환자의 채팅 요청 |
| 502 | `EMR_UNAVAILABLE` | 전달된 EMR 검증 실패 |
| 502 | `AI_PROVIDER_UNAVAILABLE` | AI 설정 부재, AI 호출 또는 응답 검증 실패 |
| 503 | `RECOMMENDATION_UNAVAILABLE` | 추천 규칙이 꺼져 있거나 사용할 수 있는 규칙이 없음(미검수 미허용 등) |
| 500 | `INTERNAL_ERROR` | 예상하지 못한 서버 오류 |

`GET /health` 성공 응답은 `{"status":"ok"}`다.
