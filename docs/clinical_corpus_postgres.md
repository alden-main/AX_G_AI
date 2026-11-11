# 임상 말뭉치 PostgreSQL·벡터 검색 설계

상태: 구현(적재기·하이브리드 검색기·compose 구성) · 2026-11-10

## 목표

- 최신화 말뭉치(근거 문단·규칙·검사/약제 개념)를 여러 서버 인스턴스가 함께 쓰는 DB에 둔다.
- 재배포 없이 말뭉치를 교체하고, 문제가 생기면 이전 릴리스로 되돌린다.
- 어휘 검색과 의미(벡터) 검색을 합친 하이브리드 검색으로 근거 문단을 찾는다.
- 의료진 검수 피드백과 근거 노출 기록을 남긴다(환자 정보 없이).

## 데이터 흐름

```
text_data/claims·mappings (YAML, 작성 원천 — Git 관리)
   │ build_corpus.py
   ▼
text_data/corpus/*.jsonl (생성물)
   │ 적재기(load_corpus) — 새 release_id 로 전부 적재 → 검증 → 활성 전환(한 트랜잭션)
   ▼
PostgreSQL (schema corpus)                     임베딩 서버(bge-m3 등)
 ├ evidence_chunk  ── body_sha256 ──▶ evidence_embedding_1024 ◀── 바뀐 본문만 계산
 ├ corpus_record / concept / concept_usage
 ├ site_code_map (병원 관리, 릴리스와 독립)
 └ review_feedback / evidence_exposure
   │ citable_evidence 뷰(인용 게이트 단일 지점)
   ▼
PostgresEvidenceRetriever (EvidenceRetriever 구현) → ClinicalChatService
```

DB는 서빙 저장소다. 내용 수정은 지금처럼 YAML에서 하고 빌드한다. DB에서 직접 고치면 출처 검증(A등급·원문 위치)을 우회하게 된다.

스키마: [schema.sql](../src/ax_g_ai/corpus_store/schema.sql) — `pgvector/pgvector:0.8.7-pg16` 임시 컨테이너에서 생성·게이트·어휘 검색·벡터 검색을 확인했다.

## 핵심 설계 결정

| 항목 | 결정 | 이유 |
|---|---|---|
| 릴리스 | 적재마다 `release_id`, 활성 릴리스 1개(부분 유일 인덱스) | 원자적 교체·즉시 롤백. 노출 기록이 어느 릴리스 근거였는지 남는다 |
| 인용 게이트 | `citable_evidence` 뷰 한곳 | 런타임 `is_citable()`과 같은 규칙(A등급 출처 전부, 원문 확인, 폐기·확인불가 제외). 검수 포함 여부만 질의 인자 |
| 한국어 어휘 검색 | 앱이 만든 토큰(한글 2-gram + 영문 단어)을 `tsvector('simple')`로 색인 | PostgreSQL에 한국어 파서가 없다. 확장 설치 없이 어디서나 동작하고 지금 메모리 검색과 결과가 같다. 관리형 DB가 `pg_bigm`을 지원하면 대안 |
| 벡터 | pgvector, HNSW + 코사인, 모델별 테이블 | 차원이 모델마다 다르다. 모델 교체 시 새 테이블만 채우면 된다 |
| 임베딩 재사용 | `body_sha256` 키 | 릴리스가 바뀌어도 본문이 같으면 다시 계산하지 않는다 |
| 순위 결합 | 어휘·벡터 각 상위 30건을 RRF(1/(60+순위))로 합산 + 개념 가산 | 점수 척도가 다른 두 검색을 안전하게 합친다 |
| 최소 관련도 | 어휘 점수 또는 벡터 유사도 하한을 함께 적용 | 벡터 검색은 항상 무언가를 돌려주므로, 하한이 없으면 무관한 근거가 인용된다 |

## 임베딩 모델: nlpai-lab/KURE-v1 (CPU, TEI)

| 항목 | 내용 |
|---|---|
| 모델 | `nlpai-lab/KURE-v1` — BAAI/bge-m3를 한국어 질의·문서 약 200만 건으로 미세조정, MIT, 1024차원, 최대 8,192토큰, CLS 풀링, 질의 접두어 없음 |
| 서빙 | Hugging Face Text Embeddings Inference CPU 이미지 1.9.4(Apache-2.0), compose `embedding` 서비스, 내부망 전용 |
| 자원 | 메모리 6GB 한도(4GB에서 모델 적재 중 OOM 확인), CPU 4코어, 모델 캐시 약 2.3GB |
| 저장 | `evidence_embedding_1024`, `model_id='kure-v1'`, 키 = 임베딩 입력(제목 경로 + 본문) SHA-256 |

질문 문장은 내부망 안의 임베딩 서버로만 간다. 모델을 바꾸면 `AX_G_EMBEDDING_MODEL`만 바꾸고 적재기를 다시 돌리면 그 모델 벡터가 채워진다(1024차원 모델끼리는 같은 테이블).

### 보정 결과 (2026-11-10, 실제 말뭉치 285문단, 개발 PC CPU)

- 적재 시 전체 임베딩 약 2분 20초 → 사전 계산 파일 사용 시 약 6초(표본 일치 확인 포함), 임베딩 서버 없이 1.4초. 같은 말뭉치 재적재는 0건 계산.
- 사전 계산 파일: 285문단, 1.6MB(base64 float32). 키는 임베딩 입력 SHA-256이라 바뀐 문단만 다시 계산된다. 현재 파일은 같은 모델의 sentence-transformers(CPU, fp32)로 만들었고, 운영 TEI와의 일치는 적재기의 표본 확인이 매번 검사한다.
- 질의 임베딩 약 0.1초(첫 질의 2~3초 워밍업).
- 주제 질문 18개: 의미 검색 1위가 정답 장인 경우 12개. 정답 1위 유사도 0.54~0.82.
- 일반 요청 10개(요약·SOAP·날씨·처방 내역 등): 최고 유사도 0.39~0.56 → **의미 검색 단독 인용 하한 0.6**(`AX_G_EVIDENCE_SIMILARITY_FLOOR`). 무관한 인용을 막는 쪽으로 정했으며, 0.54 수준의 바꿔 말한 질문("루게릭병 호흡 부전", "쓸개에 돌")은 의미 검색만으로는 인용되지 않는다.
- 영어로만 된 질문은 약했다("uncomplicated diverticulitis antibiotics" → 오답 0.48). 한영 혼용 질문은 정상. 영어 질문 비중이 크면 bge-m3 원본과 비교 평가가 필요하다.
- 효과 예: "철 과부하 환자 피 뽑는 치료 간격" — 어휘 검색은 진단 문단만, 하이브리드는 사혈 감시 문단(428-C03)을 1위로.

## 검색기 (구현: [retriever.py](../src/ax_g_ai/corpus_store/retriever.py))

[clinical_evidence.py](../src/ax_g_ai/services/clinical_evidence.py)의 `EvidenceRetriever.search(question, limit)` 구현이라 채팅 서비스·API는 그대로다.

```
PostgresEvidenceRetriever
  시작: 활성 릴리스의 citable_evidence(SQL 게이트) + 개념 사용 정보 → 메모리 어휘 색인(파이썬 게이트 재확인)
  질의: 어휘 점수(BM25 + 개념 가산, 기존과 동일) ∪ pgvector 코사인 상위 30
        인용 = 어휘 하한 통과 또는 유사도 ≥ 0.6, 순서 = RRF(1/(60+순위)) 합
  갱신: 60초마다 활성 릴리스 번호 확인, 바뀌면 색인 재적재
  장애: 임베딩 서버·DB 오류 → 마지막 어휘 색인으로 응답. 시작 시 DB 불가 → 파일 색인(backend=files)
```

- 검색은 API에서 스레드풀로 실행해 이벤트 루프를 막지 않는다.
- DB URL·비밀번호는 로그에 쓰지 않는다(설정은 `SecretStr`, 실패 로그는 오류 종류만).

## 적재기 (구현: [loader.py](../src/ax_g_ai/corpus_store/loader.py), `ax-g-ai-load-corpus`)

1. 스키마 없으면 생성(`corpus.schema_version`).
2. 말뭉치 파일 SHA-256으로 매니페스트 → 같은 매니페스트 릴리스가 있으면 재사용.
3. 새 릴리스를 비활성으로 적재(출처·문단·규칙·개념·코드·사용처) → 건수 검증.
4. 벡터 채우기: 사전 계산 파일(`corpus/embeddings/<model_id>.jsonl`, [precomputed.py](../src/ax_g_ai/corpus_store/precomputed.py)) → 임베딩 서버로 표본 3개를 다시 계산해 코사인 ≥ 0.99면 파일 벡터 사용, 아니면 파일 무시 → 남은 문단만 서버로 계산(실패해도 활성화는 진행, 다음 적재 때 채움).
5. 활성 전환(한 트랜잭션) → 최근 5개 릴리스만 보관. 롤백은 `--activate <release_id>`.

## 함께 옮길 수 있는 것

- `corpus_record`: 2단계 이후 규칙 평가(검사 해석·투약 경고)용. 조건식은 `payload` JSON 그대로 두고 평가는 앱(`clinical_corpus.evaluate`)에서 한다. SQL로 조건식을 평가하지 않는다.
- `concept`·`concept_code`·`concept_usage`: "이 검사·약을 쓰는 규칙" 조회, LOINC·RxCUI 역조회.
- `site_code_map`: 병원별 검사·약품 코드 매핑. `status=reviewed`만 런타임이 읽는다.
- `review_feedback`: 화면의 "동의/이의/원문과 다름" → 검수 대기열. 반영은 YAML `review.status` 변경 → 빌드 → 새 릴리스.

## 2단계: 규칙 평가(추천 진단·검사) — 구현 2026-11-11

`corpus_record`의 진단·검사·협진·추적·검사값 범주 규칙을 [clinical_recommendation.py](../src/ax_g_ai/services/clinical_recommendation.py)가 앱에서 평가한다([rules.py](../src/ax_g_ai/corpus_store/rules.py)가 활성 릴리스의 규칙·개념·인용 가능 문단을 읽고, DB가 없으면 `text_data/corpus` 파일). 게이트는 근거와 같다(`is_citable`). 미검수 규칙은 `AX_G_EVIDENCE_INCLUDE_UNREVIEWED`일 때만, 항상 `의료진 미검수` 표시와 출처를 붙인다. `treatment`는 읽지 않는다. 규칙의 근거 문단은 검색이 아니라 같은 `claim_id`로 직접 연결한다. API는 [bridge-ai-api-spec.md](contracts/bridge-ai-api-spec.md#추천-진단검사).

## 3단계: 약물 용량 참고 — 구현 2026-11-11

같은 `corpus_record`의 `dose_rules.jsonl`을 같은 게이트로 읽어 `AX_G_CLINICAL_DOSES_ENABLED=true`일 때 평가한다. 계산은 모두 앱에서 결정적으로 한다([clinical_dosing.py](../src/ax_g_ai/services/clinical_dosing.py)): 공식(`calculation`)은 숫자·변수·사칙연산·`min`/`max`·범위 `[a, b]`만 허용하는 해석기로 계산하고(`eval` 없음, 모르는 문법은 그 규칙을 건너뜀), 신기능 구간(`bands`)은 `renal_param`(eGFR·CrCl)의 환자 값으로 고른다. 공식에 넣는 체중·나이·신기능 값은 2단계와 같은 `emr_mapping` 사실이며, 없으면 추정하지 않고 `missing`으로 돌려준다. API는 [bridge-ai-api-spec.md](contracts/bridge-ai-api-spec.md#약물-용량-참고-3단계-ax_g_clinical_doses_enabledtrue).

## DUR 스키마 (`dur`, 구현: [dur_store](../src/ax_g_ai/dur_store/))

심평원 DUR 의약품 목록도 같은 PostgreSQL에 둔다. 말뭉치와 수명 주기(갱신 주기·출처·롤백 단위)가 달라 스키마를 나눴다([schema.sql](../src/ax_g_ai/dur_store/schema.sql)).

| 표 | 내용 |
|---|---|
| `dur.dataset` | 공고 시점별 데이터셋(출처·라이선스·원본 CSV SHA-256·표별 건수·`content_sha256`). 활성은 하나 |
| `dur.product` | 9자리 보험 제품코드 → 성분코드(EMR `medCode` 조회 키) |
| `dur.pair_rule` | 병용금기 성분 쌍(`ingredient_a < ingredient_b`로 한 번만), 비고는 그대로 |
| `dur.age_rule` · `dur.pregnancy_rule` · `dur.elderly_rule` | 연령금기·임부금기·노인주의(일반·해열진통소염제) — 성분 단위 |

```text
dur/*.csv (공공데이터포털, cp949, 약 87만 행)
  → ax-g-ai-build-dur  : 성분 단위 중복 제거 → data/dur/<표>.jsonl + manifest.json (약 7만 행, 12MB, 이미지에 포함)
  → ax-g-ai-load-dur   : 스냅샷 검증(SHA-256·건수) → 같은 content_sha256이면 재사용, 아니면 COPY로 새 데이터셋 → 건수 검증 → 활성 전환 → 최근 3개 보관
  → API 시작 시        : 활성 데이터셋을 메모리로 읽어 처방마다 DB 왕복 없이 점검(DurChecker)
```

- 운영 DB와 같은 접속 정보(`AX_G_CORPUS_DATABASE_URL`)를 쓴다. 롤백은 `ax-g-ai-load-dur --activate <dataset_id>` 후 재시작.
- 기존 SQLite(`data/dur.sqlite`)와 표 5개 모두 행 단위로 같음을 확인했다(2026-11-11, 제품 20,106 · 병용 23,639 · 연령 607 · 임부 4,011 · 노인 245).
- DUR은 코드 일치 점검이라 벡터화하지 않는다(유사 이름으로 금기를 판정하면 안 된다).
- 용량주의·투여기간주의·1일 최대투여량 목록은 원본 파일을 받으면 같은 형태(`dataset_id` + 성분코드)의 표를 추가한다.

## 남은 일

- 의료진 검수: 현재 규칙·문단 전부 `draft`. 검수 완료분부터 `approved`로 바꿔 빌드한다(응답·답변에는 검수 경고를 표시하지 않으며 `review_status`만 바뀐다).
- `finding` 조건(약 170개 키)은 EMR에 없는 증상·소견이라 의료진 입력·메모 추출로만 채워진다. 검사값 추이 변수(`SCr_rise_mg_dl_48h`, `eGFR_change_pct` 등)는 아직 계산하지 않아 `missing`으로 나온다.
- 병원 코드 매핑(`text_data/mappings/sites/<병원>.json`) 작성·검수 — 없으면 LOINC 코드가 아닌 병원 검사 코드는 규칙 값으로 쓰이지 않는다(`mapping_issues`).
- 규칙 재적재: 근거 검색기와 달리 시작 시 한 번만 읽는다.

- 노출 기록(`evidence_exposure`) 적재와 검수 피드백(`review_feedback`) API — 테이블만 있음.
- 질문 평가 세트를 늘려 하한·가중치 재보정, 필요하면 bge-m3 원본·재순위(reranker) 비교.
- 운영 서버(x86_64)에서 TEI 이미지로 같은 측정 반복 — 개발 PC는 Docker 메모리 4GB 한도로 TEI 대신 같은 모델의 sentence-transformers로 측정했다.
- 노출 기록 보관 기간 결정.
