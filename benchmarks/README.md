# 합성 EMR 벤치마크 (개발용)

임상 전문가는 아직 이 자료를 검토하지 않았습니다. 정답, 중요도, 금지 주장, 중대 오류 정의는 모두 초안입니다. 이 결과로 성능이나 의료 안전성을 입증할 수 없습니다. 독립적인 성능 추정에는 별도의 비공개 의료진 검토 사례가 필요합니다.

## 구성

| 묶음 | 파일 | 규모 | 평가 방식 |
|---|---|---|---|
| core | `data/inputs.jsonl`, `data/gold.jsonl` | 합성 환자 100명 × 요약·채팅 = 200 업무 | 사람이 판정(평가 모델 제안은 참고용) |
| recommendation | `data/recommendation_inputs.jsonl`, `data/recommendation_gold.jsonl` | 10개 시나리오 × 3 split = 30건 | DUR·임신 판정 구조를 자동 대조 |

두 묶음 모두 `python -m benchmarks.generate`가 고정 시드로 만듭니다. 손으로 고치지 말고 생성기를 고친 뒤 다시 생성하세요. `data/manifest.json`에 파일 해시, 시드, 정답 계산에 쓴 DUR 스냅샷(`dur_snapshot`)이 기록됩니다.

### core 데이터 (schema 2)

- **EMR 계약 형식을 따릅니다.** `bloodPressureList`, `bloodSugarList`, `oxygenSaturationList`, `glycatedHemoglobinList`, `labResultList`(`examCode` 포함), `diagnosisList`(점 없는 KCD 코드)를 씁니다. `medicineDtailList`에는 9자리 DUR 제품코드 `medCode`, 용량, 횟수, 처방 일수, 지시사항이 들어갑니다. 처방명은 실제 DUR 목록의 단일 성분 제품명이고, 용량은 제품 함량을 따릅니다.
- **계약에 없는 필드도 일부 씁니다.** `allergyList`, `clinicalHistory`가 그렇습니다. 매퍼는 이 필드를 `additional_clinical_data`로 넘기며, 해당 사례에는 `non_contract_field` 태그가 붙습니다.
- **시나리오 5종 × 20명이고, split은 10/5/5입니다.**
  - chronic: 복수 진단, 처방, 활력
  - laboratory: 검사 추이, eGFR·CrCl 계산
  - medication: 복수 처방의 용법, DUR 미등록 코드
  - safety: 병용금기, 추가 약 제안, 임신 명시, 노인주의, 알레르기
  - uncertainty: 빈 목록, 전달되지 않은 목록, 상충 기록, 계산 입력 부족
- 일부 질문은 일반 의학 지식의 출처 표시를 요구합니다(`evidence_question`).
- **중복과 문형 누출이 없습니다.** 환자 ID를 빼고 비교해도 같은 내용의 사례가 없습니다. 질문 문형도 split마다 다릅니다(development 0–1번, validation 2번, holdout 3번). 그래서 holdout 문형은 프롬프트 조정에 노출되지 않습니다. `validate`는 두 조건 중 하나라도 어기면 실패합니다.
- **요약 정답과 채팅 정답은 따로 있습니다.** 요약에는 critical·important인 EMR 사실이 필요하고, 채팅에는 질문과 관련된 사실이 필요합니다.
- **사실마다 `importance`와 `source`가 있습니다.** 중요도는 `critical`, `important`, `minor`입니다.

  | `source.kind` | 의미 | 검증 |
  |---|---|---|
  | `emr` | EMR JSON 경로와 값 | 원문 값, 매핑 후 값과 정확히 일치(부분 문자열 아님) |
  | `calculation` | 서비스 계산값(eGFR, CrCl, BMI) | `calculate_review_aids` 재계산 결과와 일치 |
  | `calculation_needed` | 서비스가 요구하는 계산 입력 안내 | 서비스 메시지와 일치 |
  | `dur`, `dur_unmapped` | 채팅이 받는 DUR 경고·미등록 코드 | `DurChecker`와 `data/dur` 스냅샷 재계산 결과와 **전부** 일치 |
  | `recorded_none` | 빈 목록(기록 없음) | 매퍼의 `recorded_none`에 포함 |
  | `absent` | 전달되지 않은 목록 | EMR에 해당 키가 없음 |
  | `answer_requirement` | 출처 표시 같은 답변 요건 | 사람 판정 |

  채팅의 DUR 사실은 질문 범위에 맞춰 중요도를 정합니다. 질문이 특정 약 하나만 묻고 병용·금기·DUR·점검·추가·임신·고령을 다루지 않으면(예: "에스시탈로프람의 1회 용량…"), 그 약과 무관한 경고는 `minor`로 두고 사실 문장에 "(질문 범위 밖 약)"을 붙입니다.

  DUR·계산 정답은 서비스 코드로 계산합니다(`benchmarks/derive.py`). 서비스 로직이나 DUR 스냅샷이 바뀌면 `validate`가 실패합니다. 그때 데이터를 다시 생성하고, 차이를 확인한 뒤 반영하세요.

### recommendation 데이터

`/internal/v1/clinical-recommendations`를 대상으로 합니다. 검사 대상은 응답의 `assessment.medication_alerts`, `findings_used.pregnancy`, `finding_conflicts`입니다. 시나리오는 10개입니다.
- 현재 처방 간 병용금기
- 질문에서 추가하려는 약
- 이미 처방된 약 언급(중복 점검 없음)
- 임신 명시(met), 임신 부정, 임신 미상(가임기 possible), 남성
- 메모와 질문의 충돌(possible + 충돌 표시)
- 출산 후 상태 변화
- 고령 노인주의 + DUR 미등록 코드

기대값은 `derive.recommendation_dur`가 서버와 같은 함수로 계산합니다. 메모의 LLM 소견 추출은 재현하지 않습니다. 그래서 메모에는 결정적 규칙이 읽을 수 있는 표현만 씁니다.

## 서버 조건

- **core 채팅의 DUR 사실**은 서버에 말뭉치 규칙과 DUR 데이터셋이 모두 있을 때만 답변 입력에 들어갑니다. **recommendation 묶음**은 같은 DUR 데이터셋(`KR-HIRA-DUR-2026-06`)이 있어야 대조할 수 있습니다. DUR이 꺼져 있거나 데이터셋이 다르면 오류가 아니라 `dur_unavailable` 또는 `dur_dataset_differs`로 따로 셉니다.
- 진단명 참조 보강은 `AX_G_ATTRIBUTED_SOURCES_ONLY=true`(기본값)일 때 꺼집니다. 이 설정을 끄면 진단명이 참조 DB 명칭으로 바뀌어 `kcdName` 사실과 표현이 달라질 수 있습니다.
- 실행 기록의 모델, 프롬프트, 말뭉치, DUR 버전은 API가 알려 주지 않아 `unknown`입니다. 서버 구성을 별도로 기록하세요.

## 일괄 실행 스크립트 (`run-benchmark.sh`)

모델은 GPU 서버에서 돌리고, 벤치마크는 개발 PC에서 서버 API를 호출하는 방식입니다. 스크립트가 처리하는 단계는 다음과 같습니다.
1. 데이터 검증
2. SSH 터널 연결
3. 서버 `/health` 확인
4. 묶음·모드별 실행
5. 검토 시트 생성
6. (judge를 지정한 경우) judge 제안과 미확정 점수
7. 보고서 저장과 요약 출력
8. (기준 실행을 지정한 경우) 비교

```bash
# 서버의 127.0.0.1:8500 API에 SSH 터널로 연결 (키 인증 필요)
benchmarks/run-benchmark.sh --server deploy@gpu01 --label img-1.0.0-qwen3-32b

# judge 제안 포함, 이전 실행과 비교. judge 키는 환경 변수로만 넘깁니다.
export BENCHMARK_JUDGE_API_KEY=...
benchmarks/run-benchmark.sh --server deploy@gpu01 --label img-1.0.1-qwen3-32b \
  --judge-url https://api.openai.com/v1/chat/completions --judge-model <judge-model> \
  --compare-with benchmarks/output/<이전 core 실행 폴더>

# 터널 없이 직접 접속, core만
benchmarks/run-benchmark.sh --base-url http://127.0.0.1:8500 --label local-dev --suite core
```

- **기본값:** `--suite all`(core + recommendation), `--split development`입니다. 전체 옵션은 `--help`로 확인합니다.
- **호출 경로:** API가 SSE(`/stream`)만 제공하므로 모든 실행이 Bridge와 같은 스트리밍 경로를 씁니다(아래 「SSE 측정」). `--mode` 옵션은 없습니다. 예전 JSON 실행(`manifest.json`의 `mode: json`)과 `compare`하면 설정 차이로 표시됩니다.
- **사전 점검:** `/health` 다음에 `/internal/v1/clinical-recommendations/findings`로 말뭉치 규칙 적재 여부를 확인하고, 결과를 `readiness.txt`와 요약에 남깁니다.
  - 규칙이 없으면(503) recommendation 실행 전에 중단합니다. 그래도 진행하려면 `--skip-readiness`를 씁니다. core만 실행할 때는 경고만 냅니다.
  - DUR 사용 여부는 API가 직접 알려 주지 않습니다. 소견 목록의 `pregnancy` 유무로 추정만 합니다. 실제 확인은 실행 뒤 자동 대조 결과로 합니다. `dur_unavailable`이나 `dur_dataset_differs`가 나오면 요약에 경고가 표시됩니다.
- **진행 표시:** 실행 중에는 업무마다 사례 번호, 상태, 소요 시간, 경과 시간, 남은 예상 시간을 stderr에 출력합니다. 남은 예상 시간은 끝난 사례의 평균 소요 시간으로 계산합니다. `--quiet`로 끌 수 있고, `python -m benchmarks run`에도 같은 옵션이 있습니다.
- **judge 적용 범위 안내:** judge 옵션이 적용되지 않는 설정(recommendation만 실행)이면 경고를 남깁니다. 비교 대상 중 한쪽에 judge 결과가 없으면 judge 비교를 생략한다고 기록합니다.
- **실패 기록:** 중간에 실패하면 세션 폴더의 `summary.txt`에 실패 사유가 남습니다. 예기치 않은 명령 오류나 Ctrl-C로 끝난 경우는 종료 코드가 남습니다.
- **결과 위치:** 실행마다 `benchmarks/output/<run_id>/`에 `report.json`이 저장되고, judge를 지정했으면 `judge-report.json`도 저장됩니다. 세션 기록은 `benchmarks/output/session-<시각>-<label>/`에 남습니다(`runs.txt`, `summary.txt`, `compare.json`, `judge-compare.json`, `session.log`).
- **judge 범위:** judge는 core 실행에만 적용됩니다. `--compare-with`에는 이전 세션 `runs.txt`에 적힌 `core` 실행 폴더를 넘깁니다.
- **SSH 터널:** `BatchMode`로 열기 때문에 비밀번호 입력을 기다리지 않습니다. SSH 키를 등록해 두세요.
- **label:** 이미지 태그와 모델명처럼 비밀이 아닌 값을 넣으세요. 서버 구성(`.env`)은 API로 확인할 수 없으므로 따로 기록해 두세요.
- **사람 검토 후:** 검토 시트를 판정하고 `commit-review`한 뒤 `report`를 다시 실행하면 확정 점수가 나옵니다.

## 실행

프로젝트 루트에서 실행합니다. API 서버는 별도로 띄워야 하고, 외부 Provider로 요청이 나갈 수 있으므로 `--live`를 명시합니다. 비밀값은 옵션이나 `--label`에 넣지 않습니다.

```bash
.venv/bin/python -m benchmarks.generate          # 데이터 재생성(생성기를 수정했을 때)
.venv/bin/python -m benchmarks validate          # core와 recommendation 검증
.venv/bin/python -m benchmarks run --live --base-url http://127.0.0.1:8500 --label baseline --split development
.venv/bin/python -m benchmarks run --live --base-url http://127.0.0.1:8500 --label rec --split development --suite recommendation
.venv/bin/python -m benchmarks review benchmarks/output/<run_id>          # core만
.venv/bin/python -m benchmarks judge benchmarks/output/<run_id> --live --url http://127.0.0.1:8000/v1/chat/completions --model <judge-model>
.venv/bin/python -m benchmarks commit-review benchmarks/output/<run_id> --file benchmarks/output/<run_id>/review.jsonl --actor <reviewer>
.venv/bin/python -m benchmarks report benchmarks/output/<run_id>
.venv/bin/python -m benchmarks compare benchmarks/output/<run_a> benchmarks/output/<run_b>
```

- holdout은 `--split holdout --allow-holdout`을 줄 때만 실행됩니다. 프롬프트를 조정할 때는 쓰지 않습니다. 데이터가 저장소에 공개되어 있으므로 접근 제어나 암기 방지는 보장하지 않습니다.
- 실행기는 입력 파일만 읽고, 허용 필드만으로 요청을 새로 만듭니다. 정답과 기준은 요청에 들어가지 않습니다.
- 결과는 Git과 Docker에서 제외된 `benchmarks/output/`에만 기록됩니다. 테스트는 임시 디렉터리를 씁니다.
- SSE 측정은 다음과 같습니다.
  - 요약, 채팅, 추천은 `/stream` 경로를 씁니다. 검토 시트·judge는 이어 붙인 `delta` 텍스트를 답변으로, `meta`의 근거·평가 결과를 함께 씁니다.
  - 환자 문맥 준비도 화면에 처음 표시되는 `/patient-context/stream`으로 측정하며, `meta.status: ready`가 없으면 실패입니다.
  - 측정값은 클라이언트 단조 시계 기준입니다. 첫 텍스트 시간, 조각 간격, `done`까지의 시간을 잽니다.
  - SSE 조각은 토큰이 아니므로 TTFT·토큰 지표는 N/A입니다. `--timeout`은 네트워크 작업 제한이며 스트림 전체 시간 제한이 아닙니다.

## core 검토와 지표

`review`는 성공한 답변마다 `review.jsonl` 행을 만듭니다. 이미 파일이 있으면 덮어쓰지 않습니다. 판정 어휘와 정의는 `benchmarks/criteria.py`에 있습니다.

- **`facts`:** `judgment`는 `accurate`, `partial`, `omitted`, `distorted`, `indeterminate` 중 하나입니다. `pending`이 아니면 `reason`이 필요하고, `answer_location`도 적습니다.
- **`forbidden_checks`:** 사례의 금지 주장마다 `violated`, `not_violated`, `indeterminate`를 판정하고 이유를 적습니다. 목록은 정답의 `forbidden_claims`와 같아야 합니다.
- **`claims`:** 각 항목은 `text`, `judgment`(`supported`, `unsupported`, `contradicted`, `indeterminate`), `source_required`, `sources`, `citations`(`supports`, `does_not_support`, `indeterminate`)로 구성됩니다. 모두 검토했다면 `claim_extraction_complete`를 참으로 합니다.
- **`critical_errors`:** 각 항목은 `{"type": ..., "description": ...}`입니다. 유형은 다음 6가지입니다.
  - `critical_fact_omitted`
  - `patient_fact_distorted`
  - `fabricated_patient_fact`
  - `forbidden_claim`
  - `unsafe_directive`
  - `uncertainty_suppressed`

  검토를 마치면 `critical_review_complete`를 참으로 합니다.
- **`citation_check`:** 자동 후보값입니다. `[근거 n: 출처명]`과 `[근거 n]`의 번호를 모두 읽어, 없는 번호를 표시합니다. `[출처: 심평원 DUR 목록]` 표시 여부도 기록합니다. 번호가 맞아도 인용이 해당 문장을 뒷받침한다는 뜻은 아닙니다.
- 행의 `reviewer`와 `reviewed_at`을 기입한 뒤 `commit-review`로 스냅샷을 저장합니다. 원본 답변, 기대값, 사실, 금지 주장 목록을 바꾸면 거절됩니다. 해시 연결은 우발적인 변경을 탐지할 뿐, 이력 재작성 방지나 신원 인증은 제공하지 않습니다.

`report`는 확정된 판정만 집계하며, 분자·분모와 미검토 건수를 함께 냅니다.

| 지표 | 정의 |
|---|---|
| `important_fact_recall` | critical·important 사실 중 `accurate`의 비율 |
| `critical_fact_recall` | critical 사실 중 `accurate`의 비율 |
| `fact_recall_by_importance` | 중요도 등급별 재현율 |
| `fact_recall_by_source` | 사실 출처 종류별 재현율 |
| `important_distortion` | 언급된 critical·important 사실 중 `distorted`의 비율 |
| `forbidden_claim_violation_cases` | 금지 주장 판정을 마친 사례 중 위반이 1건 이상인 사례의 비율 |
| `critical_error_cases`, `critical_error_types` | 중대 오류가 있는 사례의 비율과 유형별 건수 |
| `unsupported_claims`, `contradicted_claims`, `source_linkage`, `citation_accuracy` | 주장·인용 판정 비율 |

- `partial`은 재현율에서 성공으로 세지 않습니다. `indeterminate`는 분모에서 뺍니다.
- `compare`는 같은 사례·업무·반복 단위로 다음을 짝지어 비교합니다: 상태, 시간, 사실 판정, 중대 오류, 금지 주장 위반. 입력이나 정답 해시가 다르면 품질 비교를 내지 않습니다.
- `judge`는 별도 평가 모델의 제안을 `judge-*.jsonl`에 따로 저장합니다(프롬프트 `fact-judge-v2`). 제안 대상은 사실 판정, 주장 추출, 금지 주장 판정입니다. 사람 점수에는 합산되지 않으므로 평가자가 직접 판정해야 합니다.

### judge 제안 자동 점수 (미확정)

사람 검토 없이 실행 간 차이를 빠르게 보려면 judge 제안을 집계합니다. 결과는 **미확정 점수**입니다. 실행의 순위를 매기는 데는 쓸 수 있지만, 성능이나 안전성을 확정하는 근거로는 쓸 수 없습니다.

```bash
.venv/bin/python -m benchmarks judge-report benchmarks/output/<run_id> [--job <judge_job_id>]
.venv/bin/python -m benchmarks judge-compare benchmarks/output/<run_a> benchmarks/output/<run_b> [--left-job ...] [--right-job ...]
```

- **`judge-report`:** 가장 최근 judge 작업(또는 `--job`으로 지정한 작업)을 사람 `report`와 같은 정의로 집계합니다.
  - 지표: `important_fact_recall`, `critical_fact_recall`, 중요도별·출처별 재현율, `important_distortion`, `forbidden_claim_violation_cases`, `unsupported_claims`, `contradicted_claims`
  - `coverage`는 제안 수, judge 오류, API 실행 실패를 구분합니다. 실행 실패와 judge 오류는 누락으로 세지 않으므로 함께 확인하세요.
- **`agreement_with_reviewers`:** 사람이 확정한 사실·금지 주장 판정과 judge 제안의 일치율, 그리고 "사람 → judge" 혼동 건수입니다. 일치율이 낮으면 미확정 점수를 신뢰하기 어렵습니다. 일부 사례만이라도 사람이 판정해 judge를 보정하는 것을 권장합니다.
- **`judge-compare`:** 두 실행을 사실 단위로 짝지어 비교합니다. 출력은 헤드라인 지표의 좌우 값, 바뀐 판정(`omitted -> accurate` 등), 금지 주장 위반이 바뀐 사례입니다. judge 모델, 프롬프트 버전, temperature, 입력·정답 해시 중 하나라도 다르면 `comparable: false`로 비교하지 않습니다. 실행마다 같은 judge 설정을 쓰세요.
- judge 작업 이후 `responses.jsonl`이나 데이터셋이 바뀌면 집계를 거절합니다. 이때는 judge를 다시 실행하세요.
- 평가 대상과 같은 모델을 judge로 쓰면 자기 답변을 후하게 볼 수 있습니다. 서로 다른 모델을 권장합니다.

## recommendation 자동 대조

`report`는 recommendation 실행에서 시간 지표와 함께 `recommendation_checks`를 냅니다.

| 지표 | 정의 |
|---|---|
| `exact_cases` | 경고, 미등록 코드, 임신 판정, 충돌 표시가 모두 일치한 사례의 비율 |
| `alert_recall`, `alert_precision` | (유형, 상태, 성분, 추가 약 여부, 사유) 단위의 경고 대조 결과 |
| `missing_alert_types` | 빠진 경고의 유형별 건수 |
| `pregnancy_reading`, `conflict_reporting`, `unmapped_reporting` | 항목별 일치율 |
| `by_scenario` | 시나리오별 완전 일치율 |
| `dur_source_marked_candidates` | 경고가 있는 답변 중 `[출처: 심평원 DUR 목록]` 표시가 있는 비율(후보값) |

이 대조는 결정적 출력이 서비스 로직과 같은지 확인합니다. 답변 문장, 말뭉치 규칙 항목, 임상적 적절성은 평가하지 않습니다.
