# 검사·약제 매핑

말뭉치(`corpus/*.jsonl`)의 규칙이 쓰는 검사·변수·약제 식별자를 표준 개념으로 묶고, EMR 코드와 연결한다.

```
EMR 원문(Bridge)                       개념 사전(이 폴더)                     말뭉치 규칙
labResultList.examCode ─┐                                                    {var: K, op: ">", value: 5.0}
  + unit(또는 사이트 단위) ├─ sites/<병원>.json ─▶ tests.yaml 개념(K) ─▶ 기준 단위 환산·범위 확인 ─▶ values["K"]
medicineDtailList.medCode ┘   (검수된 코드 매핑)   drugs.yaml 성분·약효군 ─▶ medications / medication_classes ─▶ {med_class: RASi}
medName "(성분명)" ───────── 성분명 일치(보조) ──┘
```

## 파일

| 파일 | 내용 | 수정 |
|---|---|---|
| `tests.yaml` | 검사·변수 개념 249개. labs(검체 검사), panels, exams(영상·내시경), measurements, vitals, scores, derived, med_derived, clinical, tasks | 사람이 편집 |
| `drugs.yaml` | 성분 개념, 복합제, 약효군 | 사람이 편집 |
| `verified/loinc.json` | LOINC 코드별 공식 이름(NLM Clinical Tables 조회 결과) | `tools/verify_codes.py` |
| `verified/rxnorm.json` | 성분명별 RxCUI·FDA EPC(NLM RxNav 조회 결과) | `tools/verify_codes.py` |
| `sites/example.json` | 병원 코드 매핑 형식 예시(가상 코드) | 병원별로 새로 작성 |

생성물(`build_corpus.py`가 만든다):

| 파일 | 한 줄 = |
|---|---|
| `corpus/test_concepts.jsonl` | 검사·변수 개념 1개: LOINC, 기준 단위, 단위 환산, 허용 범위, EMR 별칭, 이 개념을 쓰는 말뭉치 항목(`used_by`) |
| `corpus/drug_concepts.jsonl` | 성분·복합제 1개: RxCUI, FDA EPC, 국내 성분명, 소속 약효군, `used_by`, 약효군 경유 사용(`used_by_class`) |
| `corpus/drug_classes.jsonl` | 약효군 1개: 펼친 성분 목록, 완전성(`exhaustive`), 판정 가능 여부(`derivable`), `used_by` |
| 기존 `corpus/*.jsonl` 각 항목 | `concepts: {tests, drugs, drug_classes}` 필드가 추가된다 |

## 빌드 규칙

- 말뭉치의 `lab`·`exam`·`var`·`check`·`renal_param`, `med`·`med_class`·`ingredient(s)`·`drug_class` 등은 반드시 개념으로 풀려야 한다. 풀리지 않으면 빌드가 실패한다.
- `options`·`first_line`처럼 약제와 비약물 항목(CBT 등)이 섞인 필드는 풀리는 것만 연결한다.
- `tests.yaml`의 LOINC와 `drugs.yaml`의 RxNorm 성분명은 `verified/` 캐시에 있어야 한다. 새 코드를 넣으면 먼저 실행한다.

```bash
uv run --with pyyaml python text_data/tools/verify_codes.py      # 새 코드만 조회(네트워크)
uv run --with pyyaml python text_data/tools/build_corpus.py      # 검증 + 생성(네트워크 없음)
```

## 런타임 매핑 원칙 (`src/ax_g_ai/services/emr_mapping.py`)

**검사**
1. 개념 결정: 사이트 매핑의 `examCode` → 없으면 `examCode`가 LOINC 코드인 경우 → 없으면 매핑하지 않는다.
2. 단위 결정: EMR `unit` → 없으면 사이트 매핑 `unit`. 단위를 모르면 값을 쓰지 않는다. 같은 검사라도 병원마다 단위가 다르다(예: CRP mg/dL·mg/L).
3. 기준 단위로 환산(`convert`, HbA1c IFCC→NGSP 처럼 오프셋 포함), 허용 범위(`range`) 밖이면 입력·단위 오류로 보고 버린다.
4. `<0.5`, `양성` 같은 비수치 결과는 규칙 값으로 쓰지 않는다.
5. 이름(`itemName`)만 맞는 경우는 `suggestion`으로만 돌려준다 — 사이트 매핑 작성용.
6. eGFR은 검사실 보고값을 우선하고, 없을 때만 크레아티닌·연령·성별이 모두 있으면 CKD-EPI 2021로 계산한다. CrCl은 체중까지 있을 때만 Cockcroft-Gault(실제 체중)로 계산한다.

**약제**
1. 성분 결정: 사이트 매핑의 `medCode` → 없으면 `medName` 괄호 안 성분명(예: `리피틴정10mg(아토르바스타틴칼슘수화물)`)을 국내 성분명과 앞부분 일치로 비교(염 접두어 `염산`·`황산` 등 제거)한다.
2. 처방 하나라도 성분을 확인하지 못하면 `medications_complete=False` → 규칙의 "이 약을 쓰지 않음"은 알 수 없음(None)이 된다.
3. 약효군 판정
   - `exhaustive: true`(예: statin, DOAC, SGLT2i) — 확인된 성분만으로 해당 없음(False)을 판정할 수 있다.
   - `exhaustive: false`(예: antibiotic, opioid, NSAID) — 목록 밖 성분이 처방돼 있으면 알 수 없음(None).
   - `derivable: false`(예: long_term_opioid, RASi_max_tolerated) — 용량·기간 판단이 필요하므로 사이트 매핑이 `classes`로 지정하지 않으면 항상 None.
   - `systemic_only`(전신 스테로이드·NSAID·전신 에스트로겐) — 크림·연고·점안·흡입·질정 제형은 넣지 않는다.
   - `derive_from + routes`(경구 NSAID, 경구 에스트로겐) — 투여 경로를 알 때만 판정한다.
4. 사이트 매핑에는 사전에 없는 성분(예: `amlodipine`)도 적을 수 있다. 그러면 그 처방은 "확인된 기타 성분"이 되어 완전 목록 약효군의 판정이 가능해진다.

**상병**: `diagnosisList[].diagnosisCode`의 KCD는 점 없는 형식(`K810`)도 `K81.0`으로 맞춘다.

## 사이트 매핑 만들기

```bash
uv run python text_data/tools/propose_site_map.py --labs 검사마스터.csv --meds 약품마스터.csv --site 병원명 -o site.json
```

결과는 `status: draft` 초안이다. 검사실 단위와 성분을 의료진·검사실이 확인한 뒤 사용한다. 런타임은 `SiteCodeMap.load(path, catalog)`로 읽으며, 사전에 없는 개념·복합제·약효군을 가리키면 로드를 거부한다.

## 표준 코드 출처와 이용 조건

| 코드 | 출처 | 이용 조건 | 이 저장소에서 |
|---|---|---|---|
| LOINC | Regenstrief Institute, NLM Clinical Tables API로 조회 | LOINC 라이선스: 상업적 이용 무료, 저작권 고지 필요 | 코드·공식 이름 저장. 배포물에 고지 문구 포함 필요 |
| RxNorm RxCUI | NLM RxNav | RxNorm 자체 데이터(SAB=RXNORM)는 자유 이용, 다른 원천 어휘는 UMLS 조건 | RxCUI·RxNorm 이름만 저장 |
| FDA EPC 약리 분류 | FDA SPL(RxClass 경유) | 미국 정부 저작물(공공 영역) | 참고 필드로 저장 |
| ATC | WHO Collaborating Centre | 상업적 복제·배포 제한 | 저장하지 않음 |
| 국내 성분명(`ko`) | 국내 표기 관례로 작성한 이름 매칭용 별칭 | — | 임상 근거가 아님. 사이트 매핑이 우선 |

LOINC 고지 문구(배포 시 포함):
> This material contains content from LOINC® (http://loinc.org). LOINC is copyright © Regenstrief Institute, Inc. and the Logical Observation Identifiers Names and Codes (LOINC) Committee and is available at no cost under the license at http://loinc.org/license. LOINC® is a registered United States trademark of Regenstrief Institute, Inc.

## 알려진 한계

- 국내 성분명 별칭은 이름 매칭 보조용이다. 표기가 다른 제품은 매칭되지 않으며(→ 알 수 없음으로 처리) 사이트 매핑으로 보완한다. 심평원 주성분코드 연동은 아직 없다.
- 검사 코드는 병원마다 다르므로 국내 표준 검사 코드(건강보험 수가 코드)와의 연결은 사이트 매핑에서 한다.
- `findings`(임상 소견)·`event`는 EMR 구조화 필드가 없어 매핑하지 않는다(규칙에서 알 수 없음으로 남는다).
- 24시간 소변 결석 패널 일부 항목, 대변 감염 패널, 항히스타민·5-HT3 길항제·VMAT2 억제제 성분 목록은 아직 비어 있다.
