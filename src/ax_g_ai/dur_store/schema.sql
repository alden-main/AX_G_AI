-- AX-G DUR(심평원 의약품안전사용서비스 의약품 목록) 서빙 스키마 (PostgreSQL 16)
-- 말뭉치(corpus 스키마)와 같은 DB, 별도 스키마 dur. 설계 문서: docs/clinical_corpus_postgres.md 「DUR 스키마」
-- 적용: ax_g_ai.dur_store.loader.ensure_schema() 가 dur.schema_version 이 없을 때 한 번 실행한다(한 트랜잭션).
--
-- 원칙
--   * 작성 원천은 공공데이터포털 CSV(dur/)다. 빌드 결과(data/dur 스냅샷)를 데이터셋 단위로 싣는다.
--   * 활성 데이터셋은 하나. 전환은 한 트랜잭션(롤백 = 이전 데이터셋 재활성화).
--   * 규칙은 성분코드 단위로 중복 제거되어 있고, 제품코드(9자리 보험코드) → 성분코드는 product 표로 찾는다.
--   * 환자 정보는 저장하지 않는다.

CREATE SCHEMA IF NOT EXISTS dur;
SET search_path = dur, public;

CREATE TABLE schema_version (
    version     int PRIMARY KEY,
    applied_at  timestamptz NOT NULL DEFAULT now()
);
INSERT INTO schema_version (version) VALUES (1);

-- ── 데이터셋(공고 시점별 목록) ───────────────────────────────────────────
CREATE TABLE dataset (
    dataset_id      bigserial PRIMARY KEY,
    source_id       text NOT NULL,              -- 예: KR-HIRA-DUR-2026-06
    title           text NOT NULL,
    url             text,
    license         text NOT NULL,              -- 출처 표시 문구(공공누리 제1유형)
    region          text NOT NULL,
    built_at        timestamptz NOT NULL,       -- 스냅샷 생성 시각
    inputs          jsonb NOT NULL,             -- 원본 CSV 파일명·SHA-256
    counts          jsonb NOT NULL,             -- 표별 행 수
    content_sha256  text NOT NULL,              -- 같은 내용 재적재 방지
    loaded_at       timestamptz NOT NULL DEFAULT now(),
    is_active       boolean NOT NULL DEFAULT false
);
CREATE UNIQUE INDEX dataset_one_active ON dataset (is_active) WHERE is_active;

-- ── 제품 → 성분(EMR medCode = 9자리 보험 제품코드) ───────────────────────
CREATE TABLE product (
    dataset_id      bigint NOT NULL REFERENCES dataset ON DELETE CASCADE,
    product_code    text NOT NULL,
    product_name    text,
    ingredient_code text NOT NULL,
    ingredient_name text,
    company         text,
    insured         text,
    PRIMARY KEY (dataset_id, product_code)
);
CREATE INDEX product_ingredient ON product (dataset_id, ingredient_code);

-- ── 병용금기(성분 쌍, ingredient_a < ingredient_b 로 정렬해 한 번만) ──────
CREATE TABLE pair_rule (
    dataset_id      bigint NOT NULL REFERENCES dataset ON DELETE CASCADE,
    ingredient_a    text NOT NULL,
    ingredient_b    text NOT NULL,
    name_a          text,
    name_b          text,
    detail          text,
    note            text,                       -- 비고(예: 1주 MTX 15mg 이상) — 평가하지 않고 그대로 보인다
    notice_no       text,
    notice_date     text
);
CREATE INDEX pair_rule_a ON pair_rule (dataset_id, ingredient_a);
CREATE INDEX pair_rule_b ON pair_rule (dataset_id, ingredient_b);

-- ── 연령금기 ─────────────────────────────────────────────────────────────
CREATE TABLE age_rule (
    dataset_id      bigint NOT NULL REFERENCES dataset ON DELETE CASCADE,
    ingredient_code text NOT NULL,
    ingredient_name text,
    age             double precision NOT NULL,
    unit            text NOT NULL,              -- 세 | 개월 | 주
    condition       text NOT NULL,              -- 미만 | 이하 | 이상 | 초과
    detail          text,
    notice_no       text,
    notice_date     text
);
CREATE INDEX age_rule_ingredient ON age_rule (dataset_id, ingredient_code);

-- ── 임부금기 ─────────────────────────────────────────────────────────────
CREATE TABLE pregnancy_rule (
    dataset_id      bigint NOT NULL REFERENCES dataset ON DELETE CASCADE,
    ingredient_code text NOT NULL,
    ingredient_name text,
    grade           text,                       -- 1 | 2
    detail          text,
    notice_no       text,
    notice_date     text
);
CREATE INDEX pregnancy_rule_ingredient ON pregnancy_rule (dataset_id, ingredient_code);

-- ── 노인주의(일반·해열진통소염제) ─────────────────────────────────────────
CREATE TABLE elderly_rule (
    dataset_id      bigint NOT NULL REFERENCES dataset ON DELETE CASCADE,
    ingredient_code text NOT NULL,
    ingredient_name text,
    category        text NOT NULL,              -- 노인주의 | 노인주의(해열진통소염제)
    detail          text,
    notice_no       text,
    notice_date     text
);
CREATE INDEX elderly_rule_ingredient ON elderly_rule (dataset_id, ingredient_code);

-- 용량주의·투여기간주의·1일 최대투여량 목록은 원본 파일을 받으면 같은 형태(dataset_id + 성분코드)의 표로 추가한다.
