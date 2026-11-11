-- AX-G 임상 말뭉치 서빙 스키마 (PostgreSQL 16 + pgvector 0.7 이상)
-- 설계 문서: docs/clinical_corpus_postgres.md
-- 적용: ax_g_ai.corpus_store.loader.ensure_schema() 가 corpus.schema_version 이 없을 때 한 번 실행한다(한 트랜잭션).
--
-- 원칙
--   * 작성 원천은 text_data/claims·mappings(YAML)다. 이 DB는 빌드 결과(corpus/*.jsonl)를 싣는 서빙 저장소다.
--   * 적재는 릴리스 단위(corpus_release)로 하고, 활성 릴리스 전환은 한 트랜잭션으로 한다(롤백 = 이전 릴리스 재활성화).
--   * 인용 가능 여부(A등급 출처·원문 확인·상태·검수)는 citable_evidence 뷰 한곳에서만 판정한다.
--   * 환자 식별 정보는 저장하지 않는다. 노출 로그에는 요청 ID와 근거 ID만 남긴다.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE SCHEMA IF NOT EXISTS corpus;
SET search_path = corpus, public;

CREATE TABLE schema_version (
    version     int PRIMARY KEY,
    applied_at  timestamptz NOT NULL DEFAULT now()
);
INSERT INTO schema_version (version) VALUES (1);

-- ── 릴리스 ───────────────────────────────────────────────────────────────
CREATE TABLE corpus_release (
    release_id      bigserial PRIMARY KEY,
    built_at        timestamptz NOT NULL,
    source_commit   text,                       -- 빌드한 저장소 커밋
    manifest        jsonb NOT NULL DEFAULT '{}'::jsonb,  -- 파일별 건수·SHA-256
    manifest_sha256 text NOT NULL,              -- 같은 말뭉치 재적재 방지
    loaded_at       timestamptz NOT NULL DEFAULT now(),
    is_active       boolean NOT NULL DEFAULT false
);
-- 활성 릴리스는 하나만
CREATE UNIQUE INDEX corpus_release_one_active ON corpus_release (is_active) WHERE is_active;

-- ── 출처 ─────────────────────────────────────────────────────────────────
CREATE TABLE source (
    release_id  bigint NOT NULL REFERENCES corpus_release ON DELETE CASCADE,
    source_id   text NOT NULL,
    title       text NOT NULL,
    url         text,
    ai_use      char(1) NOT NULL CHECK (ai_use IN ('A', 'B', 'C', 'U')),
    access      text,
    PRIMARY KEY (release_id, source_id)
);

-- ── 근거 문단(chunks.jsonl) ──────────────────────────────────────────────
CREATE TABLE evidence_chunk (
    release_id      bigint NOT NULL REFERENCES corpus_release ON DELETE CASCADE,
    chunk_id        text NOT NULL,
    claim_id        text NOT NULL,
    chapter         text NOT NULL,
    title_path      text[] NOT NULL,
    body            text NOT NULL,
    claim_type      text,
    claim_status    text NOT NULL,              -- 유지·개정·보완·폐기·확인불가
    ai_tier         text NOT NULL,              -- A | permission_required
    verification    text NOT NULL,              -- primary | local_db | secondary | unverified
    review_status   text NOT NULL CHECK (review_status IN ('draft', 'approved', 'rejected')),
    sources         jsonb NOT NULL,             -- [{id, title, locator, url, ai_use}]
    conflicts       jsonb NOT NULL DEFAULT '[]'::jsonb,
    updated_at      date,
    doc_path        text,
    body_sha256     text NOT NULL,              -- 임베딩 재사용 키
    -- 한국어는 PostgreSQL 기본 파서가 없으므로 앱이 만든 토큰(한글 2-gram + 영문 단어)을 'simple' 로 색인한다.
    -- 런타임 InMemoryEvidenceIndex.terms() 와 같은 토큰화다.
    search_tokens   text NOT NULL,
    search_tsv      tsvector GENERATED ALWAYS AS (to_tsvector('simple', search_tokens)) STORED,
    PRIMARY KEY (release_id, chunk_id)
);
CREATE INDEX evidence_chunk_tsv ON evidence_chunk USING gin (search_tsv);
CREATE INDEX evidence_chunk_claim ON evidence_chunk (release_id, claim_id);

-- ── 임베딩(모델별로 따로 — 모델 교체 시 재계산만 하면 된다) ───────────────
CREATE TABLE embedding_model (
    model_id    text PRIMARY KEY,               -- 예: 'bge-m3'
    dimensions  int NOT NULL,
    distance    text NOT NULL DEFAULT 'cosine',
    note        text
);

-- 차원은 모델마다 다르므로 모델별 테이블을 둔다(아래는 1024차원 예: bge-m3).
-- 다른 모델을 쓰면 같은 형태로 evidence_embedding_<모델> 을 추가한다.
CREATE TABLE evidence_embedding_1024 (
    model_id     text NOT NULL REFERENCES embedding_model,
    body_sha256  text NOT NULL,                 -- 같은 본문이면 릴리스가 바뀌어도 재사용
    embedding    vector(1024) NOT NULL,
    created_at   timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (model_id, body_sha256)
);
CREATE INDEX evidence_embedding_1024_hnsw ON evidence_embedding_1024 USING hnsw (embedding vector_cosine_ops);

-- ── 구조화 규칙(workup·treatment·dose_rules 등 — 2단계 이후 사용) ─────────
CREATE TABLE corpus_record (
    release_id      bigint NOT NULL REFERENCES corpus_release ON DELETE CASCADE,
    record_id       text NOT NULL,
    file            text NOT NULL,              -- workup.jsonl …
    kind            text NOT NULL,
    claim_id        text,
    ai_tier         text NOT NULL,
    claim_status    text,
    verification    text,
    review_status   text NOT NULL,
    usable          boolean NOT NULL,
    payload         jsonb NOT NULL,             -- 생성 레코드 원문(조건식 포함)
    PRIMARY KEY (release_id, record_id)
);
CREATE INDEX corpus_record_kind ON corpus_record (release_id, kind);

-- ── 검사·약제 개념(test_concepts·drug_concepts·drug_classes) ─────────────
CREATE TABLE concept (
    release_id  bigint NOT NULL REFERENCES corpus_release ON DELETE CASCADE,
    concept_id  text NOT NULL,
    concept_type text NOT NULL CHECK (concept_type IN ('test', 'ingredient', 'product', 'drug_class')),
    name_ko     text,
    payload     jsonb NOT NULL,                 -- LOINC·단위·환산·RxCUI·약효군 등 생성 레코드
    PRIMARY KEY (release_id, concept_type, concept_id)
);
CREATE TABLE concept_code (
    release_id  bigint NOT NULL,
    code_system text NOT NULL,                  -- LOINC | RXNORM
    code        text NOT NULL,
    concept_type text NOT NULL,
    concept_id  text NOT NULL,
    PRIMARY KEY (release_id, code_system, code, concept_type, concept_id),
    FOREIGN KEY (release_id, concept_type, concept_id) REFERENCES concept ON DELETE CASCADE
);
CREATE TABLE concept_usage (
    release_id  bigint NOT NULL,
    concept_type text NOT NULL,
    concept_id  text NOT NULL,
    record_id   text NOT NULL,
    claim_id    text,
    roles       text[] NOT NULL,
    via_class   text,                           -- 약효군 경유 사용
    FOREIGN KEY (release_id, concept_type, concept_id) REFERENCES concept ON DELETE CASCADE
);
CREATE INDEX concept_usage_lookup ON concept_usage (release_id, concept_type, concept_id);
CREATE INDEX concept_usage_claim ON concept_usage (release_id, claim_id);

-- ── 병원 사이트 매핑(릴리스와 독립 — 병원이 관리) ─────────────────────────
CREATE TABLE site_code_map (
    site        text NOT NULL,
    code_kind   text NOT NULL CHECK (code_kind IN ('lab', 'med')),
    code        text NOT NULL,
    mapping     jsonb NOT NULL,                 -- lab: {concept, unit} / med: {ingredients|product, route, classes}
    status      text NOT NULL DEFAULT 'draft' CHECK (status IN ('draft', 'reviewed', 'retired')),
    reviewed_by text,
    reviewed_at timestamptz,
    PRIMARY KEY (site, code_kind, code)
);

-- ── 검수 피드백과 노출 기록(환자 정보 없음) ──────────────────────────────
CREATE TABLE review_feedback (
    feedback_id bigserial PRIMARY KEY,
    chunk_id    text NOT NULL,
    claim_id    text NOT NULL,
    release_id  bigint NOT NULL,
    reviewer    text NOT NULL,                  -- 의료진 계정 ID(환자 아님)
    verdict     text NOT NULL CHECK (verdict IN ('agree', 'disagree', 'source_mismatch')),
    comment     text,
    created_at  timestamptz NOT NULL DEFAULT now()
);
CREATE TABLE evidence_exposure (
    request_id  text NOT NULL,
    release_id  bigint NOT NULL,
    chunk_ids   text[] NOT NULL,
    unreviewed  int NOT NULL,
    shown_at    timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX evidence_exposure_time ON evidence_exposure (shown_at);

-- ── 인용 가능 근거: 게이트는 여기 한곳 ────────────────────────────────────
-- 런타임 is_citable() 과 같은 규칙. 미검수 포함 여부는 질의에서 review_status 로 거른다.
CREATE VIEW citable_evidence AS
SELECT c.*
FROM evidence_chunk c
JOIN corpus_release r ON r.release_id = c.release_id AND r.is_active
WHERE c.ai_tier = 'A'
  AND c.verification IN ('primary', 'local_db')
  AND c.claim_status NOT IN ('폐기', '확인불가')
  AND c.review_status IN ('approved', 'draft')
  AND jsonb_array_length(c.sources) > 0
  AND NOT EXISTS (
      SELECT 1 FROM jsonb_array_elements(c.sources) s WHERE s->>'ai_use' IS DISTINCT FROM 'A'
  );

-- ── 하이브리드 검색 예시(어휘 + 벡터, 순위 융합 RRF) ─────────────────────
-- :query_tokens = 앱 토큰화 결과('담낭 낭염 ...'), :query_vec = 질문 임베딩, :include_unreviewed, :limit
--
-- WITH lexical AS (
--     SELECT chunk_id, row_number() OVER (ORDER BY ts_rank_cd(search_tsv, q) DESC) AS rnk
--     FROM citable_evidence, to_tsquery('simple', replace(:query_tokens, ' ', ' | ')) q
--     WHERE search_tsv @@ q AND (:include_unreviewed OR review_status = 'approved')
--     ORDER BY rnk LIMIT 30
-- ), semantic AS (
--     SELECT c.chunk_id, row_number() OVER (ORDER BY e.embedding <=> :query_vec) AS rnk
--     FROM citable_evidence c
--     JOIN evidence_embedding_1024 e ON e.body_sha256 = c.body_sha256 AND e.model_id = 'bge-m3'
--     WHERE (:include_unreviewed OR c.review_status = 'approved')
--     ORDER BY e.embedding <=> :query_vec LIMIT 30
-- )
-- SELECT chunk_id, sum(1.0 / (60 + rnk)) AS score
-- FROM (SELECT * FROM lexical UNION ALL SELECT * FROM semantic) ranked
-- GROUP BY chunk_id ORDER BY score DESC LIMIT :limit;
