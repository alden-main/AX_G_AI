"""릴리스 하나의 규칙 레코드·개념·인용 가능 문단을 PostgreSQL에서 읽는 모듈.

설계상 위치: 말뭉치 저장 계층의 읽기 경로. 스냅샷 빌더와 ``retriever.load_index``가 쓰고, 도구·테스트는
``read_rows_from_database``를 쓴다.

역할: ``text_data/corpus/*.jsonl``과 같은 모양의 행을 돌려준다. 그래서 ``RuleCorpus.from_rows``와
``InMemoryEvidenceIndex.from_rows``가 한 번 읽은 결과로 규칙과 근거 색인을 만들고, DB 행과 파일 행에
같은 Python 게이트(``is_citable``)를 적용한다.
"""

from __future__ import annotations

from typing import Any

import psycopg
from psycopg.rows import dict_row

from ax_g_ai.corpus.files import CHUNKS_FILE, CONCEPT_FILES, RULE_FILES

_CONCEPT_FILE = {"test": "test_concepts.jsonl", "ingredient": "drug_concepts.jsonl", "product": "drug_concepts.jsonl",
                 "drug_class": "drug_classes.jsonl"}


def read_rows(conn: psycopg.Connection, release_id: int) -> dict[str, list[dict[str, Any]]]:
    """``release_id``의 행을 말뭉치 파일 이름별로 돌려준다.

    역할: 규칙 레코드(``corpus_record``), 개념(``concept``에 ``concept_usage``를 붙여 ``used_by`` 복원),
    인용 가능 문단(게이트 뷰 ``corpus.citable_evidence``)을 읽어 jsonl 행 모양으로 맞춘다. 문단은 게이트 뷰에서
    오므로 ``release_id``는 활성 릴리스여야 한다.
    입력:
        conn: PostgreSQL 연결.
        release_id: 읽을 말뭉치 릴리스 ID(활성 릴리스).
    출력: {파일 이름: 행 목록}. 규칙·개념·문단 파일 이름이 모두 키로 있다(없으면 빈 목록).
    """
    rows: dict[str, list[dict[str, Any]]] = {name: [] for name in (*RULE_FILES, *CONCEPT_FILES, CHUNKS_FILE)}
    with conn.cursor(row_factory=dict_row) as cur:
        for row in cur.execute(
            "SELECT file, payload FROM corpus.corpus_record WHERE release_id = %s AND file = ANY(%s) ORDER BY record_id",
            (release_id, list(RULE_FILES)),
        ):
            rows[row["file"]].append(row["payload"])
        # 저장된 payload에는 ``used_by``가 빠져 있다(concept_usage에 따로 보관). 근거 색인은 개념마다 그 개념을 쓰는
        # 주장이 필요하므로 jsonl 행 모양으로 다시 조인해 붙인다.
        for row in cur.execute(
            "SELECT c.concept_type, c.payload, coalesce(json_agg(json_build_object('claim_id', u.claim_id)) "
            "  FILTER (WHERE u.claim_id IS NOT NULL), '[]') AS used_by "
            "FROM corpus.concept c LEFT JOIN corpus.concept_usage u USING (release_id, concept_type, concept_id) "
            "WHERE c.release_id = %s GROUP BY c.concept_type, c.concept_id, c.payload ORDER BY c.concept_id", (release_id,),
        ):
            name = _CONCEPT_FILE.get(row["concept_type"])
            if name is not None:
                rows[name].append({**row["payload"], "used_by": row["used_by"]})
        rows[CHUNKS_FILE] = list(cur.execute(
            "SELECT chunk_id, claim_id, chapter, title_path, body AS text, claim_status, ai_tier, verification, "
            "review_status, sources, conflicts, updated_at::text AS updated_at, doc_path "
            "FROM corpus.citable_evidence WHERE release_id = %s ORDER BY chunk_id", (release_id,),
        ))
    return rows


def read_rows_from_database(database_url: str) -> tuple[int, dict[str, list[dict[str, Any]]]]:
    """일회용 연결로 활성 릴리스의 (release_id, 행)을 읽는다(도구·테스트용)."""
    from ax_g_ai.corpus_store.retriever import active_release_id  # retriever가 이 모듈을 import하므로 순환을 피해 함수 안에서 import

    with psycopg.connect(database_url, autocommit=True, connect_timeout=3) as conn:
        release_id = active_release_id(conn)
        return release_id, read_rows(conn, release_id)
