"""PostgreSQL의 DUR 데이터셋을 메모리(``DurTables``)로 읽는 모듈.

설계상 위치: DUR 저장 계층의 런타임 읽기 경로. 스냅샷 빌드 때와 ``corpus_store.live.LiveCorpus``의 활성 ID 확인에서 쓰인다.

역할: 약 7만 개의 작은 행을 한 번에 읽어 두므로, 처방마다 조회할 때 DB 왕복이 필요 없다. 새로 활성화된 데이터셋은
``LiveCorpus``가 말뭉치 릴리스와 함께 감지해(같은 확인, 같은 교체) 반영한다.
"""

from __future__ import annotations

import psycopg
from psycopg.rows import dict_row

from ax_g_ai.dur_store.snapshot import TABLE_COLUMNS
from ax_g_ai.services.dur_check import DurTables, DurUnavailableError


def active_dataset_id(conn: psycopg.Connection) -> int:
    """활성 DUR 데이터셋 ID를 돌려준다.

    역할: ``dur.dataset`` 표가 있는지 먼저 확인한 뒤 ``is_active`` 행을 찾는다.
    입력:
        conn: PostgreSQL 연결.
    출력: 활성 데이터셋 ID.
    예외: ``dur`` 스키마가 적재되지 않았거나 활성 데이터셋이 없으면 ``DurUnavailableError``.
    """
    with conn.cursor(row_factory=dict_row) as cur:
        if cur.execute("SELECT to_regclass('dur.dataset') AS t").fetchone()["t"] is None:  # type: ignore[index]
            raise DurUnavailableError("dur schema not loaded (run ax-g-ai-load-dur)")
        row = cur.execute("SELECT dataset_id FROM dur.dataset WHERE is_active").fetchone()
    if row is None:
        raise DurUnavailableError("no active dur dataset")
    return row["dataset_id"]


def read_tables(conn: psycopg.Connection, dataset_id: int) -> DurTables:
    """데이터셋 하나의 모든 DUR 표를 읽어 ``DurTables``로 만든다.

    역할: 데이터셋 메타데이터로 출처 정보를 만들고, ``TABLE_COLUMNS``의 표마다 행을 앞 두 열 순서로 읽는다.
    입력:
        conn: PostgreSQL 연결.
        dataset_id: 읽을 데이터셋 ID.
    출력: 출처·행·``dataset_id``를 담은 ``DurTables``.
    예외: 데이터셋이 없으면 ``DurUnavailableError``.
    """
    with conn.cursor(row_factory=dict_row) as cur:
        dataset = cur.execute(
            "SELECT dataset_id, source_id, title, url, license, region FROM dur.dataset WHERE dataset_id = %s", (dataset_id,),
        ).fetchone()
        if dataset is None:
            raise DurUnavailableError(f"dur dataset {dataset_id} not found")
        rows = {
            table: list(cur.execute(
                f"SELECT {', '.join(columns)} FROM dur.{table} WHERE dataset_id = %s ORDER BY {', '.join(columns[:2])}",
                (dataset_id,),
            ))
            for table, columns in TABLE_COLUMNS.items()
        }
    source = {"id": dataset["source_id"], "title": dataset["title"], "url": dataset["url"],
              "license": dataset["license"], "region": dataset["region"]}
    return DurTables.from_rows(source, rows, dataset_id=dataset_id)


def read_active_tables(database_url: str) -> DurTables:
    """일회용 연결로 활성 데이터셋을 읽는다(도구·테스트용)."""
    with psycopg.connect(database_url, autocommit=True, connect_timeout=3) as conn:
        return read_tables(conn, active_dataset_id(conn))
