"""Read a DUR dataset from PostgreSQL into memory.

About 70k small rows: lookups per prescription then need no database round trip. A newly
activated dataset is picked up by ``corpus_store.live.LiveCorpus`` together with the corpus
release (same check, same swap).
"""

from __future__ import annotations

import psycopg
from psycopg.rows import dict_row

from ax_g_ai.dur_store.snapshot import TABLE_COLUMNS
from ax_g_ai.services.dur_check import DurTables, DurUnavailableError


def active_dataset_id(conn: psycopg.Connection) -> int:
    with conn.cursor(row_factory=dict_row) as cur:
        if cur.execute("SELECT to_regclass('dur.dataset') AS t").fetchone()["t"] is None:  # type: ignore[index]
            raise DurUnavailableError("dur schema not loaded (run ax-g-ai-load-dur)")
        row = cur.execute("SELECT dataset_id FROM dur.dataset WHERE is_active").fetchone()
    if row is None:
        raise DurUnavailableError("no active dur dataset")
    return row["dataset_id"]


def read_tables(conn: psycopg.Connection, dataset_id: int) -> DurTables:
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
    """The active dataset over a one-off connection (tools and tests)."""
    with psycopg.connect(database_url, autocommit=True, connect_timeout=3) as conn:
        return read_tables(conn, active_dataset_id(conn))
