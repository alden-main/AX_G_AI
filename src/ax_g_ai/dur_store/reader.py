"""Read the active DUR dataset from PostgreSQL into memory (once, at start-up).

About 70k small rows: lookups per prescription then need no database round trip, and a
new dataset is picked up on restart (the same as the corpus rules).
"""

from __future__ import annotations

import psycopg
from psycopg.rows import dict_row

from ax_g_ai.dur_store.snapshot import TABLE_COLUMNS
from ax_g_ai.services.dur_check import DurTables, DurUnavailableError


def read_active_tables(database_url: str) -> DurTables:
    with psycopg.connect(database_url, autocommit=True, connect_timeout=3, row_factory=dict_row) as conn:
        if conn.execute("SELECT to_regclass('dur.dataset') AS t").fetchone()["t"] is None:
            raise DurUnavailableError("dur schema not loaded (run ax-g-ai-load-dur)")
        dataset = conn.execute(
            "SELECT dataset_id, source_id, title, url, license, region, built_at::text AS built_at "
            "FROM dur.dataset WHERE is_active"
        ).fetchone()
        if dataset is None:
            raise DurUnavailableError("no active dur dataset")
        rows = {
            table: list(conn.execute(
                f"SELECT {', '.join(columns)} FROM dur.{table} WHERE dataset_id = %s ORDER BY {', '.join(columns[:2])}",
                (dataset["dataset_id"],),
            ))
            for table, columns in TABLE_COLUMNS.items()
        }
    source = {"id": dataset["source_id"], "title": dataset["title"], "url": dataset["url"],
              "license": dataset["license"], "region": dataset["region"]}
    return DurTables.from_rows(source, rows, dataset_id=dataset["dataset_id"])
