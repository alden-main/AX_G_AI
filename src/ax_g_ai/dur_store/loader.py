"""Load the DUR snapshot (``data/dur``, see ``snapshot``) into PostgreSQL schema ``dur`` as a dataset.

Steps: verify snapshot → ensure schema → reuse the dataset with the same content hash, or
insert a new inactive one (COPY) → validate row counts → activate atomically → prune old
datasets. The runtime reads the active dataset once at start-up (``reader``).

CLI (compose one-shot service ``dur-loader``)::

    ax-g-ai-load-dur                       # env: AX_G_CORPUS_DATABASE_URL, AX_G_DUR_SNAPSHOT_DIR
    ax-g-ai-load-dur --activate 2          # roll back to dataset 2
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import psycopg
from psycopg.types.json import Jsonb

from ax_g_ai.config import ConfigurationError, LoaderSettings
from ax_g_ai.dur_store.snapshot import TABLE_COLUMNS, DurDataError, Snapshot, read_snapshot
from ax_g_ai.pg_release import ReleaseTable

logger = logging.getLogger("ax_g_ai.dur_store")

KEEP_DATASETS = 3


class DurLoadError(RuntimeError):
    """The loaded dataset failed validation or the requested dataset does not exist."""


@dataclass
class DurLoadResult:
    dataset_id: int
    reused: bool
    counts: dict[str, int]


DATASETS = ReleaseTable("dur", "dataset", "dataset_id", "ax_g_ai.dur_store", KEEP_DATASETS, DurLoadError, "dataset")
ensure_schema = DATASETS.ensure_schema
activate = DATASETS.activate  # also the rollback: activate an older dataset


def load_dur(conn: psycopg.Connection, snapshot_dir: Path) -> DurLoadResult:
    snapshot = read_snapshot(snapshot_dir)
    ensure_schema(conn)
    counts = {name: len(rows) for name, rows in snapshot.rows.items()}
    existing = conn.execute(
        "SELECT dataset_id FROM dur.dataset WHERE content_sha256 = %s ORDER BY dataset_id DESC LIMIT 1",
        (snapshot.content_sha256,),
    ).fetchone()
    if existing is not None:
        dataset_id, reused = existing[0], True
    else:
        with conn.transaction():
            dataset_id = _insert_dataset(conn, snapshot)
        reused = False
    _validate(conn, dataset_id, counts)
    activate(conn, dataset_id)
    DATASETS.prune(conn)
    return DurLoadResult(dataset_id, reused, counts)


def _insert_dataset(conn: psycopg.Connection, snapshot: Snapshot) -> int:
    source, manifest = snapshot.source, snapshot.manifest
    dataset_id = conn.execute(
        "INSERT INTO dur.dataset (source_id, title, url, license, region, built_at, inputs, counts, content_sha256) "
        "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING dataset_id",
        (source["id"], source["title"], source.get("url"), source["license"], source["region"], manifest["built_at"],
         Jsonb(manifest.get("inputs") or {}), Jsonb({name: len(rows) for name, rows in snapshot.rows.items()}),
         snapshot.content_sha256),
    ).fetchone()[0]
    with conn.cursor() as cur:
        for table, columns in TABLE_COLUMNS.items():
            with cur.copy(f"COPY dur.{table} (dataset_id, {', '.join(columns)}) FROM STDIN") as copy:
                for row in snapshot.rows[table]:
                    copy.write_row((dataset_id, *(row[column] for column in columns)))
    return dataset_id


def _validate(conn: psycopg.Connection, dataset_id: int, counts: dict[str, int]) -> None:
    for table, expected in counts.items():
        actual = conn.execute(f"SELECT count(*) FROM dur.{table} WHERE dataset_id = %s", (dataset_id,)).fetchone()[0]
        if actual != expected:
            raise DurLoadError(f"dataset {dataset_id}: {table} has {actual} rows, expected {expected}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Load the HIRA DUR snapshot into PostgreSQL (schema dur).")
    parser.add_argument("--snapshot-dir", type=Path, help="snapshot folder (default: AX_G_DUR_SNAPSHOT_DIR or data/dur)")
    parser.add_argument("--activate", type=int, metavar="DATASET_ID", help="activate an existing dataset (rollback) and exit")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        settings = LoaderSettings.from_environment()
    except ConfigurationError as error:  # field names and reasons only, never values
        print(f"invalid configuration: {error}", file=sys.stderr)
        return 2
    try:
        with psycopg.connect(settings.corpus_database_url.get_secret_value(), autocommit=True, connect_timeout=10) as conn:
            if args.activate is not None:
                ensure_schema(conn)
                activate(conn, args.activate)
                print(f"activated dur dataset {args.activate}")
                return 0
            result = load_dur(conn, args.snapshot_dir or settings.dur_snapshot_dir)
    except (psycopg.Error, DurLoadError, DurDataError, OSError) as error:
        # Error class and message only; the database URL (with credentials) is never printed.
        print(f"dur load failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    print(f"dur dataset {result.dataset_id} active ({'reused' if result.reused else 'new'}): "
          + ", ".join(f"{name} {count}" for name, count in result.counts.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
