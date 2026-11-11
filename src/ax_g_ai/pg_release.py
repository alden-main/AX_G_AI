"""Versioned datasets in PostgreSQL: one active row, atomic switch, bounded history.

The corpus (``corpus.corpus_release``) and DUR (``dur.dataset``) loaders store each load as
a new row and switch ``is_active`` in one transaction; rolling back is activating an older
row. The schema file of each store is applied once, when its ``schema_version`` table is missing.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import resources

import psycopg
from psycopg import sql


@dataclass(frozen=True)
class ReleaseTable:
    schema: str              # PostgreSQL schema, e.g. "corpus"
    table: str               # release table, e.g. "corpus_release"
    id_column: str           # e.g. "release_id"
    schema_package: str      # package holding ``schema.sql``
    keep: int                # inactive releases kept besides the newest ones
    error: type[Exception]   # raised for an unknown release id
    label: str               # "release" / "dataset", for messages

    @property
    def _table(self) -> sql.Composable:
        return sql.Identifier(self.schema, self.table)

    def ensure_schema(self, conn: psycopg.Connection) -> None:
        version_table = f"{self.schema}.schema_version"
        if conn.execute("SELECT to_regclass(%s)", (version_table,)).fetchone()[0] is None:
            script = resources.files(self.schema_package).joinpath("schema.sql").read_text(encoding="utf-8")
            with conn.transaction():
                conn.execute(script)  # type: ignore[arg-type]  # trusted file shipped in the package
                conn.execute("SET search_path TO DEFAULT")

    def activate(self, conn: psycopg.Connection, release_id: int) -> None:
        column = sql.Identifier(self.id_column)
        with conn.transaction():
            exists = conn.execute(sql.SQL("SELECT 1 FROM {} WHERE {} = %s").format(self._table, column), (release_id,))
            if exists.fetchone() is None:
                raise self.error(f"{self.label} {release_id} does not exist")
            conn.execute(sql.SQL("UPDATE {} SET is_active = false WHERE is_active AND {} <> %s").format(self._table, column),
                         (release_id,))
            conn.execute(sql.SQL("UPDATE {} SET is_active = true WHERE {} = %s").format(self._table, column), (release_id,))

    def prune(self, conn: psycopg.Connection) -> None:
        """Delete inactive releases older than the newest ``keep`` (child rows cascade)."""
        column = sql.Identifier(self.id_column)
        with conn.transaction():
            conn.execute(
                sql.SQL("DELETE FROM {table} WHERE NOT is_active AND {id} NOT IN ("
                        "SELECT {id} FROM {table} ORDER BY {id} DESC LIMIT %s)").format(table=self._table, id=column),
                (self.keep,),
            )
