"""Versioned datasets in PostgreSQL: one active row, atomic switch, bounded history.

The corpus (``corpus.corpus_release``) and DUR (``dur.dataset``) loaders store each load as
a new row and switch ``is_active`` in one transaction; rolling back is activating an older
row.

Schema changes are numbered SQL migrations in each store package (``migrations/NNNN_name.sql``).
``ensure_schema`` applies the ones not yet recorded in ``<schema>.schema_version``, in order, in one
transaction under an advisory lock, so two loaders starting together cannot apply one twice and a
failing migration leaves the schema as it was. A database created before migrations existed has
version 1 recorded, which is ``0001_initial.sql``. A migration must therefore be valid inside a
transaction (no ``CREATE INDEX CONCURRENTLY``), and an applied migration is never edited: add a new one.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from importlib import resources
from typing import Any

import psycopg
from psycopg import sql
from psycopg.rows import tuple_row

_MIGRATION_NAME = re.compile(r"^(\d{4})_[a-z0-9_]+\.sql$")


def scalar(cursor: psycopg.Cursor[Any]) -> Any:
    """The first column of the one row a statement returns (``count(*)``, ``RETURNING id``); no row is an error."""
    row = cursor.fetchone()
    if row is None:
        raise psycopg.DataError("statement returned no row")
    return row[0] if isinstance(row, (tuple, list)) else next(iter(row.values()))


class MigrationError(RuntimeError):
    """The packaged migrations are malformed (duplicate or unparsable version numbers)."""


def migrations(package: str) -> list[tuple[int, str, str]]:
    """(version, file name, SQL) of every migration shipped in ``package``/migrations, in version order."""
    found: dict[int, tuple[int, str, str]] = {}
    for entry in resources.files(package).joinpath("migrations").iterdir():
        if not entry.name.endswith(".sql"):
            continue
        match = _MIGRATION_NAME.match(entry.name)
        if match is None:
            raise MigrationError(f"{package}: migration file name must be NNNN_name.sql: {entry.name}")
        version = int(match.group(1))
        if version in found:
            raise MigrationError(f"{package}: two migrations with version {version}")
        found[version] = (version, entry.name, entry.read_text(encoding="utf-8"))
    return [found[version] for version in sorted(found)]


@dataclass(frozen=True)
class ReleaseTable:
    schema: str              # PostgreSQL schema, e.g. "corpus"
    table: str               # release table, e.g. "corpus_release"
    id_column: str           # e.g. "release_id"
    schema_package: str      # package holding ``migrations/NNNN_name.sql``
    keep: int                # inactive releases kept besides the newest ones
    error: type[Exception]   # raised for an unknown release id
    label: str               # "release" / "dataset", for messages

    @property
    def _table(self) -> sql.Composable:
        return sql.Identifier(self.schema, self.table)

    def ensure_schema(self, conn: psycopg.Connection) -> list[int]:
        """Apply the pending migrations; returns their versions (empty when the schema is current)."""
        schema = sql.Identifier(self.schema)
        versions = sql.Identifier(self.schema, "schema_version")
        pending: list[int] = []
        with conn.transaction():
            cursor = conn.cursor(row_factory=tuple_row)
            cursor.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (f"ax_g_ai.migrate.{self.schema}",))
            cursor.execute(sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(schema))
            cursor.execute(sql.SQL(
                "CREATE TABLE IF NOT EXISTS {} (version int PRIMARY KEY, applied_at timestamptz NOT NULL DEFAULT now())"
            ).format(versions))
            applied = {row[0] for row in cursor.execute(sql.SQL("SELECT version FROM {}").format(versions)).fetchall()}
            for version, _name, script in migrations(self.schema_package):
                if version in applied:
                    continue
                cursor.execute(script)  # type: ignore[arg-type]  # trusted file shipped in the package
                cursor.execute("SET search_path TO DEFAULT")  # scripts set their own schema first
                cursor.execute(sql.SQL("INSERT INTO {} (version) VALUES (%s)").format(versions), (version,))
                pending.append(version)
        return pending

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
