"""Read-only lookups for the SQLite reference-data database."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sqlite3


@dataclass(frozen=True)
class MedicineReference:
    name_ko: str
    efficacy: str | None
    precautions: str | None


class SqliteReferenceLookup:
    """Look up approved local reference data without writing patient data."""

    def __init__(self, database_path: Path) -> None:
        self._database_path = database_path

    def medicine(self, code: object) -> MedicineReference | None:
        if not isinstance(code, str) or not code.strip():
            return None
        # A normalized legacy code can map to more than one source record.
        # Do not choose arbitrarily in that case.
        try:
            with sqlite3.connect(f"file:{self._database_path}?mode=ro", uri=True) as connection:
                rows = connection.execute(
                    """SELECT name_ko, efficacy, precautions
                    FROM medicines WHERE kd_code = ?""",
                    (code.strip(),),
                ).fetchall()
        except sqlite3.Error:
            # Reference enrichment is optional; an unavailable or malformed
            # local DB must not cause a patient Snapshot to be treated as data.
            return None
        if len(rows) != 1:
            return None
        name, efficacy, precautions = rows[0]
        return MedicineReference(name, efficacy, precautions)

    def disease(self, code: object) -> str | None:
        if not isinstance(code, str) or not code.strip():
            return None
        try:
            with sqlite3.connect(f"file:{self._database_path}?mode=ro", uri=True) as connection:
                row = connection.execute(
                    "SELECT name_ko FROM diseases WHERE code = ?", (code.strip(),)
                ).fetchone()
        except sqlite3.Error:
            return None
        return None if row is None else str(row[0])
