"""Read-only lookups for the SQLite reference-data database."""

from __future__ import annotations

from collections import Counter
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
import sqlite3


_MAX_MATCHES = 5
_MIN_NAME_LENGTH = 2
_REFERENCE_TEXT_LIMIT = 600


@dataclass(frozen=True)
class MedicineReference:
    name_ko: str
    efficacy: str | None
    precautions: str | None


class SqliteReferenceLookup:
    """Look up approved local reference data without writing patient data.

    ``instr(question, name_ko)`` cannot use an index, so searching the 340MB
    table per question took about a second.  Disease and medicine names are
    instead loaded once and matched in memory (a few ms); only the matched
    rows' long text is read from SQLite.
    """

    def __init__(self, database_path: Path) -> None:
        self._uri = f"file:{database_path}?mode=ro"
        self._diseases = [
            (code, name)
            for code, name in self._rows("SELECT code, name_ko FROM diseases")
            if len(name) >= _MIN_NAME_LENGTH
        ]
        medicines = self._rows("SELECT kd_code, name_ko FROM medicines")
        # A normalized legacy code can map to more than one source record.
        # Do not choose arbitrarily in that case.
        code_counts = Counter(code for code, _name in medicines)
        self._medicines = [
            (code, name)
            for code, name in medicines
            if code_counts[code] == 1 and len(name) >= _MIN_NAME_LENGTH
        ]
        self._metadata: dict[str, str] = dict(self._rows("SELECT key, value FROM metadata"))

    def medicine(self, code: object) -> MedicineReference | None:
        if not isinstance(code, str) or not code.strip():
            return None
        rows = self._rows(
            "SELECT name_ko, efficacy, precautions FROM medicines WHERE kd_code = ?", (code.strip(),)
        )
        if len(rows) != 1:
            return None
        return MedicineReference(*rows[0])

    def disease(self, code: object) -> str | None:
        if not isinstance(code, str) or not code.strip():
            return None
        rows = self._rows("SELECT name_ko FROM diseases WHERE code = ?", (code.strip(),))
        return str(rows[0][0]) if rows else None

    def search_question(self, question: str) -> list[dict[str, str]]:
        """Find explicitly named records, bounded and without searching patient text."""
        if not question.strip():
            return []
        diseases = _named_in(self._diseases, question)
        medicines = _named_in(self._medicines, question)
        details: dict[str, tuple[str | None, str | None]] = {}
        if medicines:
            codes = [code for code, _name in medicines]
            details = {
                code: (efficacy, precautions)
                for code, efficacy, precautions in self._rows(
                    f"SELECT kd_code, efficacy, precautions FROM medicines WHERE kd_code IN ({','.join('?' * len(codes))})",
                    codes,
                )
            }
        metadata = self._metadata
        results = [
            {"type": "상병명 참조", "code": code, "name": name,
             "source": metadata.get("disease_source", "출처 미상"),
             "source_sha256": metadata.get("disease_sha256", ""),
             "generated_at": metadata.get("generated_at", "")}
            for code, name in diseases
        ]
        results.extend(
            {"type": "약품 일반 참조", "code": code, "name": name,
             "efficacy": (details.get(code, (None, None))[0] or "")[:_REFERENCE_TEXT_LIMIT],
             "precautions": (details.get(code, (None, None))[1] or "")[:_REFERENCE_TEXT_LIMIT],
             "source": metadata.get("medicine_source", "출처 미상"),
             "source_sha256": metadata.get("medicine_sha256", ""),
             "generated_at": metadata.get("generated_at", "")}
            for code, name in medicines
        )
        return results

    def _rows(self, sql: str, parameters: tuple[object, ...] | list[object] = ()) -> list[tuple]:
        # Reference enrichment is optional; an unavailable or malformed local
        # DB must not block a patient request.
        try:
            with closing(sqlite3.connect(self._uri, uri=True)) as connection:
                return connection.execute(sql, parameters).fetchall()
        except sqlite3.Error:
            return []


def _named_in(entries: list[tuple[str, str]], question: str) -> list[tuple[str, str]]:
    """Return the longest names that appear in the question, as the old SQL ordered them."""
    matches = [(code, name) for code, name in entries if name in question]
    return sorted(matches, key=lambda entry: (-len(entry[1]), entry[0]))[:_MAX_MATCHES]
