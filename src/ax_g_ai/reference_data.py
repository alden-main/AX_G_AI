"""Convert the legacy medicine and disease dictionaries to a queryable SQLite DB.

The legacy inputs are Python Pickle files, which must be treated as untrusted.
Only Pickles composed of the small set of opcodes required for built-in dict,
string, and float values are accepted before they are loaded.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from datetime import UTC, datetime
import hashlib
import math
from pathlib import Path
import pickle
import pickletools
import sqlite3
import uuid
from typing import Any


class ReferenceDataError(ValueError):
    """A reference-data source does not match the supported legacy format."""


# The supplied files use only these protocol-4 opcodes.  An allow-list avoids
# invoking Pickle data that can construct objects or call arbitrary functions.
_SAFE_PICKLE_OPCODES = frozenset(
    {
        "PROTO",
        "FRAME",
        "EMPTY_DICT",
        "MARK",
        "SHORT_BINUNICODE",
        "BINUNICODE",
        "MEMOIZE",
        "SETITEMS",
        "SETITEM",
        "BINGET",
        "LONG_BINGET",
        "BINFLOAT",
        "BININT",
        "BININT1",
        "BININT2",
        "NONE",
        "NEWTRUE",
        "NEWFALSE",
        "STOP",
    }
)


def convert_reference_pickles(
    *, medicine_pickle: Path, disease_pickle: Path, output: Path, replace: bool = False
) -> tuple[int, int]:
    """Create SQLite reference data from the supported legacy Pickle files.

    The output is written to a sibling temporary file and atomically published
    only after every schema and data operation succeeds.  Existing outputs are
    preserved unless ``replace`` is explicitly requested.
    """
    medicine = _load_builtin_dictionary(medicine_pickle, "medicine")
    disease = _load_builtin_dictionary(disease_pickle, "disease")
    medicine_rows = _medicine_rows(medicine)
    disease_rows = _disease_rows(disease)

    output = output.resolve()
    if output.exists() and not replace:
        raise FileExistsError(f"output already exists: {output}; pass --replace to overwrite it")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{uuid.uuid4().hex}.tmp")
    try:
        _write_database(
            temporary,
            medicine_rows,
            disease_rows,
            medicine_pickle,
            disease_pickle,
        )
        temporary.replace(output)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return len(medicine_rows), len(disease_rows)


def _load_builtin_dictionary(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} Pickle was not found: {path}")
    with path.open("rb") as source:
        for opcode, _argument, position in pickletools.genops(source):
            if opcode.name not in _SAFE_PICKLE_OPCODES:
                raise ReferenceDataError(
                    f"{label} Pickle contains unsupported opcode {opcode.name} at byte {position}"
                )
    # The opcode allow-list above ensures this load is limited to built-in data.
    with path.open("rb") as source:
        value = pickle.load(source)  # noqa: S301 - validated immediately above
    if not isinstance(value, dict):
        raise ReferenceDataError(f"{label} Pickle must contain a dictionary")
    return value


def _medicine_rows(value: Mapping[str, Any]) -> list[tuple[str, str, str, str | None, str | None, str | None]]:
    rows: list[tuple[str, str, str, str | None, str | None, str | None]] = []
    for code, record in value.items():
        # The legacy CSV has a small number of rows without ``kd_code``;
        # pandas serialised these as float NaN dictionary keys.  They cannot
        # identify a medicine and are deliberately excluded from SQLite.
        if not isinstance(code, str) or not code.strip():
            continue
        if not isinstance(record, Mapping):
            raise ReferenceDataError(f"medicine record {code!r} must be an object")
        name = _required_text(record.get("name_ko"), f"medicine {code!r}.name_ko")
        rows.append(
            (
                code,
                code.strip(),
                name,
                _optional_text(record.get("efficacy")),
                _optional_text(record.get("usagex")),
                _optional_text(record.get("carefully")),
            )
        )
    return rows


def _disease_rows(value: Mapping[str, Any]) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for code, name in value.items():
        if not isinstance(code, str) or not code.strip():
            raise ReferenceDataError("disease codes must be non-blank strings")
        rows.append((code.strip(), _required_text(name, f"disease {code!r}")))
    return rows


def _required_text(value: object, field: str) -> str:
    text = _optional_text(value)
    if text is None:
        raise ReferenceDataError(f"{field} must be a non-blank string")
    return text


def _optional_text(value: object) -> str | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if not isinstance(value, str):
        raise ReferenceDataError("reference values must be strings, null, or NaN")
    text = value.strip()
    return text or None


def _write_database(
    path: Path,
    medicine_rows: list[tuple[str, str, str, str | None, str | None, str | None]],
    disease_rows: list[tuple[str, str]],
    medicine_pickle: Path,
    disease_pickle: Path,
) -> None:
    with sqlite3.connect(path) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.executescript(
            """
            CREATE TABLE metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE diseases (
                code TEXT PRIMARY KEY,
                name_ko TEXT NOT NULL
            ) WITHOUT ROWID;
            CREATE TABLE medicines (
                source_kd_code TEXT PRIMARY KEY,
                kd_code TEXT NOT NULL,
                name_ko TEXT NOT NULL,
                efficacy TEXT,
                usage TEXT,
                precautions TEXT
            ) WITHOUT ROWID;
            CREATE INDEX diseases_name_ko_idx ON diseases(name_ko);
            CREATE INDEX medicines_kd_code_idx ON medicines(kd_code);
            CREATE INDEX medicines_name_ko_idx ON medicines(name_ko);
            """
        )
        metadata = (
            ("schema_version", "1"),
            ("generated_at", datetime.now(UTC).isoformat()),
            ("medicine_source", medicine_pickle.name),
            ("medicine_sha256", _sha256(medicine_pickle)),
            ("disease_source", disease_pickle.name),
            ("disease_sha256", _sha256(disease_pickle)),
        )
        connection.executemany("INSERT INTO metadata(key, value) VALUES (?, ?)", metadata)
        connection.executemany("INSERT INTO diseases(code, name_ko) VALUES (?, ?)", disease_rows)
        connection.executemany(
            """INSERT INTO medicines(
                source_kd_code, kd_code, name_ko, efficacy, usage, precautions
            ) VALUES (?, ?, ?, ?, ?, ?)""",
            medicine_rows,
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert validated legacy medicine and disease Pickles into SQLite."
    )
    parser.add_argument("--medicine", type=Path, default=Path("medicine_dict.pkl"))
    parser.add_argument("--disease", type=Path, default=Path("disease_dict.pkl"))
    parser.add_argument("--output", type=Path, default=Path("reference_data.sqlite"))
    parser.add_argument("--replace", action="store_true", help="replace an existing output database")
    args = parser.parse_args()
    medicine_count, disease_count = convert_reference_pickles(
        medicine_pickle=args.medicine,
        disease_pickle=args.disease,
        output=args.output,
        replace=args.replace,
    )
    print(f"created {args.output} ({medicine_count} medicines, {disease_count} diseases)")


if __name__ == "__main__":
    main()
