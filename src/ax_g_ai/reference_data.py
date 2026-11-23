"""예전 약품·질환 사전(Pickle)을 조회 가능한 SQLite DB로 변환한다.

설계상 위치: 오프라인 데이터 준비 도구. ``ax-g-ai-reference-db`` 명령(``main``)으로 실행하며, 만든 DB는
런타임의 ``SqliteReferenceLookup``(``AX_G_REFERENCE_DATA_DB_PATH``)이 읽는다.

역할: 예전 입력은 Python Pickle 파일이므로 신뢰할 수 없는 데이터로 다룬다. 내장 dict·문자열·실수 값에 필요한
소수의 opcode로만 이루어진 Pickle만 허용하고, 그 검사를 통과한 뒤에야 불러온다.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import pickle
import pickletools
import sqlite3
import uuid
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class ReferenceDataError(ValueError):
    """참조 데이터 원본이 지원하는 예전 형식과 맞지 않는다.

    역할: 허용되지 않은 opcode, dict가 아닌 최상위 값, 빈 코드·이름, 문자열이 아닌 값 등을 알린다.
    """


# 제공된 파일은 protocol 4의 아래 opcode만 쓴다. 허용 목록으로 객체를 만들거나 임의 함수를 호출할 수 있는
# Pickle 데이터가 실행되지 않게 한다.
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
    """지원하는 예전 Pickle 파일로 SQLite 참조 데이터를 만든다.

    역할: 두 Pickle을 검사·변환한 뒤 같은 폴더의 임시 파일에 쓰고, 모든 스키마·데이터 작업이 성공한 뒤에만
    결과 경로로 원자적으로 교체한다. 실패하면 임시 파일을 지운다. ``replace``를 명시하지 않으면 기존 결과를 보존한다.
    입력:
        medicine_pickle: 약품 사전 Pickle 경로.
        disease_pickle: 질환 사전 Pickle 경로.
        output: 만들 SQLite 파일 경로.
        replace: 기존 결과 파일을 덮어쓸지 여부.
    출력: (저장한 약품 행 수, 저장한 질환 행 수).
    예외: ``FileExistsError``(결과가 이미 있고 ``replace``가 아님), ``FileNotFoundError``(입력 없음),
        ``ReferenceDataError``(형식 불일치).
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
    """opcode 허용 목록을 통과한 Pickle만 불러와 dict로 돌려준다.

    역할: 먼저 ``pickletools.genops``로 모든 opcode를 훑어 허용 목록 밖의 것이 있으면 거부하고, 통과하면 불러온다.
    입력:
        path: Pickle 파일 경로.
        label: 오류 메시지에 쓸 이름("medicine" / "disease").
    출력: 불러온 dict.
    예외: ``FileNotFoundError``(파일 없음), ``ReferenceDataError``(허용되지 않은 opcode 또는 dict가 아닌 값).
    """
    if not path.is_file():
        raise FileNotFoundError(f"{label} Pickle was not found: {path}")
    with path.open("rb") as source:
        for opcode, _argument, position in pickletools.genops(source):
            if opcode.name not in _SAFE_PICKLE_OPCODES:
                raise ReferenceDataError(
                    f"{label} Pickle contains unsupported opcode {opcode.name} at byte {position}"
                )
    # 위의 opcode 허용 목록 검사 덕분에 이 load는 내장 데이터만 만든다.
    with path.open("rb") as source:
        value = pickle.load(source)  # noqa: S301 - 바로 위에서 검증함
    if not isinstance(value, dict):
        raise ReferenceDataError(f"{label} Pickle must contain a dictionary")
    return value


def _medicine_rows(value: Mapping[str, Any]) -> list[tuple[str, str, str, str | None, str | None, str | None]]:
    """약품 사전을 ``medicines`` 테이블 행으로 바꾼다.

    역할: 코드가 빈 값이거나 문자열이 아닌 항목은 건너뛰고, 원래 코드와 앞뒤 공백을 없앤 코드를 함께 저장한다.
    입력:
        value: ``kd_code`` → {``name_ko``, ``efficacy``, ``usagex``, ``carefully``} 사전.
    출력: (원래 코드, 정리한 코드, 한글명, 효능, 용법, 주의사항) 튜플 목록.
    예외: ``ReferenceDataError`` — 항목이 객체가 아니거나 한글명이 없거나 값이 문자열·null·NaN이 아님.
    """
    rows: list[tuple[str, str, str, str | None, str | None, str | None]] = []
    for code, record in value.items():
        # 예전 CSV에는 ``kd_code``가 없는 행이 조금 있고, pandas가 이를 float NaN 키로 직렬화했다.
        # 약품을 식별할 수 없으므로 일부러 SQLite에서 뺀다.
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
    """질환 사전(코드 → 한글명)을 (정리한 코드, 이름) 행으로 바꾼다. 빈 코드·이름은 ``ReferenceDataError``다."""
    rows: list[tuple[str, str]] = []
    for code, name in value.items():
        if not isinstance(code, str) or not code.strip():
            raise ReferenceDataError("disease codes must be non-blank strings")
        rows.append((code.strip(), _required_text(name, f"disease {code!r}")))
    return rows


def _required_text(value: object, field: str) -> str:
    """``_optional_text``와 같되 값이 없으면 ``ReferenceDataError``를 낸다."""
    text = _optional_text(value)
    if text is None:
        raise ReferenceDataError(f"{field} must be a non-blank string")
    return text


def _optional_text(value: object) -> str | None:
    """앞뒤 공백을 없앤 문자열을 돌려준다. ``None``·NaN·빈 문자열은 ``None``, 그 밖의 타입은 ``ReferenceDataError``다."""
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
    """새 SQLite 파일에 스키마를 만들고 메타데이터·질환·약품 행을 넣는다.

    역할: ``metadata``(스키마 버전, 생성 시각, 원본 파일 이름·SHA-256), ``diseases``, ``medicines`` 테이블과
    이름·코드 색인을 만들고 행을 채운다.
    입력:
        path: 쓸 SQLite 파일 경로(임시 파일).
        medicine_rows: ``_medicine_rows`` 결과.
        disease_rows: ``_disease_rows`` 결과.
        medicine_pickle: 메타데이터에 기록할 약품 원본 경로.
        disease_pickle: 메타데이터에 기록할 질환 원본 경로.
    출력: 없음.
    """
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
            CREATE INDEX medicines_kd_code_idx ON medicines(kd_code, name_ko);
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
    """파일의 SHA-256 16진 문자열."""
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main() -> None:
    """``ax-g-ai-reference-db`` 명령의 진입점.

    역할: ``--medicine``·``--disease``·``--output``·``--replace`` 인자를 읽어 ``convert_reference_pickles``를 실행하고
    만든 행 수를 출력한다.
    입력: 없음(명령행 인자).
    출력: 없음(결과 요약을 표준 출력에 쓴다).
    """
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
