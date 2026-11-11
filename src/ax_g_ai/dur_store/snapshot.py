"""Convert the HIRA DUR product lists (공공데이터포털 CSV, cp949) into a compact load snapshot.

Source: 건강보험심사평가원_의약품안전사용서비스(DUR) 의약품 목록 (공공데이터포털 15127983,
공공누리 제1유형 — 출처표시 후 상업 이용 가능). The product lists expand each ingredient rule to
every product; the snapshot keeps one row per ingredient rule plus a product → ingredient map
(about 870k CSV rows → 70k rows), one JSONL file per ``dur`` table with the same columns.

The snapshot is what ships in the image; ``ax_g_ai.dur_store.loader`` loads it into PostgreSQL
(the same flow as ``text_data/corpus`` → ``corpus-loader``). The raw CSVs stay out of the image.

Usage (project root)::

    uv run ax-g-ai-build-dur --input dur --output data/dur [--replace]

Only contraindication / caution lists are converted (병용금기, 연령금기, 임부금기, 노인주의).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping

FORMAT = "ax-g-dur-snapshot/1"
MANIFEST = "manifest.json"
SOURCE = {
    "id": "KR-HIRA-DUR-2026-06",
    "title": "건강보험심사평가원 의약품안전사용서비스(DUR) 의약품 목록(2026.6) — 공공데이터포털",
    "url": "https://www.data.go.kr/data/15127983/fileData.do",
    "license": "KOGL Type 1 (공공누리 제1유형 — 출처표시)",
    "region": "KR",
}
FILES = {  # NFC file-name fragment → list kind
    "병용금기": "pair",
    "연령금기": "age",
    "임부금기": "pregnancy",
    "노인주의(해열진통소염제)": "elderly_nsaid",
    "노인주의 품목": "elderly",
}
# Snapshot file = PostgreSQL table ``dur.<name>``; the loader and reader use these column lists.
TABLE_COLUMNS: dict[str, tuple[str, ...]] = {
    "product": ("product_code", "product_name", "ingredient_code", "ingredient_name", "company", "insured"),
    "pair_rule": ("ingredient_a", "ingredient_b", "name_a", "name_b", "detail", "note", "notice_no", "notice_date"),
    "age_rule": ("ingredient_code", "ingredient_name", "age", "unit", "condition", "detail", "notice_no", "notice_date"),
    "pregnancy_rule": ("ingredient_code", "ingredient_name", "grade", "detail", "notice_no", "notice_date"),
    "elderly_rule": ("ingredient_code", "ingredient_name", "category", "detail", "notice_no", "notice_date"),
}


class DurDataError(ValueError):
    """Input files are missing, have unexpected columns, or the snapshot is inconsistent."""


@dataclass(frozen=True)
class Snapshot:
    manifest: Mapping[str, Any]
    rows: Mapping[str, list[dict[str, Any]]]

    @property
    def source(self) -> Mapping[str, str]:
        return self.manifest["source"]

    @property
    def content_sha256(self) -> str:
        return self.manifest["content_sha256"]


def _kind_of(name: str) -> str | None:
    normalized = unicodedata.normalize("NFC", name)
    if not normalized.lower().endswith(".csv"):
        return None
    for fragment, kind in FILES.items():
        if fragment in normalized:
            return kind
    return None


def find_files(directory: Path) -> dict[str, Path]:
    found: dict[str, Path] = {}
    for entry in sorted(os.listdir(directory)):
        kind = _kind_of(entry)
        if kind is not None:
            if kind in found:
                raise DurDataError(f"같은 종류의 파일이 둘 이상: {kind}")
            found[kind] = directory / entry
    missing = sorted(set(FILES.values()) - set(found))
    if missing:
        raise DurDataError(f"DUR 파일 없음: {', '.join(missing)}")
    return found


def _rows(path: Path, required: set[str]) -> Iterator[dict[str, str]]:
    with path.open(encoding="cp949", errors="strict", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or ())
        if not required <= columns:
            raise DurDataError(f"{path.name}: 필요한 열 없음 {sorted(required - columns)}")
        for row in reader:
            yield {key: (value or "").strip() for key, value in row.items() if key is not None}


def _label(value: str) -> str:
    """'세 (3)' → '세', '미만 (1)' → '미만'."""
    return value.split("(")[0].strip()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def convert(directory: Path) -> tuple[dict[str, list[dict[str, Any]]], dict[str, dict[str, str]]]:
    """(rows by table, input files) from the CSV folder. Rows are deduplicated and sorted."""
    files = find_files(directory)
    products: dict[str, dict[str, Any]] = {}

    def product(code: str, name: str, ingredient_code: str, ingredient_name: str, company: str, insured: str) -> None:
        code = code.strip()
        if code and code not in products:
            products[code] = {"product_code": code, "product_name": name, "ingredient_code": ingredient_code.strip(),
                              "ingredient_name": ingredient_name, "company": company, "insured": insured}

    pairs: dict[tuple, dict[str, Any]] = {}
    for row in _rows(files["pair"], {"성분코드A", "성분코드B", "제품코드A", "제품코드B", "상세정보"}):
        product(row["제품코드A"], row["제품명A"], row["성분코드A"], row["성분명A"], row["업체명A"], row["급여여부A"])
        product(row["제품코드B"], row["제품명B"], row["성분코드B"], row["성분명B"], row["업체명B"], row["급여여부B"])
        a, b = (row["성분코드A"], row["성분명A"]), (row["성분코드B"], row["성분명B"])
        if b[0] < a[0]:
            a, b = b, a
        key = (a[0], b[0], row["상세정보"], row.get("비고", ""))
        pairs.setdefault(key, {"ingredient_a": a[0], "ingredient_b": b[0], "name_a": a[1], "name_b": b[1],
                               "detail": row["상세정보"], "note": row.get("비고", ""),
                               "notice_no": row["고시번호"], "notice_date": row["고시일자"]})

    ages: dict[tuple, dict[str, Any]] = {}
    for row in _rows(files["age"], {"성분코드", "제품코드", "특정연령", "특정연령단위", "연령처리조건", "상세정보"}):
        product(row["제품코드"], row["제품명"], row["성분코드"], row["성분명"], row.get("업체명", ""), row.get("급여여부", ""))
        try:
            age = float(row["특정연령"])
        except ValueError:
            continue
        key = (row["성분코드"], age, _label(row["특정연령단위"]), _label(row["연령처리조건"]), row["상세정보"])
        ages.setdefault(key, {"ingredient_code": row["성분코드"], "ingredient_name": row["성분명"], "age": age,
                              "unit": key[2], "condition": key[3], "detail": row["상세정보"],
                              "notice_no": row.get("고시번호", ""), "notice_date": row.get("고시일자", "")})

    pregnancy: dict[tuple, dict[str, Any]] = {}
    for row in _rows(files["pregnancy"], {"성분코드", "제품코드", "금기등급", "상세정보"}):
        product(row["제품코드"], row["제품명"], row["성분코드"], row["성분명"], row.get("업체명", ""), row.get("급여여부", ""))
        key = (row["성분코드"], row["금기등급"], row["상세정보"])
        pregnancy.setdefault(key, {"ingredient_code": row["성분코드"], "ingredient_name": row["성분명"], "grade": row["금기등급"],
                                   "detail": row["상세정보"], "notice_no": row.get("고시번호", ""), "notice_date": row.get("고시일자", "")})

    elderly: dict[tuple, dict[str, Any]] = {}
    for kind, category in (("elderly", "노인주의"), ("elderly_nsaid", "노인주의(해열진통소염제)")):
        for row in _rows(files[kind], {"성분코드", "제품코드", "약품상세정보"}):
            product(row["제품코드"], row["제품명"], row["성분코드"], row["성분명"], row.get("업소명", row.get("업체명", "")), row.get("급여여부", ""))
            key = (row["성분코드"], category, row["약품상세정보"])
            elderly.setdefault(key, {"ingredient_code": row["성분코드"], "ingredient_name": row["성분명"], "category": category,
                                     "detail": row["약품상세정보"], "notice_no": row.get("공고번호", ""), "notice_date": row.get("공고일자", "")})

    def ordered(rows: dict[Any, dict[str, Any]]) -> list[dict[str, Any]]:
        return [rows[key] for key in sorted(rows, key=lambda item: tuple(map(str, item if isinstance(item, tuple) else (item,))))]

    tables = {"product": ordered(products), "pair_rule": ordered(pairs), "age_rule": ordered(ages),
              "pregnancy_rule": ordered(pregnancy), "elderly_rule": ordered(elderly)}
    inputs = {kind: {"file": unicodedata.normalize("NFC", path.name), "sha256": _sha256_file(path)} for kind, path in files.items()}
    return tables, inputs


def _table_bytes(rows: list[dict[str, Any]], columns: tuple[str, ...]) -> bytes:
    return "".join(
        json.dumps({column: row[column] for column in columns}, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows
    ).encode("utf-8")


def _content_sha(source: Mapping[str, Any], tables: Mapping[str, Mapping[str, Any]]) -> str:
    """Dataset identity: same source + same table contents → same hash (build time is excluded)."""
    identity = {"source": source, "tables": {name: [info["count"], info["sha256"]] for name, info in tables.items()}}
    return hashlib.sha256(json.dumps(identity, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def build(directory: Path, output: Path) -> dict[str, int]:
    """Write ``output/<table>.jsonl`` + ``manifest.json`` atomically (a temp folder is swapped in)."""
    tables, inputs = convert(directory)
    temporary = output.with_name(output.name + ".tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    table_info: dict[str, dict[str, Any]] = {}
    for name, columns in TABLE_COLUMNS.items():
        data = _table_bytes(tables[name], columns)
        (temporary / f"{name}.jsonl").write_bytes(data)
        table_info[name] = {"file": f"{name}.jsonl", "count": len(tables[name]), "sha256": hashlib.sha256(data).hexdigest()}
    manifest = {
        "format": FORMAT, "built_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": SOURCE, "inputs": inputs, "tables": table_info, "content_sha256": _content_sha(SOURCE, table_info),
    }
    (temporary / MANIFEST).write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if output.exists():
        shutil.rmtree(output)
    os.replace(temporary, output)
    return {name: info["count"] for name, info in table_info.items()}


def read_snapshot(directory: Path) -> Snapshot:
    """Read and verify a snapshot (format, per-file SHA-256 and row counts)."""
    path = directory / MANIFEST
    if not path.is_file():
        raise DurDataError(f"DUR 스냅샷 없음: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("format") != FORMAT:
        raise DurDataError(f"지원하지 않는 DUR 스냅샷 형식: {manifest.get('format')!r}")
    rows: dict[str, list[dict[str, Any]]] = {}
    for name, columns in TABLE_COLUMNS.items():
        info = (manifest.get("tables") or {}).get(name)
        if not isinstance(info, Mapping):
            raise DurDataError(f"DUR 스냅샷에 {name} 표가 없음")
        data = (directory / info["file"]).read_bytes()
        if hashlib.sha256(data).hexdigest() != info["sha256"]:
            raise DurDataError(f"{info['file']}: SHA-256 불일치(스냅샷 손상 또는 수기 편집)")
        rows[name] = [json.loads(line) for line in data.decode("utf-8").splitlines() if line.strip()]
        if len(rows[name]) != info["count"]:
            raise DurDataError(f"{info['file']}: {len(rows[name])}행, manifest {info['count']}행")
        if rows[name] and set(rows[name][0]) != set(columns):
            raise DurDataError(f"{info['file']}: 열이 {list(columns)}와 다름")
    if _content_sha(manifest["source"], manifest["tables"]) != manifest.get("content_sha256"):
        raise DurDataError("manifest content_sha256 불일치")
    return Snapshot(manifest, rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=Path("dur"), help="DUR CSV 폴더")
    parser.add_argument("--output", type=Path, default=Path("data/dur"), help="스냅샷 폴더")
    parser.add_argument("--replace", action="store_true", help="기존 스냅샷을 교체")
    args = parser.parse_args(argv)
    if args.output.exists() and not args.replace:
        print(f"{args.output} 가 이미 있습니다. 교체하려면 --replace", file=sys.stderr)
        return 1
    try:
        counts = build(args.input, args.output)
    except (DurDataError, UnicodeDecodeError) as error:
        print(f"변환 실패: {error}", file=sys.stderr)
        return 1
    print(f"{args.output}: " + ", ".join(f"{key} {value}" for key, value in counts.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
