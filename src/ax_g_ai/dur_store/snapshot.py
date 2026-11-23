"""심평원 DUR 품목 목록(공공데이터포털 CSV, cp949)을 적재용 압축 스냅샷으로 변환하는 모듈과 CLI.

설계상 위치: DUR 저장 계층의 빌드 단계(오프라인 도구 ``ax-g-ai-build-dur``). 만든 스냅샷은 이미지에 실리고,
``ax_g_ai.dur_store.loader``가 PostgreSQL에 적재하며(``text_data/corpus`` → ``corpus-loader``와 같은 흐름),
``loader``·``reader``는 여기의 ``TABLE_COLUMNS``·``read_snapshot``을 쓴다. 원본 CSV는 이미지에 넣지 않는다.

역할: 출처는 건강보험심사평가원_의약품안전사용서비스(DUR) 의약품 목록(공공데이터포털 15127983, 공공누리 제1유형 —
출처표시 후 상업 이용 가능)이다. 품목 목록은 성분 규칙 하나를 모든 제품으로 펼쳐 두었으므로, 스냅샷은 성분 규칙마다
한 행과 제품 → 성분 매핑만 남긴다(CSV 약 87만 행 → 7만 행). ``dur`` 표마다 같은 열을 가진 JSONL 파일 하나를 만든다.

사용법(프로젝트 루트)::

    uv run ax-g-ai-build-dur --input dur --output data/dur [--replace]

금기·주의 목록만 변환한다(병용금기, 연령금기, 임부금기, 노인주의).
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
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ax_g_ai.jsonl import parse_jsonl

FORMAT = "ax-g-dur-snapshot/1"
MANIFEST = "manifest.json"
SOURCE = {
    "id": "KR-HIRA-DUR-2026-06",
    "title": "건강보험심사평가원 의약품안전사용서비스(DUR) 의약품 목록(2026.6) — 공공데이터포털",
    "url": "https://www.data.go.kr/data/15127983/fileData.do",
    "license": "KOGL Type 1 (공공누리 제1유형 — 출처표시)",
    "region": "KR",
}
FILES = {  # NFC로 정규화한 파일 이름 조각 → 목록 종류
    "병용금기": "pair",
    "연령금기": "age",
    "임부금기": "pregnancy",
    "노인주의(해열진통소염제)": "elderly_nsaid",
    "노인주의 품목": "elderly",
}
# 스냅샷 파일 = PostgreSQL 표 ``dur.<name>``. 적재기와 리더가 이 열 목록을 쓴다.
TABLE_COLUMNS: dict[str, tuple[str, ...]] = {
    "product": ("product_code", "product_name", "ingredient_code", "ingredient_name", "company", "insured"),
    "pair_rule": ("ingredient_a", "ingredient_b", "name_a", "name_b", "detail", "note", "notice_no", "notice_date"),
    "age_rule": ("ingredient_code", "ingredient_name", "age", "unit", "condition", "detail", "notice_no", "notice_date"),
    "pregnancy_rule": ("ingredient_code", "ingredient_name", "grade", "detail", "notice_no", "notice_date"),
    "elderly_rule": ("ingredient_code", "ingredient_name", "category", "detail", "notice_no", "notice_date"),
}


class DurDataError(ValueError):
    """입력 파일이 없거나 열이 예상과 다르거나 스냅샷이 일관되지 않을 때의 오류.

    역할: 빌드 CLI와 적재 CLI가 잡아 종료 코드 1로 보고하는 데이터 오류.
    """


@dataclass(frozen=True)
class Snapshot:
    """검증을 마친 DUR 스냅샷(manifest + 표별 행).

    역할: ``read_snapshot``의 결과로 적재기에 전달되며, 출처와 내용 해시를 속성으로 제공한다.
    """

    manifest: Mapping[str, Any]
    rows: Mapping[str, list[dict[str, Any]]]

    @property
    def source(self) -> Mapping[str, str]:
        """manifest의 출처 정보."""
        return self.manifest["source"]

    @property
    def content_sha256(self) -> str:
        """데이터셋 동일성 해시(빌드 시각 제외)."""
        return self.manifest["content_sha256"]


def _kind_of(name: str) -> str | None:
    """파일 이름(NFC 정규화)이 ``.csv``이고 ``FILES`` 조각을 포함하면 그 목록 종류, 아니면 None."""
    normalized = unicodedata.normalize("NFC", name)
    if not normalized.lower().endswith(".csv"):
        return None
    for fragment, kind in FILES.items():
        if fragment in normalized:
            return kind
    return None


def find_files(directory: Path) -> dict[str, Path]:
    """CSV 폴더에서 목록 종류별 입력 파일을 찾는다.

    역할: 폴더 항목을 이름순으로 훑어 ``_kind_of``로 분류한다.
    입력:
        directory: DUR CSV 폴더.
    출력: {목록 종류: 파일 경로}.
    예외: 같은 종류 파일이 둘 이상이거나 필요한 종류가 없으면 ``DurDataError``.
    """
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
    """cp949 CSV의 행을 앞뒤 공백을 지운 dict로 내보낸다. 필수 열이 없으면 ``DurDataError``."""
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
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def convert(directory: Path) -> tuple[dict[str, list[dict[str, Any]]], dict[str, dict[str, str]]]:
    """CSV 폴더를 표별 스냅샷 행으로 변환한다.

    역할: 목록마다 제품 행을 제품코드 기준 처음 본 것으로 모으고, 규칙은 키로 중복을 없앤다(병용금기는 성분코드 A·B를
    정렬해 한 쌍으로, 연령금기는 숫자가 아닌 특정연령 행을 건너뛰고 단위·조건 라벨에서 괄호 코드를 뗀다, 노인주의 두
    목록은 ``category``로 구분해 합친다). 각 표는 키 문자열 순으로 정렬한다.
    입력:
        directory: DUR CSV 폴더.
    출력: (표별 행, 입력 파일별 {file, sha256}).
    예외: 파일 누락·열 누락이면 ``DurDataError``, cp949로 읽을 수 없으면 ``UnicodeDecodeError``.
    """
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
        age_key = (row["성분코드"], age, _label(row["특정연령단위"]), _label(row["연령처리조건"]), row["상세정보"])
        ages.setdefault(age_key, {"ingredient_code": row["성분코드"], "ingredient_name": row["성분명"], "age": age,
                                  "unit": age_key[2], "condition": age_key[3], "detail": row["상세정보"],
                              "notice_no": row.get("고시번호", ""), "notice_date": row.get("고시일자", "")})

    pregnancy: dict[tuple, dict[str, Any]] = {}
    for row in _rows(files["pregnancy"], {"성분코드", "제품코드", "금기등급", "상세정보"}):
        product(row["제품코드"], row["제품명"], row["성분코드"], row["성분명"], row.get("업체명", ""), row.get("급여여부", ""))
        pregnancy_key = (row["성분코드"], row["금기등급"], row["상세정보"])
        pregnancy.setdefault(pregnancy_key, {"ingredient_code": row["성분코드"], "ingredient_name": row["성분명"], "grade": row["금기등급"],
                                   "detail": row["상세정보"], "notice_no": row.get("고시번호", ""), "notice_date": row.get("고시일자", "")})

    elderly: dict[tuple, dict[str, Any]] = {}
    for kind, category in (("elderly", "노인주의"), ("elderly_nsaid", "노인주의(해열진통소염제)")):
        for row in _rows(files[kind], {"성분코드", "제품코드", "약품상세정보"}):
            product(row["제품코드"], row["제품명"], row["성분코드"], row["성분명"], row.get("업소명", row.get("업체명", "")), row.get("급여여부", ""))
            elderly_key = (row["성분코드"], category, row["약품상세정보"])
            elderly.setdefault(elderly_key, {"ingredient_code": row["성분코드"], "ingredient_name": row["성분명"], "category": category,
                                     "detail": row["약품상세정보"], "notice_no": row.get("공고번호", ""), "notice_date": row.get("공고일자", "")})

    def ordered(rows: dict[Any, dict[str, Any]]) -> list[dict[str, Any]]:
        return [rows[key] for key in sorted(rows, key=lambda item: tuple(map(str, item if isinstance(item, tuple) else (item,))))]

    tables = {"product": ordered(products), "pair_rule": ordered(pairs), "age_rule": ordered(ages),
              "pregnancy_rule": ordered(pregnancy), "elderly_rule": ordered(elderly)}
    inputs = {kind: {"file": unicodedata.normalize("NFC", path.name), "sha256": _sha256_file(path)} for kind, path in files.items()}
    return tables, inputs


def _table_bytes(rows: list[dict[str, Any]], columns: tuple[str, ...]) -> bytes:
    """행들을 지정 열 순서의 압축 JSONL UTF-8 바이트로 만든다."""
    return "".join(
        json.dumps({column: row[column] for column in columns}, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows
    ).encode("utf-8")


def _content_sha(source: Mapping[str, Any], tables: Mapping[str, Mapping[str, Any]]) -> str:
    """데이터셋 동일성 해시: 출처와 표 내용(행 수·SHA-256)이 같으면 같은 해시(빌드 시각은 제외)."""
    identity = {"source": source, "tables": {name: [info["count"], info["sha256"]] for name, info in tables.items()}}
    return hashlib.sha256(json.dumps(identity, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def build(directory: Path, output: Path) -> dict[str, int]:
    """``output/<table>.jsonl``과 ``manifest.json``을 임시 폴더에 쓴 뒤 ``output``으로 교체한다.

    역할: ``convert`` 결과를 표마다 파일로 쓰고 행 수·SHA-256·입력 파일·내용 해시를 manifest에 기록한다. 기존
    ``output``은 지운 뒤 임시 폴더로 바꾼다(삭제와 교체 사이에는 ``output``이 잠시 없다).
    입력:
        directory: DUR CSV 폴더.
        output: 스냅샷 출력 폴더.
    출력: {표 이름: 행 수}.
    예외: ``convert``의 ``DurDataError``·``UnicodeDecodeError``를 그대로 전파한다.
    """
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
        "format": FORMAT, "built_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": SOURCE, "inputs": inputs, "tables": table_info, "content_sha256": _content_sha(SOURCE, table_info),
    }
    (temporary / MANIFEST).write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if output.exists():
        shutil.rmtree(output)
    os.replace(temporary, output)
    return {name: info["count"] for name, info in table_info.items()}


def read_snapshot(directory: Path) -> Snapshot:
    """스냅샷을 읽고 검증한다.

    역할: manifest 형식, 표마다 파일 SHA-256·행 수·열 집합(첫 행 기준), 그리고 manifest의 ``content_sha256``을 확인한다.
    입력:
        directory: 스냅샷 폴더.
    출력: ``Snapshot``.
    예외: manifest가 없거나 어느 검증이든 실패하면 ``DurDataError``. 표 파일이 없으면 ``OSError``.
    """
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
        rows[name] = parse_jsonl(data.decode("utf-8"))
        if len(rows[name]) != info["count"]:
            raise DurDataError(f"{info['file']}: {len(rows[name])}행, manifest {info['count']}행")
        if rows[name] and set(rows[name][0]) != set(columns):
            raise DurDataError(f"{info['file']}: 열이 {list(columns)}와 다름")
    if _content_sha(manifest["source"], manifest["tables"]) != manifest.get("content_sha256"):
        raise DurDataError("manifest content_sha256 불일치")
    return Snapshot(manifest, rows)


def main(argv: list[str] | None = None) -> int:
    """``ax-g-ai-build-dur`` CLI 진입점.

    역할: 출력 폴더가 이미 있으면 ``--replace`` 없이는 멈추고, 아니면 스냅샷을 만들어 표별 행 수를 출력한다.
    입력:
        argv: 명령행 인자(``None``이면 ``sys.argv``).
    출력: 종료 코드. 성공 0, 출력 폴더 존재·변환 실패 1.
    """
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
