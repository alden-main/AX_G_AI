"""openFDA 의약품 허가사항(SPL) 원문을 내려받아 버전 정보와 함께 저장한다.

openFDA 데이터는 CC0(퍼블릭 도메인 헌정)으로 제공된다: https://open.fda.gov/license
저장 위치: text_data/sources/raw/openfda/<key>.json (필요한 섹션만 보관)

예:
    python text_data/tools/fetch_openfda_label.py metformin-glucophage \
        'openfda.brand_name:"GLUCOPHAGE"'
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "sources" / "raw" / "openfda"
API = "https://api.fda.gov/drug/label.json"
SECTIONS = (
    "indications_and_usage", "dosage_and_administration", "dosage_forms_and_strengths", "contraindications",
    "warnings_and_cautions", "boxed_warning", "use_in_specific_populations", "pregnancy", "pediatric_use",
    "geriatric_use", "renal_impairment", "hepatic_impairment", "drug_interactions", "adverse_reactions",
    # 구형식(PLR 이전) 라벨은 경고·주의·검사 항목이 아래 섹션에 있다.
    "warnings", "precautions", "general_precautions", "laboratory_tests",
)


def fetch(search: str) -> dict:
    url = f"{API}?{urllib.parse.urlencode({'search': search, 'limit': 1, 'sort': 'effective_time:desc'})}"
    request = urllib.request.Request(url, headers={"User-Agent": "ax-g-ai-corpus-builder"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as error:
        if error.code == 404:
            raise SystemExit(f"결과 없음: {search}") from None
        raise
    results = payload.get("results") or []
    if not results:
        raise SystemExit(f"결과 없음: {search}")
    return results[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("key", help="저장 파일 이름(예: metformin-glucophage)")
    parser.add_argument("search", help="openFDA search 식")
    parser.add_argument("--expect-generic", required=True, help="openfda.generic_name에 반드시 포함돼야 하는 성분명(대소문자 무시)")
    parser.add_argument("--register", metavar="SOURCE_ID", help="sources/registry.yaml에 A등급 출처로 등록(이미 있으면 건너뜀)")
    parser.add_argument("--name", help="레지스트리 제목에 쓸 제품 설명(예: 'Pregabalin capsules')")
    args = parser.parse_args()
    label = fetch(args.search)
    openfda = label.get("openfda", {})
    generic = " ".join(openfda.get("generic_name", [])).upper()
    if not generic:
        # 일부 SPL은 openfda 블록이 비어 있다. 제품 데이터(첫 제품의 성분 목록)로 확인한다.
        generic = " ".join(label.get("spl_product_data_elements", [])[:1]).upper()
    if args.expect_generic.upper() not in generic:
        raise SystemExit(f"성분 불일치: 기대 {args.expect_generic!r}, 실제 {generic or '(openfda 정보 없음)'} — 저장하지 않음")
    record = {
        "key": args.key,
        "retrieved_at": date.today().isoformat(),
        "license": "CC0 1.0 (openFDA, https://open.fda.gov/license)",
        "query": args.search,
        "set_id": label.get("set_id"),
        "id": label.get("id"),
        "version": label.get("version"),
        "effective_time": label.get("effective_time"),
        "brand_name": openfda.get("brand_name") or label.get("spl_product_data_elements", [""])[0].split(" ")[:1],
        "generic_name": openfda.get("generic_name"),
        "manufacturer_name": openfda.get("manufacturer_name"),
        "dailymed_url": f"https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid={label.get('set_id')}",
        "sections": {name: label[name] for name in SECTIONS if name in label},
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{args.key}.json"
    path.write_text(json.dumps(record, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"{path.relative_to(ROOT)}  set_id={record['set_id']} version={record['version']} effective={record['effective_time']}")
    if args.register:
        register(args.register, args.name or args.key, record, path)
    return 0


def register(source_id: str, name: str, record: dict, path: Path) -> None:
    registry = ROOT / "sources" / "registry.yaml"
    text = registry.read_text(encoding="utf-8")
    if f"id: {source_id}\n" in text:
        print(f"이미 등록됨: {source_id}")
        return
    effective = record["effective_time"] or ""
    effective = f"{effective[:4]}-{effective[4:6]}-{effective[6:]}" if len(effective) == 8 else effective
    block = (
        f"  - id: {source_id}\n"
        f"    title: \"openFDA SPL — {name} (set_id {record['set_id']}, v{record['version']}, {effective})\"\n"
        f"    type: label\n"
        f"    url: {record['dailymed_url']}\n"
        f"    location: text_data/{path.relative_to(ROOT)}\n"
        f"    access: primary_text\n"
        f"    checked_at: \"{record['retrieved_at']}\"\n"
        f"    ai_use: A\n"
        f"    region: US\n"
        f"    license_basis: openFDA CC0 1.0. 미국 허가사항 — 국내 허가와 다를 수 있음.\n\n"
    )
    marker = "  # ───────────── B: 비상업 조건"
    if marker not in text:
        raise SystemExit("레지스트리 구분선(B등급)을 찾지 못함 — 수동 등록 필요")
    registry.write_text(text.replace(marker, block + marker, 1), encoding="utf-8")
    print(f"등록: {source_id}")


if __name__ == "__main__":
    sys.exit(main())
