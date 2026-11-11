"""Europe PMC에서 오픈액세스 논문을 검색·저장하고 출처 레지스트리에 등록한다.

AI 활용 A등급은 라이선스가 CC BY 또는 CC0 인 논문만 해당한다(출처 표시 시 상업 이용·변형 허용).
CC BY-NC / -ND / -SA 등은 등록을 거부한다(허가 필요 등급).

예:
    # 후보 검색(지침·합의문·체계적 고찰 우선, CC BY/CC0만)
    python text_data/tools/fetch_europepmc.py search 'cirrhosis AND (TITLE:guideline OR TITLE:consensus)' --from 2021
    # 본문 저장 + 레지스트리 등록
    python text_data/tools/fetch_europepmc.py fetch PMC1234567 --register PAPER-XXXX
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
OUT_DIR = ROOT / "sources" / "raw" / "europepmc"
API = "https://www.ebi.ac.uk/europepmc/webservices/rest"
A_LICENSES = {"cc by", "cc0"}
MARKER = "  # ───────────── B: 비상업 조건"


def _get(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "ax-g-ai-corpus-builder"})
    with urllib.request.urlopen(request, timeout=90) as response:
        return response.read()


def _meta(pmcid: str) -> dict:
    query = urllib.parse.urlencode({"query": f"PMCID:{pmcid}", "format": "json", "resultType": "core"})
    results = json.loads(_get(f"{API}/search?{query}"))["resultList"]["result"]
    if not results:
        raise SystemExit(f"메타데이터 없음: {pmcid}")
    return results[0]


def search(query: str, year_from: int, size: int) -> None:
    full = f'({query}) AND OPEN_ACCESS:y AND (LICENSE:"cc by" OR LICENSE:"cc0") AND PUB_YEAR:[{year_from} TO 2026]'
    params = urllib.parse.urlencode({"query": full, "format": "json", "resultType": "core", "pageSize": size, "sort": "CITED desc"})
    data = json.loads(_get(f"{API}/search?{params}"))
    print(f"hits={data.get('hitCount')}")
    for item in data["resultList"]["result"]:
        types = ",".join(item.get("pubTypeList", {}).get("pubType", []))[:40]
        journal = item.get("journalInfo", {}).get("journal", {}).get("title", "")[:35]
        print(f"{item.get('pmcid')} | {item.get('license')} | {item.get('pubYear')} | cited {item.get('citedByCount')} | {journal} | {types} | {item.get('title', '')[:120]}")


def fetch(pmcid: str, register_id: str | None, region: str | None = None) -> None:
    meta = _meta(pmcid)
    license_name = (meta.get("license") or "").lower()
    if license_name not in A_LICENSES:
        raise SystemExit(f"라이선스 {license_name or '(없음)'} — A등급 아님, 저장·등록하지 않음")
    try:
        xml = _get(f"{API}/{pmcid}/fullTextXML")
    except urllib.error.HTTPError as error:
        raise SystemExit(f"본문 XML 없음({error.code}): {pmcid}") from None
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / f"{pmcid}.xml").write_bytes(xml)
    record = {
        "pmcid": pmcid, "pmid": meta.get("pmid"), "doi": meta.get("doi"), "title": meta.get("title"),
        "authors": meta.get("authorString"), "journal": meta.get("journalInfo", {}).get("journal", {}).get("title"),
        "year": meta.get("pubYear"), "license": license_name, "retrieved_at": date.today().isoformat(),
        "pub_types": meta.get("pubTypeList", {}).get("pubType", []),
    }
    (OUT_DIR / f"{pmcid}.json").write_text(json.dumps(record, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"저장 sources/raw/europepmc/{pmcid}.xml  license={license_name}  {record['journal']} {record['year']}")
    if register_id:
        _register(register_id, record, region)


def _register(source_id: str, record: dict, region: str) -> None:
    registry = ROOT / "sources" / "registry.yaml"
    text = registry.read_text(encoding="utf-8")
    if f"id: {source_id}\n" in text:
        print(f"이미 등록됨: {source_id}")
        return
    title = (record["title"] or "").replace('"', "'").rstrip(".")
    authors = (record["authors"] or "").split(",")[0].strip()
    citation = f"{authors} et al. {record['journal']} {record['year']}. doi:{record['doi']}".replace('"', "'")
    block = (
        f"  - id: {source_id}\n"
        f"    title: \"{title}\"\n"
        f"    type: paper\n"
        f"    citation: \"{citation}\"\n"
        f"    url: https://europepmc.org/article/PMC/{record['pmcid']}\n"
        f"    location: text_data/sources/raw/europepmc/{record['pmcid']}.xml\n"
        f"    published: \"{record['year']}\"\n"
        f"    access: primary_text\n"
        f"    checked_at: \"{record['retrieved_at']}\"\n"
        f"    ai_use: A\n"
        f"    region: {region}\n"
        f"    license_basis: \"Europe PMC 표기 라이선스 {record['license'].upper()} — 출처 표시 시 상업 이용·변형 허용. 본문 내 제3자 그림·표는 별도 권리 확인.\"\n\n"
    )
    registry.write_text(text.replace(MARKER, block + MARKER, 1), encoding="utf-8")
    print(f"등록: {source_id}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("search")
    s.add_argument("query")
    s.add_argument("--from", dest="year_from", type=int, default=2020)
    s.add_argument("--size", type=int, default=12)
    f = sub.add_parser("fetch")
    f.add_argument("pmcid")
    f.add_argument("--register", metavar="SOURCE_ID")
    f.add_argument("--region", choices=("KR", "US", "EU", "INTL"), help="출처 지역(등록 시 필수) — 국내 > 미국 > 유럽 > 그 밖 우선순위에 쓰임")
    args = parser.parse_args()
    if args.command == "search":
        search(args.query, args.year_from, args.size)
    else:
        if args.register and not args.region:
            parser.error("--register 에는 --region(KR·US·EU·INTL)이 필요합니다")
        fetch(args.pmcid, args.register, args.region)
    return 0


if __name__ == "__main__":
    sys.exit(main())
