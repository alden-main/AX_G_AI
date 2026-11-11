"""mappings/*.yaml 의 표준 코드를 공개 API로 확인해 mappings/verified/*.json 에 저장한다.

- LOINC : NLM Clinical Tables(loinc_items) 에서 코드별 공식 이름(LONG_COMMON_NAME)을 받는다.
- RxNorm: NLM RxNav 에서 성분명 → RxCUI(TTY=IN/MIN)와 FDA 약리 분류(EPC)를 받는다.

build_corpus.py 는 네트워크 없이 이 캐시만 읽는다. 캐시에 없는 코드는 빌드 오류다.

실행(프로젝트 루트):
    uv run --with pyyaml python text_data/tools/verify_codes.py            # 캐시에 없는 코드만 조회
    uv run --with pyyaml python text_data/tools/verify_codes.py --refresh  # 전부 다시 조회
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
MAPPINGS = ROOT / "mappings"
VERIFIED = MAPPINGS / "verified"
LOINC_API = "https://clinicaltables.nlm.nih.gov/api/loinc_items/v3/search"
RXNAV = "https://rxnav.nlm.nih.gov/REST"
USER_AGENT = "ax-g-ai-corpus-builder/1.0"


def _get_json(url: str) -> object:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                return json.load(response)
        except OSError:
            if attempt == 2:
                raise
            time.sleep(2 * (attempt + 1))
    raise AssertionError("unreachable")


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def _save(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(sorted(data.items())), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def loinc_codes() -> set[str]:
    tests = yaml.safe_load((MAPPINGS / "tests.yaml").read_text(encoding="utf-8"))
    return {code for items in tests.values() for item in items for code in item.get("loinc") or ()}


def lookup_loinc(code: str) -> dict | None:
    query = urllib.parse.urlencode({"terms": code, "sf": "LOINC_NUM", "df": "LOINC_NUM,LONG_COMMON_NAME", "maxList": 5})
    data = _get_json(f"{LOINC_API}?{query}")
    for number, name in data[3]:
        if number == code:
            return {"name": name, "retrieved_at": _now(), "source": "NLM Clinical Tables loinc_items"}
    return None


def drug_names() -> dict[str, list[str]]:
    drugs = yaml.safe_load((MAPPINGS / "drugs.yaml").read_text(encoding="utf-8"))
    result = {}
    for item in drugs["ingredients"]:
        if "rxnorm" in item and item["rxnorm"] is None:
            continue  # 국내 전용 성분 — RxNorm 미등재로 명시
        names = item.get("rxnorm") or item["id"].replace("_", " ")
        result[item["id"]] = [names] if isinstance(names, str) else list(names)
    return result


def lookup_rxnorm(name: str) -> dict | None:
    query = urllib.parse.urlencode({"name": name, "search": 0})
    ids = (_get_json(f"{RXNAV}/rxcui.json?{query}").get("idGroup") or {}).get("rxnormId") or []
    for rxcui in ids:
        props = (_get_json(f"{RXNAV}/rxcui/{rxcui}/properties.json").get("properties") or {})
        if props.get("tty") in {"IN", "MIN", "PIN"}:
            classes = _get_json(f"{RXNAV}/rxclass/class/byRxcui.json?rxcui={rxcui}&relaSource=FDASPL&relas=has_epc")
            epc = sorted({
                info["rxclassMinConceptItem"]["className"]
                for info in ((classes.get("rxclassDrugInfoList") or {}).get("rxclassDrugInfo") or [])
            })
            return {"rxcui": rxcui, "name": props.get("name"), "tty": props.get("tty"), "epc": epc,
                    "retrieved_at": _now(), "source": "NLM RxNav"}
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--only", choices=["loinc", "rxnorm"])
    args = parser.parse_args()
    missing = []

    if args.only in (None, "loinc"):
        path = VERIFIED / "loinc.json"
        cache = {} if args.refresh else _load(path)
        for code in sorted(loinc_codes() - set(cache)):
            found = lookup_loinc(code)
            if found is None:
                missing.append(f"LOINC {code}")
            else:
                cache[code] = found
        _save(path, cache)
        print(f"LOINC: {len(cache)}건 확인")

    if args.only in (None, "rxnorm") and (MAPPINGS / "drugs.yaml").is_file():
        path = VERIFIED / "rxnorm.json"
        cache = {} if args.refresh else _load(path)
        todo = [(concept, name) for concept, names in drug_names().items() for name in names if name not in cache]
        # RxNav 허용 한도(초당 20건) 안에서 4개씩 병렬 조회
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {pool.submit(lookup_rxnorm, name): (concept, name) for concept, name in todo}
            for done, future in enumerate(as_completed(futures), start=1):
                concept, name = futures[future]
                try:
                    found = future.result()
                except OSError as error:
                    found = None
                    print(f"  조회 오류 {name!r}: {error}", file=sys.stderr)
                if found is None:
                    missing.append(f"RxNorm {concept}: {name!r}")
                else:
                    cache[name] = found
                if done % 25 == 0:
                    _save(path, cache)  # 중간 저장 — 중단돼도 다시 실행하면 이어서 조회
                    print(f"  RxNorm {done}/{len(todo)}", flush=True)
        _save(path, cache)
        print(f"RxNorm: {len(cache)}건 확인")

    for item in missing:
        print("확인 실패:", item, file=sys.stderr)
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
