"""병원 검사·약품 마스터(CSV)에서 사이트 매핑 초안(JSON)을 만든다 — 검수 전용 제안.

입력 CSV(UTF-8, 머리글 필수)
  검사: code,name[,unit]      예) LAB-0002,Creatinine,mg/dL
  약품: code,name             예) 052400180,리피틴정10mg(아토르바스타틴칼슘수화물)

출력은 status=draft 이며 각 항목에 제안 근거(suggested_by)가 붙는다. 검사 단위가 비어 있으면
unit=null 로 남긴다 — 단위가 확인되기 전에는 런타임이 그 값을 규칙에 쓰지 않는다.

실행(프로젝트 루트, 말뭉치 생성 후):
    uv run python text_data/tools/propose_site_map.py --labs labs.csv --meds meds.csv --site 병원명 -o site.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent / "src"))

from ax_g_ai.services.emr_mapping import ConceptCatalog  # noqa: E402


def _rows(path: Path | None) -> list[dict[str, str]]:
    if path is None:
        return []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return [{key.strip(): (value or "").strip() for key, value in row.items() if key} for row in csv.DictReader(handle)]


def propose(catalog: ConceptCatalog, labs: list[dict[str, str]], meds: list[dict[str, str]], site: str) -> dict:
    lab_entries, med_entries, unmatched = [], [], {"labs": [], "meds": []}
    for row in labs:
        code, name = row.get("code"), row.get("name")
        concept = catalog.test_by_loinc.get(code or "") or catalog.suggest_test(name, code)
        if concept is None or catalog.tests[concept].get("section") != "labs":
            unmatched["labs"].append({"code": code, "name": name})
            continue
        unit = row.get("unit") or None
        expected = catalog.tests[concept].get("unit")
        lab_entries.append({
            "code": code, "concept": concept, "unit": unit, "suggested_by": "loinc" if code in catalog.test_by_loinc else "name",
            "source_name": name, "concept_unit": expected,
        })
    for row in meds:
        code, name = row.get("code"), row.get("name") or ""
        ingredients, complete = catalog.match_ingredients(name)
        if not ingredients:
            unmatched["meds"].append({"code": code, "name": name})
            continue
        med_entries.append({"code": code, "ingredients": list(ingredients), "suggested_by": "name",
                            "source_name": name, "all_components_matched": complete})
    return {"site": site, "version": date.today().isoformat(), "status": "draft",
            "note": "자동 제안 — 의료진·검사실 검수 후 status 를 reviewed 로 바꾸고 suggested_by 등 보조 필드를 확인한다.",
            "labs": lab_entries, "meds": med_entries, "unmatched": unmatched}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labs", type=Path)
    parser.add_argument("--meds", type=Path)
    parser.add_argument("--site", required=True)
    parser.add_argument("--corpus", type=Path, default=ROOT / "corpus")
    parser.add_argument("-o", "--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        print(f"{args.output} 가 이미 있습니다 — 덮어쓰지 않습니다.", file=sys.stderr)
        return 1
    catalog = ConceptCatalog.load(args.corpus)
    result = propose(catalog, _rows(args.labs), _rows(args.meds), args.site)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"검사 {len(result['labs'])}건·약품 {len(result['meds'])}건 제안, 미매칭 검사 {len(result['unmatched']['labs'])}건·약품 {len(result['unmatched']['meds'])}건")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
