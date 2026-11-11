"""claims/*.yaml → updated/*.md, corpus/*.jsonl 생성기.

단일 원천은 claims/*.yaml, sources/registry.yaml, mappings/{tests,drugs}.yaml 이다(매핑 단계: tools/mappings.py). 생성물(updated/, corpus/)은 직접 수정하지 않는다.

출처 등급 규칙(registry.yaml ai_use):
- claim.current  : AI가 사용할 수 있는 내용. A등급 출처만 인용 가능.
- claim.reference: B·C·U등급 근거(허가 필요). 사람 검토용이며 AI 말뭉치(ai_tier=A)에 들어가지 않는다.
- corpus           : current 근거로 만든 구조화 항목 → ai_tier=A
- reference_corpus : reference 근거만 있는 구조화 항목 → ai_tier=permission_required

실행(프로젝트 루트):
    uv run --with pyyaml python text_data/tools/build_corpus.py          # 검증 + 생성
    uv run --with pyyaml python text_data/tools/build_corpus.py --check  # 검증만
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mappings  # noqa: E402  (같은 폴더의 mappings.py)

ROOT = Path(__file__).resolve().parents[1]
CLAIMS_DIR = ROOT / "claims"
REGISTRY = ROOT / "sources" / "registry.yaml"
UPDATED_DIR = ROOT / "updated"
CORPUS_DIR = ROOT / "corpus"

STATUSES = {"유지", "개정", "보완", "폐기", "확인불가"}
VERIFICATIONS = {"primary", "secondary", "local_db", "unverified"}
REVIEW_STATUSES = {"draft", "approved", "rejected"}
AI_USE = {"A", "B", "C", "U"}
CLAIM_TYPES = {"diagnosis", "workup", "treatment", "dose", "threshold", "monitoring", "referral", "formula"}
KIND_TO_FILE = {
    "diagnosis": "diagnosis.jsonl",
    "workup": "workup.jsonl",
    "treatment": "treatment.jsonl",
    "dose_rule": "dose_rules.jsonl",
    "dose_principle": "dose_rules.jsonl",
    "lab_threshold": "lab_thresholds.jsonl",
    "formula": "formulas.jsonl",
    "monitoring": "monitoring.jsonl",
    "referral": "referral.jsonl",
}
VERIFICATION_KO = {"primary": "원문 확인", "secondary": "2차 자료", "local_db": "로컬 DB", "unverified": "미검증"}
AI_USE_KO = {"A": "A(사용 가능)", "B": "B(비상업 조건)", "C": "C(허가 필요)", "U": "U(조건 미확인)"}
CLAIM_REF = re.compile(r"\[((?:\d{3}e?|S\d{2})-C\d{2})\]")
# 보충 장: 19판 원문 범위 밖 주제(순환기·호흡기·감염·암 검진 등). 원문 대조 없이 A등급 최신 근거로만 쓴다.
SUPPLEMENT_SOURCE = "SUPPLEMENT"


def is_supplement(chapter: dict) -> bool:
    return chapter.get("original_source") == SUPPLEMENT_SOURCE
# 조건식 문법 — src/ax_g_ai/services/clinical_corpus.py 의 evaluate()와 일치해야 한다.
CONDITION_SHAPES = (
    {"all"}, {"any"}, {"not"}, {"always"}, {"var", "op", "value"},
    {"dx"}, {"med"}, {"med_class"}, {"finding"}, {"event"}, {"risk"}, {"indication"}, {"lab_present"}, {"dx_history"},
)
CONDITION_OPERATORS = {"<", "<=", ">", ">=", "=="}


def iter_conditions(node: object):
    """corpus 항목에서 조건식(when/logic/trigger, referral criteria)을 모두 꺼낸다."""
    if isinstance(node, dict):
        for key, value in node.items():
            if key in {"when", "logic", "trigger"} and isinstance(value, dict):
                yield value
            elif key == "criteria" and isinstance(value, list):
                yield from (item for item in value if isinstance(item, dict))
            else:
                yield from iter_conditions(value)
    elif isinstance(node, list):
        for item in node:
            yield from iter_conditions(item)


def condition_errors(condition: dict) -> list[str]:
    keys = set(condition)
    if keys not in CONDITION_SHAPES:
        return [f"지원하지 않는 조건식 키 {sorted(keys)}"]
    if "var" in keys and condition["op"] not in CONDITION_OPERATORS:
        return [f"지원하지 않는 연산자 {condition['op']!r}"]
    children = condition.get("all") or condition.get("any") or ([condition["not"]] if "not" in condition else [])
    return [error for child in children for error in condition_errors(child)]


PAGE_MARK = re.compile(r"PDF (\d+)페이지")
_PAGE_CACHE: dict[str, list[int | None]] = {}


def pdf_page(file: str, line: int) -> int | None:
    """원문 파일의 'PDF n페이지' 표시로 해당 행의 PDF 쪽을 계산한다(수기 입력보다 우선)."""
    if file not in _PAGE_CACHE:
        pages: list[int | None] = []
        current: int | None = None
        for text in (ROOT / file).read_text(encoding="utf-8").splitlines():
            match = PAGE_MARK.search(text)
            if match:
                current = int(match.group(1))
            pages.append(current)
        _PAGE_CACHE[file] = pages
    pages = _PAGE_CACHE[file]
    return pages[line - 1] if 0 < line <= len(pages) else None


def load_yaml(path: Path) -> object:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_inputs() -> tuple[dict[str, dict], list[dict]]:
    registry = load_yaml(REGISTRY)
    sources = {entry["id"]: entry for entry in registry["sources"]}
    chapters = [load_yaml(path) for path in sorted(CLAIMS_DIR.glob("*.yaml"))]
    return sources, chapters


def current_of(claim: dict) -> dict | None:
    current = claim.get("current")
    return current if current and current.get("text") else None


def ai_tier(claim: dict) -> str:
    return "A" if current_of(claim) else "permission_required"


def _validate_entries(where: str, entries: list, sources: dict, a_only: bool, dose_ids: set, dose_refs: list) -> list[str]:
    errors = []
    for entry in entries:
        kind = entry.get("kind")
        if kind not in KIND_TO_FILE:
            errors.append(f"{where}: 알 수 없는 corpus kind {kind!r}")
        for condition in iter_conditions(entry):
            errors.extend(f"{where}: {error}" for error in condition_errors(condition))
        if kind == "dose_rule":
            if not entry.get("id", "").startswith("DR-"):
                errors.append(f"{where}: dose_rule id는 DR-로 시작해야 함")
            if entry.get("id") in dose_ids:
                errors.append(f"{where}: 중복 dose_rule id {entry.get('id')}")
            dose_ids.add(entry.get("id", ""))
            for sid in entry.get("source_priority", []):
                if sid not in sources:
                    errors.append(f"{where}: dose_rule 출처 {sid} 미등록")
                elif a_only and sources[sid].get("ai_use") != "A":
                    errors.append(f"{where}: AI용 dose_rule {entry.get('id')}에 A등급이 아닌 출처 {sid}")
        if "dose_rule_ref" in entry:
            dose_refs.append((where, entry["dose_rule_ref"]))
    return errors


def validate(sources: dict[str, dict], chapters: list[dict]) -> list[str]:
    errors: list[str] = []
    for sid, entry in sources.items():
        if entry.get("ai_use") not in AI_USE:
            errors.append(f"출처 {sid}: ai_use 누락/오류")
        elif entry.get("ai_use") == "A" and license_label(entry) is None:
            errors.append(f"출처 {sid}: A등급인데 license_basis에서 라이선스(CC0·CC BY·퍼블릭 도메인)를 확인할 수 없음 — 출처 표시 불가")
        if entry.get("region") not in REGIONS:
            errors.append(f"출처 {sid}: region 누락/오류(KR·US·EU·INTL)")
    seen: set[str] = set()
    dose_ids: set[str] = set()
    dose_refs: list[tuple[str, str]] = []
    for chapter in chapters:
        prefix = f"{chapter['chapter']}-C"
        for claim in chapter["claims"]:
            cid = claim.get("id", "?")
            where = cid
            if not cid.startswith(prefix):
                errors.append(f"{where}: id가 장 번호({chapter['chapter']})와 맞지 않음")
            if cid in seen:
                errors.append(f"{where}: 중복 id")
            seen.add(cid)
            if claim.get("type") not in CLAIM_TYPES:
                errors.append(f"{where}: 알 수 없는 type {claim.get('type')!r}")
            if claim.get("status") not in STATUSES:
                errors.append(f"{where}: 알 수 없는 status {claim.get('status')!r}")
            if claim.get("verification") not in VERIFICATIONS:
                errors.append(f"{where}: 알 수 없는 verification {claim.get('verification')!r}")
            if claim.get("review", {}).get("status") not in REVIEW_STATUSES:
                errors.append(f"{where}: review.status 누락/오류")
            location = (claim.get("original") or {}).get("location", {})
            source_file = ROOT / location.get("file", "")
            if is_supplement(chapter):
                # 원문이 없으므로 '보완'만 허용하고, A등급 최신 근거(current)가 반드시 있어야 한다.
                if claim.get("original"):
                    errors.append(f"{where}: 보충 장 주장에는 original을 두지 않음")
                if claim.get("status") != "보완":
                    errors.append(f"{where}: 보충 장 주장의 status는 보완만 허용")
                if current_of(claim) is None:
                    errors.append(f"{where}: 보충 장 주장은 A등급 current가 필요")
            elif not source_file.is_file():
                errors.append(f"{where}: 원문 파일 없음 {location.get('file')}")
            else:
                line_count = source_file.read_text(encoding="utf-8").count("\n") + 1
                if not 1 <= int(location.get("line", 0)) <= line_count:
                    errors.append(f"{where}: 원문 줄 번호 범위 밖 {location.get('line')}")
            current = current_of(claim)
            reference = claim.get("reference") or {}
            if current is None and not reference.get("text") and claim.get("status") != "확인불가":
                errors.append(f"{where}: current와 reference가 모두 비어 있음(근거가 없으면 status=확인불가)")
            for ref in (current or {}).get("sources", []):
                entry = sources.get(ref.get("id"))
                if entry is None:
                    errors.append(f"{where}: 레지스트리에 없는 출처 {ref.get('id')}")
                elif entry.get("ai_use") != "A":
                    errors.append(f"{where}: current에 A등급이 아닌 출처 {ref['id']}({entry.get('ai_use')}) — reference로 옮길 것")
            for ref in reference.get("sources", []):
                if ref.get("id") not in sources:
                    errors.append(f"{where}: 레지스트리에 없는 출처 {ref.get('id')}")
            if current is not None and not current.get("sources"):
                errors.append(f"{where}: current에 출처가 없음")
            if current is None and claim.get("verification") != "unverified":
                errors.append(f"{where}: current가 없으면 verification은 unverified")
            if claim.get("review", {}).get("status") == "approved" and (current is None or claim.get("verification") == "unverified"):
                errors.append(f"{where}: A등급 근거가 확인되지 않은 주장은 approved 불가")
            if current is None and claim.get("corpus"):
                errors.append(f"{where}: current 없이 corpus를 둘 수 없음 — reference_corpus로 옮길 것")
            errors += _validate_entries(where, claim.get("corpus", []), sources, True, dose_ids, dose_refs)
            errors += _validate_entries(where, claim.get("reference_corpus", []), sources, False, dose_ids, dose_refs)
    for where, ref in dose_refs:
        if ref not in dose_ids:
            errors.append(f"{where}: dose_rule_ref {ref} 정의 없음")
    errors += _validate_preferred(sources, chapters)
    return errors


def claim_region_rank(claim: dict, sources: dict[str, dict]) -> int:
    """Best (lowest) region rank among the claim's current sources."""
    ranks = [REGIONS.index(sources[ref["id"]].get("region", "INTL")) for ref in (current_of(claim) or {}).get("sources", [])
             if ref.get("id") in sources and sources[ref["id"]].get("region") in REGIONS]
    return min(ranks, default=len(REGIONS))


def _validate_preferred(sources: dict[str, dict], chapters: list[dict]) -> list[str]:
    """preferred_over(주장 id 또는 corpus 항목 id)는 실제로 있어야 하고, 지역 우선순위가 더 높은 주장만 쓸 수 있다."""
    claims = {claim["id"]: claim for chapter in chapters for claim in chapter["claims"]}
    records = {name: claim for name, claim in _record_ids(chapters)}
    errors = []
    for cid, claim in claims.items():
        targets = [(target, claims.get(target)) for target in claim.get("preferred_over", [])]
        targets += [(target, records.get(target)) for entry in claim.get("corpus", []) for target in entry.get("preferred_over", [])]
        for target, other in targets:
            if other is None:
                errors.append(f"{cid}: preferred_over 대상 {target} 없음")
            elif other["id"] == cid:
                errors.append(f"{cid}: preferred_over가 자기 자신을 가리킴")
            elif claim_region_rank(claim, sources) >= claim_region_rank(other, sources):
                errors.append(f"{cid}: preferred_over {target} — 출처 지역 우선순위가 더 높지 않음(국내 > 미국 > 유럽 > 그 밖)")
    return errors


def _record_ids(chapters: list[dict]):
    for chapter in chapters:
        for claim in chapter["claims"]:
            for tag, entries in (("K", claim.get("corpus", [])), ("R", claim.get("reference_corpus", []))):
                for index, entry in enumerate(entries, start=1):
                    yield entry.get("id") or f"{claim['id']}-{tag}{index}", claim


def is_usable(claim: dict) -> bool:
    """운영 AI가 써도 되는지: A등급 근거 + 의료진 승인 + 근거 확인 + 유효한 상태."""
    return (
        ai_tier(claim) == "A"
        and claim["review"]["status"] == "approved"
        and claim["verification"] in {"primary", "local_db"}
        and claim["status"] not in {"폐기", "확인불가"}
    )


# 출처 표시에 쓸 라이선스 이름. license_basis(사람이 쓴 판단 근거)에서 뽑는다 — 첫 번째로 맞는 규칙.
# A등급 출처는 반드시 하나에 맞아야 한다(맞지 않으면 빌드 실패). 라이선스를 모르면 출처를 제대로 밝힐 수 없다.
LICENSE_RULES = (
    (re.compile(r"CC0\s*1\.0|\bCC0\b", re.I), "CC0 1.0"),
    (re.compile(r"CC[ -]BY[ -]?(\d\.\d)", re.I), "CC BY {0}"),
    (re.compile(r"CC[ -]BY(?![ -]?N[CD])", re.I), "CC BY"),
    (re.compile(r"17 U\.S\.C|미국 (연방정부|국립보건원)|퍼블릭 도메인|public domain", re.I), "Public domain (U.S. Government work)"),
    (re.compile(r"copyright free", re.I), "Copyright-free (U.S. Government work)"),
    (re.compile(r"저작권법 제7조"), "Not protected (Korean Copyright Act Art. 7 — statutes·public notices)"),
    (re.compile(r"공공누리 제1유형"), "KOGL Type 1 (공공누리 제1유형 — 출처표시)"),
)
# 출처 지역 우선순위: 국내 > 미국 > 유럽 > 그 밖. 같은 주제는 preferred_over로 국내 기준이 국외 기준을 앞선다.
REGIONS = ("KR", "US", "EU", "INTL")


def license_label(entry: dict) -> str | None:
    basis = str(entry.get("license_basis") or "")
    for pattern, label in LICENSE_RULES:
        match = pattern.search(basis)
        if match:
            return label.format(*match.groups())
    return None


def _source_refs(refs: list[dict], sources: dict[str, dict]) -> list[dict]:
    result = []
    for ref in refs:
        entry = sources[ref["id"]]
        item = {**ref, "title": entry["title"], "ai_use": entry["ai_use"], "url": entry.get("url"), "access": entry.get("access"),
                "region": entry.get("region")}
        license_name = license_label(entry)
        if license_name:
            item["license"] = license_name
        if entry.get("citation"):
            item["citation"] = entry["citation"]
        result.append(item)
    return result


def corpus_records(sources: dict[str, dict], chapters: list[dict]) -> dict[str, list[dict]]:
    buckets: dict[str, list[dict]] = {name: [] for name in set(KIND_TO_FILE.values())}
    for chapter in chapters:
        for claim in chapter["claims"]:
            current = current_of(claim)
            groups = (
                ("A", claim.get("corpus", []), (current or {}).get("sources", []), "K"),
                ("permission_required", claim.get("reference_corpus", []), (claim.get("reference") or {}).get("sources", []), "R"),
            )
            for tier, entries, refs, tag in groups:
                for index, entry in enumerate(entries, start=1):
                    record = {
                        "id": entry.get("id") or f"{claim['id']}-{tag}{index}",
                        "claim_id": claim["id"],
                        "chapter": chapter["chapter"],
                        "topic": claim["topic"],
                        "ai_tier": tier,
                        "claim_status": claim["status"],
                        "verification": claim["verification"] if tier == "A" else "reference_only",
                        "review_status": claim["review"]["status"],
                        "usable": tier == "A" and is_usable(claim),
                        "sources": _source_refs(refs, sources),
                        "conflicts": claim.get("conflicts", []),
                        **{key: value for key, value in entry.items() if key not in {"id", "preferred_over"}},
                    }
                    preferred = [*claim.get("preferred_over", []), *entry.get("preferred_over", [])]
                    if preferred:
                        record["preferred_over"] = preferred
                    buckets[KIND_TO_FILE[entry["kind"]]].append(record)
    return buckets


def write_corpus(buckets: dict[str, list[dict]]) -> Counter:
    CORPUS_DIR.mkdir(exist_ok=True)
    counts: Counter = Counter()
    for name, records in sorted(buckets.items()):
        with (CORPUS_DIR / name).open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        counts[name] = len(records)
    return counts


def _source_line(ref: dict, sources: dict[str, dict]) -> str:
    entry = sources[ref["id"]]
    url = entry.get("url")
    title = f"[{entry['title']}]({url})" if url else entry["title"]
    citation = f" {entry['citation']}." if entry.get("citation") else ""
    locator = f" — {ref['locator']}" if ref.get("locator") else ""
    return (f"- `{ref['id']}` {title}.{citation}{locator} "
            f"(AI 활용 {AI_USE_KO[entry['ai_use']]}, 확인 방식 {entry.get('access')}, 확인일 {entry.get('checked_at')})")


def render_chapter(chapter: dict, sources: dict[str, dict]) -> str:
    claims = chapter["claims"]
    status_counts = Counter(claim["status"] for claim in claims)
    tier_counts = Counter(ai_tier(claim) for claim in claims)
    lines = [
        "---",
        f"id: harrison-{chapter['chapter']}-updated",
        f"chapter: '{chapter['chapter']}'",
        f"title: {json.dumps(chapter['title'], ensure_ascii=False)}",
        f"original_source: {chapter['original_source']}",
        f"updated_at: '{chapter['updated_at']}'",
        f"generated_from: text_data/claims/{chapter['chapter']}.yaml",
        "review_status: draft",
        "---",
        "",
        f"# {chapter['title']} — 최신화 정리",
        "",
        "> 이 문서는 `tools/build_corpus.py`가 주장 파일에서 생성한다. 직접 수정하지 말고 주장 파일을 고친다.",
        "> 모든 항목은 의료진 검수 전(draft)이며 직접적인 임상 의사결정이나 자동 처방에 사용하지 않는다.",
        "> 'AI 사용 가능 내용'은 A등급(상업 AI 사용 가능) 출처만 인용한다. '참고(허가 필요)'는 AI 말뭉치에 들어가지 않는다.",
        "",
        "## 요약",
        "",
        "| 상태 | 건수 |",
        "| --- | ---: |",
        *[f"| {name} | {status_counts.get(name, 0)} |" for name in ("유지", "개정", "보완", "폐기", "확인불가")],
        "",
        "| AI 활용 | 건수 |",
        "| --- | ---: |",
        f"| A등급 근거 있음 | {tier_counts.get('A', 0)} |",
        f"| 허가 필요 근거만 있음 | {tier_counts.get('permission_required', 0)} |",
        "",
        "| ID | 주제 | 상태 | AI 활용 | 검증 |",
        "| --- | --- | --- | --- | --- |",
        *[f"| {c['id']} | {c['topic']} | {c['status']} | {'A' if ai_tier(c) == 'A' else '허가 필요'} | {VERIFICATION_KO[c['verification']]} |" for c in claims],
        "",
    ]
    if is_supplement(chapter):
        lines += ["> 보충 장: 19판 원문 범위 밖 주제다. 원문 요지 없이 A등급 최신 근거만으로 작성했다(모든 주장 상태 '보완').", ""]
    for claim in claims:
        current = current_of(claim)
        reference = claim.get("reference") or {}
        lines += [
            f"## {claim['topic']} [{claim['id']}]",
            "",
            f"- 상태: **{claim['status']}** · AI 활용: {'A등급 근거' if current else '허가 필요 근거만'} · 검증: {VERIFICATION_KO[claim['verification']]} · 검수: {claim['review']['status']}",
            "",
        ]
        if claim.get("original"):
            location = claim["original"]["location"]
            lines += [
                f"**19판 원문 요지** ([{location['file']}](../{location['file']}) {location['line']}행, PDF {pdf_page(location['file'], int(location['line'])) or '?'}쪽)",
                "",
                claim["original"]["text"].strip(),
                "",
            ]
        lines += [
            "**AI 사용 가능 내용 (A등급 출처)**",
            "",
            current["text"].strip() if current else "_A등급 출처로 확인된 내용 없음._",
            "",
        ]
        if current:
            lines += [_source_line(ref, sources) for ref in current["sources"]] + [""]
        if reference.get("text"):
            lines += ["**참고 — 허가 필요 (AI 사용 불가)**", "", reference["text"].strip(), ""]
            lines += [_source_line(ref, sources) for ref in reference.get("sources", [])] + [""]
        if claim.get("conflicts"):
            lines += ["**출처 간 차이·주의**", "", *[f"- {item}" for item in claim["conflicts"]], ""]
        if claim.get("note"):
            lines += [f"> 참고: {claim['note']}", ""]
    return "\n".join(lines).rstrip() + "\n"


def build_updated(sources: dict[str, dict], chapters: list[dict]) -> list[Path]:
    UPDATED_DIR.mkdir(exist_ok=True)
    written = []
    for chapter in chapters:
        slug = re.sub(r"[^0-9A-Za-z가-힣]+", "", chapter["title"])
        path = UPDATED_DIR / f"{chapter['chapter']}_{slug}.md"
        path.write_text(render_chapter(chapter, sources), encoding="utf-8")
        written.append(path)
    return written


def build_chunks(chapters: list[dict], updated_files: list[Path], sources: dict[str, dict]) -> int:
    """AI 검색용 청크: A등급(current) 내용만 담는다."""
    count = 0
    by_chapter = {chapter["chapter"]: chapter for chapter in chapters}
    doc_paths = {path.name.split("_", 1)[0]: path for path in updated_files}
    with (CORPUS_DIR / "chunks.jsonl").open("w", encoding="utf-8") as handle:
        for number, chapter in by_chapter.items():
            for claim in chapter["claims"]:
                current = current_of(claim)
                if current is None:
                    continue
                record = {
                    "chunk_id": f"{claim['id']}-chunk",
                    "claim_id": claim["id"],
                    "chapter": number,
                    "title_path": [chapter["title"], claim["topic"]],
                    "claim_type": claim["type"],
                    "claim_status": claim["status"],
                    "ai_tier": "A",
                    "verification": claim["verification"],
                    "review_status": claim["review"]["status"],
                    "usable": is_usable(claim),
                    "source_ids": [ref["id"] for ref in current["sources"]],
                    "sources": _source_refs(current["sources"], sources),
                    "conflicts": claim.get("conflicts", []),
                    **({"preferred_over": claim["preferred_over"]} if claim.get("preferred_over") else {}),
                    "updated_at": chapter.get("updated_at"),
                    "text": f"{claim['topic']}\n{current['text'].strip()}",
                    "doc_path": str(doc_paths[number].relative_to(ROOT)),
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="검증만 수행")
    args = parser.parse_args()
    sources, chapters = load_inputs()
    errors = validate(sources, chapters)
    if errors:
        print("검증 실패:", *errors, sep="\n  ", file=sys.stderr)
        return 1
    claims = [claim for chapter in chapters for claim in chapter["claims"]]
    tiers = Counter(ai_tier(claim) for claim in claims)
    print(f"검증 통과: 장 {len(chapters)}개, 주장 {len(claims)}건(A등급 근거 {tiers.get('A', 0)}, 허가 필요만 {tiers.get('permission_required', 0)}), 출처 {len(sources)}건")
    buckets = corpus_records(sources, chapters)
    concept_map = mappings.load_mappings()
    map_errors, concept_files = mappings.reorganize(concept_map, buckets)
    if map_errors:
        print("검사·약제 매핑 실패:", *map_errors, sep="\n  ", file=sys.stderr)
        return 1
    print(f"매핑 통과: 검사·변수 개념 {len(concept_map.tests)}개(LOINC {len(concept_map.loinc)}건), "
          f"성분 {len(concept_map.ingredients)}개·복합제 {len(concept_map.products)}개·약효군 {len(concept_map.classes)}개(RxNorm {len(concept_map.rxnorm)}건)")
    if args.check:
        return 0
    updated = build_updated(sources, chapters)
    counts = write_corpus({**buckets, **concept_files})
    counts["chunks.jsonl"] = build_chunks(chapters, updated, sources)
    for path in updated:
        print(f"  생성 {path.relative_to(ROOT)}")
    for name, count in sorted(counts.items()):
        print(f"  생성 corpus/{name}: {count}건")
    _report_embedding_coverage()
    return 0


def _report_embedding_coverage() -> None:
    """사전 계산 임베딩이 새 말뭉치와 맞지 않으면 알린다(이미지 빌드는 이 상태에서 실패한다)."""
    sys.path.insert(0, str(ROOT.parent / "src"))
    try:
        from ax_g_ai.corpus_store.precomputed import coverage
    except ImportError:
        return
    total, missing, error = coverage(CORPUS_DIR)
    if error or missing:
        print(f"  주의: 사전 계산 임베딩 누락 {len(missing)}/{total}건"
              f"{' (' + error + ')' if error else ''} — tools/build_embeddings.py 를 실행하세요.", file=sys.stderr)
    else:
        print(f"  사전 계산 임베딩: {total}건 모두 최신")


if __name__ == "__main__":
    raise SystemExit(main())
