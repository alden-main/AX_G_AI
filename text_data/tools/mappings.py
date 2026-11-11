"""mappings/tests.yaml·drugs.yaml 을 검증하고 말뭉치를 검사·약제 개념 기준으로 재정리한다.

build_corpus.py 가 호출한다. 네트워크를 쓰지 않으며 표준 코드는 mappings/verified/*.json 캐시로만 확인한다.

생성물(corpus/):
  test_concepts.jsonl  검사·변수 개념 1줄 = LOINC·단위·환산·EMR 별칭 + 이 개념을 쓰는 말뭉치 항목(used_by)
  drug_concepts.jsonl  성분·복합제 1줄 = RxCUI·FDA EPC·국내 성분명·소속 약효군 + used_by(약효군 경유 포함)
  drug_classes.jsonl   약효군 1줄 = 펼친 성분 목록·판정 가능 여부 + used_by
그리고 각 말뭉치 항목에 "concepts": {"tests": [...], "drugs": [...], "drug_classes": [...]} 를 붙인다.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
MAPPINGS_DIR = ROOT / "mappings"
VERIFIED_DIR = MAPPINGS_DIR / "verified"

TEST_SECTIONS = ("labs", "panels", "exams", "measurements", "vitals", "scores", "derived", "med_derived", "clinical", "tasks")
# 값(var)으로 규칙에 들어갈 수 있는 구역과 그 값의 출처
VALUE_SOURCE = {
    "labs": "lab", "panels": "lab", "measurements": "exam_report", "vitals": "vital", "scores": "clinician",
    "derived": "derived", "med_derived": "medication_history", "clinical": "clinician", "exams": "exam_report", "tasks": "clinician",
}
# 반드시 개념으로 풀려야 하는 필드(풀리지 않으면 빌드 오류)
STRICT_TEST_FIELDS = {"lab": "lab", "exam": "exam", "var": "condition", "check": "monitoring", "renal_param": "dose_param"}
STRICT_DRUG_FIELDS = {
    "med": "condition", "med_class": "condition", "ingredient": "recommend", "ingredients": "recommend",
    "ingredients_any": "recommend", "drug": "recommend", "drugs": "recommend", "drug_class": "recommend",
    "drug_class_any": "recommend", "drug_classes": "recommend", "ingredient_class": "recommend",
    "prescribe": "recommend", "alternative": "recommend", "adjunct_option": "recommend",
}
# 약제·비약물 항목이 섞인 필드 — 풀리는 것만 연결
LOOSE_DRUG_FIELDS = {"options": "option", "first_line": "option", "first_choice": "option", "prefer": "option", "add": "option", "with": "option"}
LOOSE_TEST_FIELDS = {"options": "option"}
ROUTES = {"oral", "topical", "injection", "transdermal", "inhaled", "other"}
UNIT_ALIASES = {"μ": "u", "µ": "u", "㎎": "mg", "㎗": "dl", "ℓ": "l"}


def normalize_unit(unit: str) -> str:
    text = unicodedata.normalize("NFKC", unit)
    for old, new in UNIT_ALIASES.items():
        text = text.replace(old, new)
    return re.sub(r"\s+", "", text).lower()


def normalize_name(name: str) -> str:
    """EMR 항목명 비교용: 대소문자·공백·기호 무시."""
    return re.sub(r"[^0-9a-z가-힣+*]", "", unicodedata.normalize("NFKC", name).lower())


@dataclass
class Mappings:
    tests: dict[str, dict] = field(default_factory=dict)          # concept id → entry(+section)
    test_index: dict[str, str] = field(default_factory=dict)      # corpus id → concept id
    ingredients: dict[str, dict] = field(default_factory=dict)
    products: dict[str, dict] = field(default_factory=dict)
    classes: dict[str, dict] = field(default_factory=dict)
    drug_index: dict[str, tuple[str, str]] = field(default_factory=dict)  # corpus id → (kind, concept id)
    loinc: dict[str, dict] = field(default_factory=dict)
    rxnorm: dict[str, dict] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)


def _read_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def load_mappings() -> Mappings:
    m = Mappings(loinc=_read_json(VERIFIED_DIR / "loinc.json"), rxnorm=_read_json(VERIFIED_DIR / "rxnorm.json"))
    tests = _read_yaml(MAPPINGS_DIR / "tests.yaml")
    for section in tests:
        if section not in TEST_SECTIONS:
            m.errors.append(f"tests.yaml: 알 수 없는 구역 {section!r}")
    for section in TEST_SECTIONS:
        for entry in tests.get(section) or ():
            cid = entry.get("id")
            if not cid or cid in m.tests:
                m.errors.append(f"tests.yaml {section}: id 누락 또는 중복 {cid!r}")
                continue
            m.tests[cid] = {**entry, "section": section}
            for alias in [cid, *(entry.get("corpus_ids") or ())]:
                if m.test_index.setdefault(alias, cid) != cid:
                    m.errors.append(f"tests.yaml: 말뭉치 식별자 {alias!r} 가 {m.test_index[alias]}·{cid} 에 중복")

    drugs = _read_yaml(MAPPINGS_DIR / "drugs.yaml")
    for kind, target in (("ingredients", m.ingredients), ("products", m.products), ("classes", m.classes)):
        for entry in drugs.get(kind) or ():
            cid = entry.get("id")
            if not cid or cid in m.ingredients or cid in m.products or cid in m.classes:
                m.errors.append(f"drugs.yaml {kind}: id 누락 또는 중복 {cid!r}")
                continue
            target[cid] = entry
            for alias in [cid, *(entry.get("corpus_ids") or ())]:
                previous = m.drug_index.setdefault(alias, (kind, cid))
                if previous != (kind, cid):
                    m.errors.append(f"drugs.yaml: 말뭉치 식별자 {alias!r} 가 {previous[1]}·{cid} 에 중복")
    _validate(m)
    return m


def _validate(m: Mappings) -> None:
    for cid, entry in m.tests.items():
        for code in entry.get("loinc") or ():
            if code not in m.loinc:
                m.errors.append(f"tests.yaml {cid}: LOINC {code} 미확인 — tools/verify_codes.py 실행")
        for member in entry.get("members") or ():
            if m.tests.get(member, {}).get("section") != "labs":
                m.errors.append(f"tests.yaml {cid}: 구성 항목 {member!r} 가 labs 에 없음")
        for ref in entry.get("inputs") or ():
            if ref not in m.tests and ref != "sex":
                m.errors.append(f"tests.yaml {cid}: 입력 {ref!r} 미등록")
        if entry.get("from_exam") and entry["from_exam"] not in m.tests:
            m.errors.append(f"tests.yaml {cid}: from_exam {entry['from_exam']!r} 미등록")
        bounds = entry.get("range")
        if bounds is not None and not (isinstance(bounds, list) and len(bounds) == 2 and bounds[0] < bounds[1]):
            m.errors.append(f"tests.yaml {cid}: range 형식 오류")
        for unit, factor in (entry.get("convert") or {}).items():
            ok = isinstance(factor, (int, float)) or (isinstance(factor, dict) and set(factor) <= {"factor", "offset"} and "factor" in factor)
            if not ok:
                m.errors.append(f"tests.yaml {cid}: 환산 {unit!r} 형식 오류")
    seen_alias: dict[str, str] = {}
    for cid, entry in m.tests.items():
        for alias in entry.get("aliases") or ():
            key = normalize_name(alias)
            if seen_alias.setdefault(key, cid) != cid:
                m.errors.append(f"tests.yaml: EMR 별칭 {alias!r} 가 {seen_alias[key]}·{cid} 에 중복")

    ko_seen: dict[str, str] = {}
    for cid, entry in m.ingredients.items():
        if not entry.get("ko"):
            m.errors.append(f"drugs.yaml {cid}: 국내 성분명(ko) 없음")
        for name in entry.get("ko") or ():
            key = normalize_name(name)
            if ko_seen.setdefault(key, cid) != cid:
                m.errors.append(f"drugs.yaml: 국내 성분명 {name!r} 가 {ko_seen[key]}·{cid} 에 중복")
        if "rxnorm" in entry and entry["rxnorm"] is None:
            continue
        names = entry.get("rxnorm") or cid.replace("_", " ")
        for name in [names] if isinstance(names, str) else names:
            if name not in m.rxnorm:
                m.errors.append(f"drugs.yaml {cid}: RxNorm {name!r} 미확인 — tools/verify_codes.py 실행")
    for cid, entry in m.products.items():
        for comp in entry.get("components") or ():
            if comp not in m.ingredients:
                m.errors.append(f"drugs.yaml {cid}: 구성 성분 {comp!r} 미등록")
    for cid, entry in m.classes.items():
        for member in entry.get("members") or ():
            if member not in m.ingredients:
                m.errors.append(f"drugs.yaml 약효군 {cid}: 성분 {member!r} 미등록")
        for inc in entry.get("includes") or ():
            if inc not in m.classes:
                m.errors.append(f"drugs.yaml 약효군 {cid}: 포함 약효군 {inc!r} 미등록")
        if "derive_from" in entry and (entry["derive_from"] not in m.classes or not set(entry.get("routes") or ()) <= ROUTES):
            m.errors.append(f"drugs.yaml 약효군 {cid}: derive_from/routes 오류")
    for cid in m.classes:
        try:
            expand_class(m, cid)
        except RecursionError:
            m.errors.append(f"drugs.yaml 약효군 {cid}: includes 순환")


def expand_class(m: Mappings, class_id: str, _stack: tuple[str, ...] = ()) -> set[str]:
    if class_id in _stack:
        raise RecursionError(class_id)
    entry = m.classes[class_id]
    members = set(entry.get("members") or ())
    if "derive_from" in entry:
        members |= expand_class(m, entry["derive_from"], (*_stack, class_id))
    for inc in entry.get("includes") or ():
        members |= expand_class(m, inc, (*_stack, class_id))
    return members


def classes_of(m: Mappings) -> dict[str, set[str]]:
    """성분 id → 소속 약효군(includes 상위 포함). 복합제 classes 는 복합제에만 붙는다."""
    result: dict[str, set[str]] = defaultdict(set)
    for class_id in m.classes:
        for ingredient in expand_class(m, class_id):
            result[ingredient].add(class_id)
    return result


def _iter_refs(node, fields: dict[str, str]):
    """(필드, 역할, 문자열 값) — chunks 의 sources 는 건너뛴다."""
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "sources":
                continue
            if key in fields:
                for item in value if isinstance(value, list) else [value]:
                    if isinstance(item, str):
                        yield key, fields[key], item
            if isinstance(value, (dict, list)):
                yield from _iter_refs(value, fields)
    elif isinstance(node, list):
        for item in node:
            yield from _iter_refs(item, fields)


def resolve_drug(m: Mappings, token: str) -> tuple[str, str] | None:
    return m.drug_index.get(token)


def reorganize(m: Mappings, buckets: dict[str, list[dict]]) -> tuple[list[str], dict[str, list[dict]]]:
    """말뭉치 항목에 concepts 를 붙이고 개념별 재정리 파일 내용을 만든다. (오류, {파일명: 줄 목록})"""
    errors = list(m.errors)
    test_use: dict[str, list[dict]] = defaultdict(list)
    drug_use: dict[str, list[dict]] = defaultdict(list)
    class_use: dict[str, list[dict]] = defaultdict(list)
    for file_name, records in buckets.items():
        kind = file_name.removesuffix(".jsonl")
        for record in records:
            tests: dict[str, set[str]] = defaultdict(set)
            drugs: dict[str, set[str]] = defaultdict(set)
            drug_classes: dict[str, set[str]] = defaultdict(set)
            for key, role, token in _iter_refs(record, STRICT_TEST_FIELDS):
                concept = m.test_index.get(token)
                if concept is None:
                    errors.append(f"{record['id']}: 검사·변수 {key}={token!r} 개념 미등록(mappings/tests.yaml)")
                else:
                    tests[concept].add(role)
            for key, role, token in _iter_refs(record, LOOSE_TEST_FIELDS):
                if (concept := m.test_index.get(token)) is not None:
                    tests[concept].add(role)
            for fields, strict in ((STRICT_DRUG_FIELDS, True), (LOOSE_DRUG_FIELDS, False)):
                for key, role, token in _iter_refs(record, fields):
                    found = resolve_drug(m, token)
                    if found is None:
                        if strict:
                            errors.append(f"{record['id']}: 약제 {key}={token!r} 개념 미등록(mappings/drugs.yaml)")
                        continue
                    target_kind, concept = found
                    (drug_classes if target_kind == "classes" else drugs)[concept].add(role)
            record["concepts"] = {
                "tests": sorted(tests), "drugs": sorted(drugs), "drug_classes": sorted(drug_classes),
            }
            base = {"record_id": record["id"], "file": file_name, "kind": kind, "claim_id": record.get("claim_id"),
                    "ai_tier": record.get("ai_tier"), "topic": record.get("topic")}
            for concept, roles in tests.items():
                test_use[concept].append({**base, "roles": sorted(roles)})
            for concept, roles in drugs.items():
                drug_use[concept].append({**base, "roles": sorted(roles)})
            for concept, roles in drug_classes.items():
                class_use[concept].append({**base, "roles": sorted(roles)})

    membership = classes_of(m)
    outputs: dict[str, list[dict]] = {"test_concepts.jsonl": [], "drug_concepts.jsonl": [], "drug_classes.jsonl": []}
    for cid, entry in m.tests.items():
        section = entry["section"]
        outputs["test_concepts.jsonl"].append({
            "id": cid, "section": section, "value_source": VALUE_SOURCE[section],
            "name_ko": entry.get("name_ko"), "name_en": entry.get("name_en"), "kind": entry.get("kind"),
            "unit": entry.get("unit"), "convert": entry.get("convert") or {}, "range": entry.get("range"),
            "loinc": [{"code": code, "name": m.loinc[code]["name"]} for code in entry.get("loinc") or () if code in m.loinc],
            "corpus_ids": sorted({cid, *(entry.get("corpus_ids") or ())}),
            "aliases": entry.get("aliases") or [], "members": entry.get("members") or [],
            "inputs": entry.get("inputs") or [], "series": bool(entry.get("series")), "from_exam": entry.get("from_exam"),
            "note": entry.get("note"), "used_by": test_use.get(cid, []),
        })
    for cid, entry in m.ingredients.items():
        names = None if ("rxnorm" in entry and entry["rxnorm"] is None) else (entry.get("rxnorm") or cid.replace("_", " "))
        rx = [m.rxnorm[name] for name in ([names] if isinstance(names, str) else names or []) if name in m.rxnorm]
        via_class = [
            {**use, "via_class": class_id}
            for class_id in sorted(membership.get(cid, ()))
            for use in class_use.get(class_id, [])
        ]
        outputs["drug_concepts.jsonl"].append({
            "id": cid, "type": "ingredient", "name_ko": (entry.get("ko") or [None])[0], "ko": entry.get("ko") or [],
            "rxnorm": [{"rxcui": r["rxcui"], "name": r["name"], "tty": r["tty"]} for r in rx],
            "fda_epc": sorted({c for r in rx for c in r.get("epc", ())}),
            "classes": sorted(membership.get(cid, ())), "corpus_ids": sorted({cid, *(entry.get("corpus_ids") or ())}),
            "note": entry.get("note"), "used_by": drug_use.get(cid, []), "used_by_class": via_class,
        })
    for cid, entry in m.products.items():
        outputs["drug_concepts.jsonl"].append({
            "id": cid, "type": "product", "components": entry["components"],
            "classes": sorted(set(entry.get("classes") or ()) | set().union(*(membership.get(c, set()) for c in entry["components"]))),
            "corpus_ids": sorted({cid, *(entry.get("corpus_ids") or ())}), "used_by": drug_use.get(cid, []),
        })
    for cid, entry in m.classes.items():
        outputs["drug_classes.jsonl"].append({
            "id": cid, "name_ko": entry.get("name_ko"), "members": sorted(expand_class(m, cid)),
            "includes": entry.get("includes") or [], "exhaustive": bool(entry.get("exhaustive")),
            "derivable": entry.get("derivable", True), "systemic_only": bool(entry.get("systemic_only")),
            "derive_from": entry.get("derive_from"), "routes": entry.get("routes") or [], "corpus_ids": sorted({cid, *(entry.get("corpus_ids") or ())}),
            "note": entry.get("note"), "used_by": class_use.get(cid, []),
        })
    return errors, outputs
