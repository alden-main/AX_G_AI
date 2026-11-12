"""Check a patient's prescriptions against the HIRA DUR lists (PostgreSQL schema ``dur``).

The active dataset is read once into memory (``ax_g_ai.dur_store.reader``); see
``ax_g_ai.dur_store.snapshot`` for the source and the conversion.

Prescriptions are matched by EMR ``medCode`` = 9-digit 보험 제품코드 only (no name guessing);
unmatched codes are reported so the site can fix them. Alerts:

- 병용금기: two current prescriptions whose ingredients form a listed pair (the list's 비고 —
  e.g. "48시간 이내 병용금기", MTX weekly-dose conditions — is shown verbatim, never evaluated).
- 연령금기: the patient's age meets the listed age condition (age unknown → no alert).
- 임부금기: pregnancy recorded → ``met``; pregnancy unknown in a woman aged 15–49 → ``possible``.
- 노인주의: age 65 or older.

Drugs named in a chat question (``find_drugs``) are checked as drugs to add: matched by the Korean
ingredient name in single-ingredient product names (salt suffixes dropped, e.g. 와파린나트륨 → 와파린) or the
English ingredient name, against every dosage-form code of that ingredient. They carry ``proposed: true``.

Dose-related DUR lists (용량주의, 투여기간주의) are not in the downloaded files yet.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

ELDERLY_AGE = 65
REPRODUCTIVE_AGE = (15, 49)
_WEEKS_PER_YEAR = 52.1775
_CONDITIONS = {"미만": lambda value, limit: value < limit, "이하": lambda value, limit: value <= limit,
               "이상": lambda value, limit: value >= limit, "초과": lambda value, limit: value > limit}
_GRADE_LABELS = {"1": "1등급", "2": "2등급"}
# Korean ingredient name in a product name: "클로시드정250밀리그램(클래리트로마이신)_(0.25g/1정)".
_KO_INGREDIENT = re.compile(r"\(([^()]*[가-힣][^()]*)\)_\(")
_KO_SUFFIXES = re.compile(
    r"(?:[\d.]+(?:밀리그람|밀리그램|마이크로그램|그램|mg|g)?|염산염|황산염|수화물|무수물|나트륨|칼륨|칼슘|베실산염|말레산염|타르타르산염"
    r"|숙신산염|메실산염|브롬화수소산염|인산염|아세트산염|푸마르산염|시트르산염|질산염|장용과립|제피과립|장용|서방)$"
)
_MIN_KO, _MIN_EN = 3, 4


class DurUnavailableError(RuntimeError):
    """The DUR dataset is not loaded or has an unexpected shape."""


Row = Mapping[str, Any]


@dataclass(frozen=True)
class DurTables:
    """One DUR dataset, indexed for lookups by product code and ingredient code."""

    source: Mapping[str, str]
    products: Mapping[str, Row]
    pairs_by_a: Mapping[str, tuple[Row, ...]]
    ages: Mapping[str, tuple[Row, ...]]
    pregnancy: Mapping[str, tuple[Row, ...]]
    elderly: Mapping[str, tuple[Row, ...]]
    dataset_id: int | None = None

    @classmethod
    def from_rows(cls, source: Mapping[str, str], rows: Mapping[str, Iterable[Row]], *, dataset_id: int | None = None) -> "DurTables":
        def by(table: str, key: str) -> dict[str, tuple[Row, ...]]:
            grouped: dict[str, list[Row]] = defaultdict(list)
            for row in rows.get(table, ()):
                grouped[row[key]].append(row)
            return {code: tuple(items) for code, items in grouped.items()}

        products = {row["product_code"]: row for row in rows.get("product", ())}
        if not products:
            raise DurUnavailableError("dur dataset has no products")
        return cls(dict(source), products, by("pair_rule", "ingredient_a"), by("age_rule", "ingredient_code"),
                   by("pregnancy_rule", "ingredient_code"), by("elderly_rule", "ingredient_code"), dataset_id)


@dataclass(frozen=True)
class ProposedDrug:
    """A drug named in the question: every DUR ingredient code (dosage form) of that ingredient."""

    name: str  # as written in the question (Korean ingredient name or English name)
    ingredient_name: str
    codes: frozenset[str]


@dataclass(frozen=True)
class DurResult:
    alerts: tuple[Mapping[str, Any], ...] = ()
    unmapped: tuple[Mapping[str, Any], ...] = ()
    source: Mapping[str, str] = field(default_factory=dict)
    proposed: tuple[Mapping[str, Any], ...] = ()

    def as_dict(self) -> dict[str, object]:
        result: dict[str, object] = {"alerts": [dict(alert) for alert in self.alerts],
                                     "unmapped": [dict(item) for item in self.unmapped], "source": dict(self.source)}
        if self.proposed:
            result["proposed"] = [dict(item) for item in self.proposed]
        return result


def _en_base(name: str) -> str:
    """First word of an English ingredient name: 'warfarin sodium' → 'warfarin', 'aspirin(enteric coated)' → 'aspirin'."""
    return re.split(r"[\s(,+]", name.strip().lower(), maxsplit=1)[0]


def _ko_bases(name: str) -> set[str]:
    names = {name}
    while (stripped := _KO_SUFFIXES.sub("", name)) != name and stripped:
        name = stripped
        names.add(name)
    return {item for item in names if len(item) >= _MIN_KO}


class _DrugNames:
    """Korean and English ingredient names → English base name → ingredient codes, from the product list."""

    def __init__(self, products: Iterable[Row]) -> None:
        codes: dict[str, set[str]] = defaultdict(set)
        labels: dict[str, str] = {}
        ko: dict[str, str] = {}
        for row in products:
            english = str(row.get("ingredient_name") or "")
            base = _en_base(english)
            if "+" in english or not base:
                continue
            codes[base].add(row["ingredient_code"])
            if base not in labels or len(english) < len(labels[base]):
                labels[base] = english  # shortest form: "aspirin", not "aspirin(enteric coated) (as aspirin)"
            match = _KO_INGREDIENT.search(str(row.get("product_name") or ""))
            # Combination products ("A/B") name several ingredients but carry one code: skip them.
            if match and not re.search(r"[,/·+]", match.group(1)):
                for key in _ko_bases(re.sub(r"\s+", "", match.group(1))):
                    ko.setdefault(key, base)
        self._codes = {base: frozenset(items) for base, items in codes.items()}
        self._labels = labels
        self._ko = ko
        self._ko_pattern = self._pattern(ko)
        english_keys = [base for base in self._codes if len(base) >= _MIN_EN and base.isalpha()]
        self._en_pattern = re.compile(
            r"(?<![a-z])(" + "|".join(re.escape(key) for key in sorted(english_keys, key=len, reverse=True)) + r")(?![a-z])"
        ) if english_keys else None

    @staticmethod
    def _pattern(keys: Iterable[str]) -> re.Pattern[str] | None:
        ordered = sorted(keys, key=len, reverse=True)
        return re.compile("|".join(re.escape(key) for key in ordered)) if ordered else None

    def find(self, text: str) -> tuple[ProposedDrug, ...]:
        found: dict[str, str] = {}
        if self._ko_pattern is not None:
            for match in self._ko_pattern.finditer(re.sub(r"\s+", "", text)):
                found.setdefault(self._ko[match.group(0)], match.group(0))
        if self._en_pattern is not None:
            for match in self._en_pattern.finditer(text.lower()):
                found.setdefault(match.group(1), match.group(1))
        return tuple(ProposedDrug(name, self._labels[base], self._codes[base]) for base, name in found.items())


class DurChecker:
    def __init__(self, tables: DurTables) -> None:
        self._tables = tables
        self.source = dict(tables.source)
        self._names = _DrugNames(tables.products.values())

    def __len__(self) -> int:
        return len(self._tables.products)

    @property
    def dataset_id(self) -> int | None:
        return self._tables.dataset_id

    def find_drugs(self, text: str) -> tuple[ProposedDrug, ...]:
        """Drugs named in ``text`` (a chat question or memo), matched to DUR ingredient codes."""
        return self._names.find(text) if text else ()

    def check(
        self, medications: Sequence[tuple[str | None, str | None]], *, age_years: float | None, sex: str | None,
        pregnant: bool | None, proposed: Sequence[ProposedDrug] = (),
    ) -> DurResult:
        prescribed: list[dict[str, Any]] = []
        unmapped: list[dict[str, Any]] = []
        for code, name in medications:
            row = self._tables.products.get(code.strip()) if code and code.strip() else None
            if row is None:
                unmapped.append({"med_code": code, "med_name": name, "detail": "DUR 목록에 없는 제품코드(9자리 보험코드) — DUR 점검 제외"})
                continue
            prescribed.append({"med_code": row["product_code"], "med_name": name or row["product_name"], "product_name": row["product_name"],
                               "ingredient_code": row["ingredient_code"], "ingredient_name": row["ingredient_name"]})
        # Drugs to add, named in the question: checked like prescriptions under every dosage-form code. A drug
        # already prescribed is checked once, as the prescription.
        current = {_en_base(item["ingredient_name"]) for item in prescribed}
        recognized: list[dict[str, Any]] = []
        for drug in proposed:
            if _en_base(drug.ingredient_name) in current:
                continue
            current.add(_en_base(drug.ingredient_name))
            recognized.append({"name": drug.name, "ingredient_name": drug.ingredient_name})
            prescribed.extend({"med_code": None, "med_name": drug.name, "product_name": None, "ingredient_code": code,
                               "ingredient_name": drug.ingredient_name, "proposed": True} for code in sorted(drug.codes))
        alerts = [
            *self._pairs(prescribed),
            *self._ages(prescribed, age_years),
            *self._pregnancy(prescribed, age_years, sex, pregnant),
            *self._elderly(prescribed, age_years),
        ]
        # One alert per finding: a proposed drug's dosage forms share the same rules.
        unique = {json.dumps(alert, ensure_ascii=False, sort_keys=True): alert for alert in alerts}
        return DurResult(tuple(unique.values()), tuple(unmapped), dict(self.source), tuple(recognized))

    @staticmethod
    def _product(item: Mapping[str, Any]) -> dict[str, Any]:
        product = {key: item[key] for key in ("med_code", "med_name", "ingredient_name")}
        if item.get("proposed"):
            product["proposed"] = True
        return product

    @staticmethod
    def _notice(rule: Row) -> dict[str, Any]:
        return {"no": rule.get("notice_no"), "date": rule.get("notice_date")}

    def _pairs(self, prescribed: list[dict[str, Any]]) -> Iterable[dict[str, Any]]:
        by_ingredient: dict[str, list[dict[str, Any]]] = {}
        for item in prescribed:
            by_ingredient.setdefault(item["ingredient_code"], []).append(item)
        seen: set[tuple[frozenset[str], str]] = set()
        for a in sorted(by_ingredient):
            for rule in self._tables.pairs_by_a.get(a, ()):
                b = rule["ingredient_b"]
                if b == a or b not in by_ingredient or (frozenset((a, b)), rule["detail"]) in seen:
                    continue
                seen.add((frozenset((a, b)), rule["detail"]))
                yield {
                    "type": "병용금기", "status": "met",
                    "products": [self._product(item) for item in (*by_ingredient[a], *by_ingredient[b])],
                    "detail": rule["detail"], **({"condition_note": rule["note"]} if rule.get("note") else {}),
                    "notice": self._notice(rule),
                }

    def _ages(self, prescribed: list[dict[str, Any]], age_years: float | None) -> Iterable[dict[str, Any]]:
        if age_years is None:
            return
        for item in prescribed:
            for rule in self._tables.ages.get(item["ingredient_code"], ()):
                unit, age = rule["unit"], float(rule["age"])
                value = age_years if unit == "세" else age_years * 12 if unit == "개월" else age_years * _WEEKS_PER_YEAR if unit == "주" else None
                test = _CONDITIONS.get(rule["condition"])
                if value is None or test is None or not test(value, age):
                    continue
                yield {"type": "연령금기", "status": "met", "products": [self._product(item)],
                       "age_condition": f"{age:g}{unit} {rule['condition']}", "detail": rule["detail"],
                       "notice": self._notice(rule)}

    def _pregnancy(
        self, prescribed: list[dict[str, Any]], age_years: float | None, sex: str | None, pregnant: bool | None,
    ) -> Iterable[dict[str, Any]]:
        if pregnant is False or sex == "male":
            return
        if pregnant is None:
            # Unknown pregnancy: only flag women of reproductive age, as a question to confirm.
            if sex != "female" or age_years is None or not REPRODUCTIVE_AGE[0] <= age_years <= REPRODUCTIVE_AGE[1]:
                return
        status = "met" if pregnant else "possible"
        for item in prescribed:
            for rule in self._tables.pregnancy.get(item["ingredient_code"], ()):
                grade = rule.get("grade")
                alert = {"type": "임부금기", "status": status, "products": [self._product(item)],
                         "grade": _GRADE_LABELS.get(grade, grade or "미표기"), "detail": rule["detail"],
                         "notice": self._notice(rule)}
                if status == "possible":
                    alert["missing"] = [{"kind": "finding", "key": "pregnancy", "label": "임신 여부(가임기 여성)"}]
                yield alert

    def _elderly(self, prescribed: list[dict[str, Any]], age_years: float | None) -> Iterable[dict[str, Any]]:
        if age_years is None or age_years < ELDERLY_AGE:
            return
        for item in prescribed:
            for rule in self._tables.elderly.get(item["ingredient_code"], ()):
                yield {"type": "노인주의", "status": "met", "category": rule["category"], "products": [self._product(item)],
                       "detail": rule["detail"], "notice": self._notice(rule),
                       "note": "노인 기준 65세 이상은 DUR 운영 기준(보도 확인) — 공고 원문 대조 필요"}
