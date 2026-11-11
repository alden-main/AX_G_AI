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

Dose-related DUR lists (용량주의, 투여기간주의) are not in the downloaded files yet.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

ELDERLY_AGE = 65
REPRODUCTIVE_AGE = (15, 49)
_WEEKS_PER_YEAR = 52.1775
_CONDITIONS = {"미만": lambda value, limit: value < limit, "이하": lambda value, limit: value <= limit,
               "이상": lambda value, limit: value >= limit, "초과": lambda value, limit: value > limit}
_GRADE_LABELS = {"1": "1등급", "2": "2등급"}


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
class DurResult:
    alerts: tuple[Mapping[str, Any], ...] = ()
    unmapped: tuple[Mapping[str, Any], ...] = ()
    source: Mapping[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {"alerts": [dict(alert) for alert in self.alerts], "unmapped": [dict(item) for item in self.unmapped],
                "source": dict(self.source)}


class DurChecker:
    def __init__(self, tables: DurTables) -> None:
        self._tables = tables
        self.source = dict(tables.source)

    def __len__(self) -> int:
        return len(self._tables.products)

    @property
    def dataset_id(self) -> int | None:
        return self._tables.dataset_id

    def check(
        self, medications: Sequence[tuple[str | None, str | None]], *, age_years: float | None, sex: str | None,
        pregnant: bool | None,
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
        alerts = [
            *self._pairs(prescribed),
            *self._ages(prescribed, age_years),
            *self._pregnancy(prescribed, age_years, sex, pregnant),
            *self._elderly(prescribed, age_years),
        ]
        return DurResult(tuple(alerts), tuple(unmapped), dict(self.source))

    @staticmethod
    def _product(item: Mapping[str, Any]) -> dict[str, Any]:
        return {key: item[key] for key in ("med_code", "med_name", "ingredient_name")}

    @staticmethod
    def _notice(rule: Row) -> dict[str, Any]:
        return {"no": rule.get("notice_no"), "date": rule.get("notice_date")}

    def _pairs(self, prescribed: list[dict[str, Any]]) -> Iterable[dict[str, Any]]:
        by_ingredient: dict[str, list[dict[str, Any]]] = {}
        for item in prescribed:
            by_ingredient.setdefault(item["ingredient_code"], []).append(item)
        for a in sorted(by_ingredient):
            for rule in self._tables.pairs_by_a.get(a, ()):
                b = rule["ingredient_b"]
                if b == a or b not in by_ingredient:
                    continue
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
