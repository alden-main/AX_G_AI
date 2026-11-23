"""Check a patient's prescriptions against the HIRA DUR lists (PostgreSQL schema ``dur``).

The active dataset is read once into memory (``ax_g_ai.dur_store.reader``); see
``ax_g_ai.dur_store.snapshot`` for the source and the conversion.

Prescriptions are matched by EMR ``medCode`` = 9-digit 보험 제품코드 only (no name guessing);
unmatched codes are reported so the site can fix them. Alerts:

- 병용금기: two current prescriptions whose ingredients form a listed pair (the list's 비고 —
  e.g. "48시간 이내 병용금기", MTX weekly-dose conditions — is shown verbatim, never evaluated).
- 연령금기: the patient's age meets the listed age condition (age unknown → no alert).
- 임부금기: pregnancy recorded → ``met``; pregnancy unknown in a woman aged 15–49, or memo and question disagreeing
  on it (any age, unless male) → ``possible``.
- 노인주의: age 65 or older.

Drugs named in a chat question (``find_drugs``) are checked as drugs to add, against every dosage-form code
of that ingredient. They carry ``proposed: true``. Only single-ingredient codes are named this way: the product
list gives a combination product one row per component under one ingredient code, so a code whose rows name
more than one ingredient (or a name listing several) is a combination and is checked only as a prescription.
An ingredient is its English name without salt and hydrate words ("amlodipine besylate" → "amlodipine",
"sodium alendronate" → "alendronate"; "potassium chloride", all salt words, stays whole). It is named by that
key, its full English names, or the Korean name in its single-ingredient product names (salt suffixes
dropped, e.g. 와파린나트륨 → 와파린) — each Korean name linked to the codes of its own products.

Dose-related DUR lists (용량주의, 투여기간주의) are not in the downloaded files yet.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ax_g_ai.services.term_matcher import TermMatcher

ELDERLY_AGE = 65
REPRODUCTIVE_AGE = (15, 49)
_WEEKS_PER_YEAR = 52.1775
_CONDITIONS = {"미만": lambda value, limit: value < limit, "이하": lambda value, limit: value <= limit,
               "이상": lambda value, limit: value >= limit, "초과": lambda value, limit: value > limit}
_GRADE_LABELS = {"1": "1등급", "2": "2등급"}
# Korean ingredient name in a product name: "클로시드정250밀리그램(클래리트로마이신)_(0.25g/1정)". Later parentheses
# may hold an export name or the packaging instead: "클리신정(클로닉신리시네이트)(수출명:CLICIN Tab)_(…)",
# "트리세녹스주(삼산화비소)(바이알)_(…)"; those are skipped.
_KO_PARENTHESES = re.compile(r"\(([^()]*[가-힣][^()]*)\)")
_KO_NOT_INGREDIENT = frozenset({
    "1회용", "일회용", "바이알", "앰플", "프리필드", "카트리지", "주사용", "수출용", "관납용", "군납용", "소아용", "병", "별첨",
    "유전자재조합", "재조합", "미분화", "단클론항체",
})
_KO_RECOMBINANT = re.compile(r",?유전자재조합$")  # "니볼루맙,유전자재조합": a biologic, not a combination
_KO_SUFFIXES = re.compile(
    r"(?:[\d.]+(?:밀리그람|밀리그램|마이크로그램|그램|mg|g)?|염산염|황산염|수화물|무수물|나트륨|칼륨|칼슘|마그네슘|베실산염|말레산염|타르타르산염"
    r"|숙신산염|메실산염|브롬화수소산염|인산염|아세트산염|푸마르산염|시트르산염|질산염|장용과립|제피과립|장용|서방)$"
)
_MIN_KO, _MIN_EN = 3, 4
# Counter-ion, hydrate, co-former and particle-form words: dropped to find the ingredient ("atorvastatin calcium" → "atorvastatin").
# Esters (furoate, propionate, valerate) are part of the drug and stay. A name made only of these words
# ("sodium bicarbonate", "potassium chloride") is kept whole.
_SALT_WORDS = frozenset("""
    hydrochloride dihydrochloride hcl hydrobromide hydroiodide hydrate monohydrate dihydrate trihydrate hemihydrate
    sesquihydrate anhydrous sodium disodium monosodium trisodium potassium dipotassium calcium magnesium
    besylate besilate camsylate camsilate mesylate mesilate dimesylate maleate fumarate hemifumarate tartrate
    bitartrate succinate hemisuccinate citrate sulfate bisulfate hydrogen phosphate diphosphate dihydrogen acetate
    oxalate tosylate tosilate napadisilate napsylate xinafoate chloride bromide iodide bicarbonate carbonate
    hydroxide oxide nitrate lactate gluconate salt l-proline l-lysine lysine arginine l-arginine meglumine
    tromethamine erbumine granule granules micronized esilate esylate hemitartrate strontium tetrahydrate diluted solution
""".split())
# One ingredient the HIRA list spells two ways (found in the 2026.6 list; reviewed as the same active ingredient).
# Applied after the salt words are dropped. A spelling missing here only makes the variants two ingredients:
# a code carrying both is then treated as a combination and is checked as a prescription only.
_SAME_INGREDIENT = {
    "esomeprazol": "esomeprazole",
    "ibandronic acid": "ibandronate",
    "povidone-iodine": "povidone iodine",
    "cloazapine": "clozapine",
    "diphenylhydantoin": "phenytoin",
    "cyclosporin": "cyclosporine",
    "microemulsion cyclosporine": "cyclosporine",
    "iodised oil": "ethyl esters of the iodised fatty acids of poppyseed oil",
    "recombinant human tissue-type plasminogen activator": "alteplase",
    "leuprorelin": "leuprolide",
    "methylergonovine": "methylergometrine",
    "benedipine": "benidipine",
    "raloxifen": "raloxifene",
    "albendazol": "albendazole",
    "candesartan cilexetil": "candesartan",
    "baloxavir marboxil": "baloxavir",
    "dexibuprofen d.c": "dexibuprofen",
    "menotrophin h.p": "menotrophin",
    "s-amlodipine nicotinate": "s-amlodipine",
    "estrogen": "conjugated estrogen",  # "estrogen(conjugated)": the list's only plain "estrogen"
    "nanoliposomal irinotecan": "irinotecan",  # same active ingredient; DUR rules are per code either way
}
# A name that lists several ingredients, or a product the list calls a combination ("복합체" is a complex, one drug).
_COMBINATION = re.compile(r"[,+/·]|복합(?!체)|함유")
# A strength or amount in a name ("24mg", "0.16g", "1KI.U", "300unit"), not a name starting with a digit ("5-fluorouracil").
_STRENGTH = re.compile(r"(?=[\d.,]*\d)[\d.,]+[a-zμ㎍㎎%.]*(?:/[\w.]*)?\)*", re.I)


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
    # (ingredient code, ingredient name) as the rule lists write them: they can name components the product rows do not.
    ingredient_names: tuple[tuple[str, str], ...] = ()

    @classmethod
    def from_rows(cls, source: Mapping[str, str], rows: Mapping[str, Iterable[Row]], *, dataset_id: int | None = None) -> DurTables:
        def by(table: str, key: str) -> dict[str, tuple[Row, ...]]:
            grouped: dict[str, list[Row]] = defaultdict(list)
            for row in rows.get(table, ()):
                grouped[row[key]].append(row)
            return {code: tuple(items) for code, items in grouped.items()}

        products = {row["product_code"]: row for row in rows.get("product", ())}
        names = [(row["ingredient_code"], row["ingredient_name"]) for table in ("age_rule", "pregnancy_rule", "elderly_rule")
                 for row in rows.get(table, ()) if row.get("ingredient_name")]
        names += [(row[code], row[name]) for row in rows.get("pair_rule", ())
                  for code, name in (("ingredient_a", "name_a"), ("ingredient_b", "name_b")) if row.get(name)]
        if not products:
            raise DurUnavailableError("dur dataset has no products")
        return cls(dict(source), products, by("pair_rule", "ingredient_a"), by("age_rule", "ingredient_code"),
                   by("pregnancy_rule", "ingredient_code"), by("elderly_rule", "ingredient_code"), dataset_id, tuple(names))


@dataclass(frozen=True)
class ProposedDrug:
    """A drug named in the question: every single-ingredient DUR code (dosage form) of that ingredient."""

    name: str  # as written in the question (Korean ingredient name or English name)
    ingredient_name: str
    codes: frozenset[str]
    # Ingredient keys (``_ingredient_key``; two when the list spells one drug two ways): a drug already prescribed is
    # checked as the prescription.
    keys: frozenset[str] = frozenset()


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


def _bare_words(name: str) -> list[str]:
    """The words of a name without parentheses and strengths; separators (",", "+", "/") are kept."""
    text = name.lower()
    while (stripped := re.sub(r"\([^()]*\)", " ", text)) != text:
        text = stripped
    return [word for word in text.split() if not _STRENGTH.fullmatch(word)]


def _english_name(name: str) -> str:
    """'Amlodipine Besylate (as amlodipine 5mg)' → 'amlodipine besylate': no parentheses, strengths or extra spaces."""
    words = [word.strip(".,()[]") for word in _bare_words(name)]
    return " ".join("extract" if word == "ext" else word for word in words if word)


def _is_combination_name(name: str) -> bool:
    """'linagliptin, metformin hydrochloride' lists two ingredients; 'plerixafor 24mg(20mg/mL)' lists one."""
    return _COMBINATION.search(" ".join(_bare_words(name))) is not None


def _salt_free(name: str) -> str:
    """An English name without its salt and hydrate words, unless nothing else is left."""
    full = _english_name(name)
    kept = [word for word in full.split() if word not in _SALT_WORDS]
    return " ".join(kept) if kept else full


def _ingredient_key(name: str) -> str:
    """The ingredient an English name refers to (``_salt_free``, then one spelling for ``_SAME_INGREDIENT``)."""
    key = _salt_free(name)
    return _SAME_INGREDIENT.get(key, key)


def _distinct_ingredients(keys: Iterable[str]) -> frozenset[str]:
    """Keys naming different ingredients: a key extending another ("clonixin lysinate", "clonixin") is the same one."""
    ordered = sorted(set(keys), key=len)
    kept: list[str] = []
    for key in ordered:
        if not any(key.startswith(shorter) for shorter in kept):
            kept.append(key)
    return frozenset(kept)


def _korean_ingredient(product_name: str) -> str | None:
    """The Korean ingredient name in a product name (spaces removed), or None when it names none."""
    head = product_name.split("_(", 1)[0]
    candidates = []
    for group in _KO_PARENTHESES.findall(head):
        name = _KO_RECOMBINANT.sub("", re.sub(r"\s+", "", group))
        if name and ":" not in name and name not in _KO_NOT_INGREDIENT:
            candidates.append(name)
    return candidates[-1] if candidates else None


def _ko_bases(name: str) -> set[str]:
    names = {name}
    while (stripped := _KO_SUFFIXES.sub("", name)) != name and stripped:
        name = stripped
        names.add(name)
    return {item for item in names if len(item) >= _MIN_KO}


class _DrugNames:
    """Korean and English ingredient names → ingredient key → single-ingredient codes, from the product list."""

    def __init__(self, products: Iterable[Row], rule_names: Iterable[tuple[str, str]] = ()) -> None:
        """``rule_names``: (ingredient code, ingredient name) from the rule lists, which may name a combination's
        other components where every product row names only one of them."""
        rows = list(products)
        raw_names: dict[str, set[str]] = defaultdict(set)
        for row in rows:
            raw_names[row["ingredient_code"]].add(str(row.get("ingredient_name") or "").strip())
        for code, name in rule_names:
            if code in raw_names and name:
                raw_names[code].add(name.strip())
        # Ingredient keys of every code: one for a single ingredient, one per component for a combination.
        self._components = {
            code: _distinct_ingredients(_ingredient_key(name) for name in names if _english_name(name))
            for code, names in raw_names.items()
        }
        combinations = {code for code, names in raw_names.items()
                        if len(self._components[code]) != 1 or any(_is_combination_name(name) for name in names)}
        codes: dict[str, set[str]] = defaultdict(set)
        labels: dict[str, str] = {}
        label_rank: dict[str, tuple[bool, int, str]] = {}
        english: dict[str, str] = {}  # full English name or ingredient key → ingredient key
        korean: dict[str, set[str]] = defaultdict(set)  # Korean name (and its salt-free bases) → ingredient keys
        for row in rows:
            code = row["ingredient_code"]
            if code in combinations:
                continue
            name = str(row.get("ingredient_name") or "").strip()
            key = _ingredient_key(name)
            if not key:  # no English ingredient name: it cannot be named
                continue
            codes[key].add(code)
            # The usual spelling first, then the shortest form: "aspirin", not "aspirin(enteric coated) (as aspirin)".
            rank = (_salt_free(name) != key, len(name), name)
            if key not in labels or rank < label_rank[key]:
                labels[key], label_rank[key] = name, rank
            for alias in (key, _english_name(name), _salt_free(name)):  # also the list's other spelling ("diphenylhydantoin")
                english.setdefault(alias, key)
            korean_name = _korean_ingredient(str(row.get("product_name") or ""))
            # A product name that lists several Korean ingredients ("A/B") is a combination even when coded alone.
            if korean_name and not _COMBINATION.search(korean_name):
                for base in _ko_bases(korean_name):
                    korean[base].add(key)
        # The first word alone ("clobetasol" for "clobetasol propionate") names an ingredient only when no other
        # ingredient starts with it: "ethyl", "insulin" or a salt word name several and stay unmatched.
        first_words: dict[str, set[str]] = defaultdict(set)
        for key in codes:
            first_words[key.split()[0]].add(key)
        for word, keys in first_words.items():
            if len(keys) == 1 and word not in _SALT_WORDS and word not in english:
                english[word] = next(iter(keys))
        self._codes = {key: frozenset(items) for key, items in codes.items()}
        self._labels = labels
        # Leftmost-longest, non-overlapping: "와파린나트륨" is one drug, not "와파린" plus a fragment.
        self._ko_names = TermMatcher((name, key) for name, keys in korean.items() for key in sorted(keys))
        self._en_names = TermMatcher(
            ((alias, key) for alias, key in english.items() if len(alias) >= _MIN_EN and re.fullmatch(r"[a-z0-9][a-z0-9 -]*[a-z]", alias)),
            boundary="a-z",
        )

    def components(self, code: str) -> frozenset[str]:
        """Ingredient keys of an ingredient code (several for a combination product)."""
        return self._components.get(code, frozenset())

    def find(self, text: str) -> tuple[ProposedDrug, ...]:
        """Drugs named in ``text``. A Korean name the list spells two ways in English ("류프로렐린": leuprolide,
        leuprorelin) is one drug with the codes of both."""
        found: list[ProposedDrug] = []
        seen: set[str] = set()
        matches = [(match.term, match.values) for match in self._ko_names.longest(re.sub(r"\s+", "", text))]
        matches += [(match.term, match.values) for match in self._en_names.longest(text.lower())]
        for name, keys in matches:
            if seen.intersection(keys):
                continue
            seen.update(keys)
            label = min((self._labels[key] for key in keys), key=lambda item: (len(item), item))
            codes = frozenset(code for key in keys for code in self._codes[key])
            found.append(ProposedDrug(name, label, codes, frozenset(keys)))
        return tuple(found)


class DurChecker:
    def __init__(self, tables: DurTables) -> None:
        self._tables = tables
        self.source = dict(tables.source)
        self._names = _DrugNames(tables.products.values(), tables.ingredient_names)

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
        pregnant: bool | None, proposed: Sequence[ProposedDrug] = (), pregnancy_conflict: bool = False,
    ) -> DurResult:
        """``pregnancy_conflict``: the memo and the question disagree on pregnancy (``pregnant`` is then None)."""
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
        # "Already prescribed" is by ingredient: a component of a prescribed combination counts too.
        current = {key for item in prescribed for key in self._names.components(item["ingredient_code"])}
        recognized: list[dict[str, Any]] = []
        for drug in proposed:
            if current.intersection(drug.keys):
                continue
            current.update(drug.keys)
            recognized.append({"name": drug.name, "ingredient_name": drug.ingredient_name})
            prescribed.extend({"med_code": None, "med_name": drug.name, "product_name": None, "ingredient_code": code,
                               "ingredient_name": drug.ingredient_name, "proposed": True} for code in sorted(drug.codes))
        alerts = [
            *self._pairs(prescribed),
            *self._ages(prescribed, age_years),
            *self._pregnancy(prescribed, age_years, sex, pregnant, pregnancy_conflict),
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
        conflict: bool = False,
    ) -> Iterable[dict[str, Any]]:
        if pregnant is False or sex == "male":
            return
        if pregnant is None and not conflict:
            # Unknown pregnancy: only flag women of reproductive age, as a question to confirm.
            if sex != "female" or age_years is None or not REPRODUCTIVE_AGE[0] <= age_years <= REPRODUCTIVE_AGE[1]:
                return
        # Conflicting statements are flagged whatever the age: someone stated a pregnancy, so missing a
        # contraindication is worse than one more question.
        status = "met" if pregnant else "possible"
        label = "임신 여부(진료 메모와 질문이 서로 다름 — 확인 필요)" if conflict else "임신 여부(가임기 여성)"
        for item in prescribed:
            for rule in self._tables.pregnancy.get(item["ingredient_code"], ()):
                grade = rule.get("grade")
                alert = {"type": "임부금기", "status": status, "products": [self._product(item)],
                         "grade": _GRADE_LABELS.get(str(grade), grade or "미표기"), "detail": rule["detail"],
                         "notice": self._notice(rule)}
                if status == "possible":
                    alert["missing"] = [{"kind": "finding", "key": "pregnancy", "label": label}]
                yield alert

    def _elderly(self, prescribed: list[dict[str, Any]], age_years: float | None) -> Iterable[dict[str, Any]]:
        if age_years is None or age_years < ELDERLY_AGE:
            return
        for item in prescribed:
            for rule in self._tables.elderly.get(item["ingredient_code"], ()):
                yield {"type": "노인주의", "status": "met", "category": rule["category"], "products": [self._product(item)],
                       "detail": rule["detail"], "notice": self._notice(rule),
                       "note": "노인 기준 65세 이상은 DUR 운영 기준(보도 확인) — 공고 원문 대조 필요"}
