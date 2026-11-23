"""Bridge EMR payload를 말뭉치의 검사·약물 개념에 매핑하고 ``PatientFacts``를 만든다.

설계상 위치: 서비스 계층의 임상 추천 입력 변환 모듈. 임상 추천(``clinical_recommendation``)이 ``map_emr``로
환자 사실을 만들고, 말뭉치 적재(``clinical_recommendation/corpus.py``)가 ``ConceptCatalog``를, 런타임
(``runtime.py``)이 ``SiteCodeMap``을 읽어 들인다. 용량 단계는 ``EmrMapping``·``MedicationMatch``를 읽는다.

역할: 개념 카탈로그는 ``text_data/tools/build_corpus.py``가 만든다
(``corpus/test_concepts.jsonl``, ``drug_concepts.jsonl``, ``drug_classes.jsonl``).
병원 코드는 검수된 사이트 코드 맵(JSON, ``text_data/mappings/README.md`` 참고)으로 매핑한다.

안전 규칙(``clinical_calculations``와 같이 입력을 추정하지 않음):
- 검사값은 개념을 알고(사이트 맵 또는 LOINC ``examCode``) **그리고** 단위를 알며(EMR ``unit`` 또는 사이트 맵)
  환산할 수 있고, 환산값이 생리적으로 가능한 범위일 때만 규칙에 들어간다. 이름으로만 맞춘 것은 값이 아니라
  사이트 맵 후보 제안으로만 돌려준다.
- 약은 ``medName``의 성분명으로 맞출 수 있다(한국 제품명은 괄호 안에 성분명을 적는다). 매핑할 수 없는 처방이
  하나라도 있으면 복용 약 부재를 거짓이 아니라 알 수 없음(``medications_complete=False``)으로 둔다.
"""

from __future__ import annotations

import json
import math
import re
import unicodedata
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ax_g_ai.corpus.files import CONCEPT_FILES
from ax_g_ai.jsonl import read_jsonl
from ax_g_ai.services.clinical_calculations import ckd_epi_2021_egfr, cockcroft_gault_crcl
from ax_g_ai.services.clinical_corpus import PatientFacts
from ax_g_ai.services.term_matcher import TermMatcher

LOCAL_ROUTES = frozenset({"topical", "inhaled"})
_SALT_PREFIXES = ("염산", "황산", "인산", "질산", "초산", "주석산", "엘주석산", "타르타르산", "말레산", "말레인산", "푸마르산", "헤미푸마르산", "숙신산", "브롬화수소산", "메실산", "베실산", "수산", "건조")
_TOPICAL_FORM = re.compile(r"(크림|연고|겔|젤|로션|점안|안연고|외용|질정|질크림|질좌|점비|나잘|가글|cream|ointment|gel|lotion|ophthalmic|eye ?drop|vaginal|nasal)", re.I)
_INHALED_FORM = re.compile(r"(흡입|인헬러|에어로솔|디스커스|터부헬러|엘립타|inhal)", re.I)
_TRANSDERMAL_FORM = re.compile(r"(패취|패치|patch)", re.I)
_ORAL_FORM = re.compile(r"(정|캡슐|캅셀|시럽|과립|현탁|tab|cap|syrup)", re.I)
_PARENS = re.compile(r"\(([^()]*)\)")
_NUMBER = re.compile(r"^[+]?\d+(?:\.\d+)?$")


def normalize_name(name: str) -> str:
    """NFKC + 소문자로 바꾸고 영숫자·한글·``+``·``*`` 외의 문자(공백 포함)를 지운다."""
    return re.sub(r"[^0-9a-z가-힣+*]", "", unicodedata.normalize("NFKC", name).lower())


def normalize_unit(unit: str) -> str:
    """단위 비교용 정규화: NFKC, ``μ``/``µ`` → ``u``, 공백 제거, 소문자."""
    text = unicodedata.normalize("NFKC", unit).replace("μ", "u").replace("µ", "u")
    return re.sub(r"\s+", "", text).lower()


def normalize_kcd(code: str) -> str:
    """KCD 코드를 공백·점을 지운 대문자로 만들고, 4자 이상이면 세 번째 글자 뒤에 점을 넣는다(``E119`` → ``E11.9``)."""
    compact = re.sub(r"[\s.]", "", code).upper()
    return compact if len(compact) <= 3 else f"{compact[:3]}.{compact[3:]}"


class ConceptMapError(ValueError):
    """사이트 코드 맵이 카탈로그에 없는 개념을 가리킨다.

    역할: ``SiteCodeMap.from_dict``가 잘못된 항목을 모두 모아 한 번에 알리는 오류.
    """


@dataclass(frozen=True)
class ConceptCatalog:
    """말뭉치 개념 카탈로그(검사·성분·제품·약물 계열)와 그 조회 색인.

    역할: 검사 개념을 LOINC·별칭으로, 성분을 한글·영문 이름으로 찾게 하고, 말뭉치 약물 ID를 카탈로그 ID로 잇는다.
    생성 후 바뀌지 않는다.
    """

    tests: Mapping[str, Mapping[str, Any]]
    ingredients: Mapping[str, Mapping[str, Any]]
    products: Mapping[str, Mapping[str, Any]]
    classes: Mapping[str, Mapping[str, Any]]
    test_by_loinc: Mapping[str, str]
    test_by_alias: Mapping[str, str]
    ko_names: tuple[tuple[str, str], ...]      # (정규화한 한글명, 성분), 긴 이름 먼저
    # 영문명(일반명 ID, RxNorm 이름) → (카탈로그 위치, 성분): 결과가 카탈로그 순서를 유지한다.
    en_names: TermMatcher[tuple[int, str]]
    # 세 글자 이상 한글명. 정규화한 텍스트(질문·메모) 어디에서든 찾는다.
    ko_terms: TermMatcher[str]
    drug_by_corpus_id: Mapping[str, str]

    @classmethod
    def load(cls, corpus_dir: Path) -> ConceptCatalog:
        """``corpus_dir``의 개념 JSONL 파일(``CONCEPT_FILES``)로 카탈로그를 만든다."""
        return cls.from_rows(*(read_jsonl(corpus_dir / name) for name in CONCEPT_FILES))

    @classmethod
    def from_rows(cls, tests: Iterable[Mapping[str, Any]], drugs: Iterable[Mapping[str, Any]], classes: Iterable[Mapping[str, Any]]) -> ConceptCatalog:
        """개념 행들로 카탈로그와 색인을 만든다.

        역할: 검사는 LOINC 코드·별칭(ID 포함, 정규화)으로 색인하고(먼저 나온 것 우선), 약물 행을 성분·제품으로 나눈다.
        한글명 목록(긴 것 먼저), 영문명 매처(``a-z`` 경계), 세 글자 이상 한글명 매처, 말뭉치 ID(``corpus_ids``) → 카탈로그
        ID 표를 만든다.
        입력:
            tests: ``test_concepts.jsonl`` 행.
            drugs: ``drug_concepts.jsonl`` 행(``type``이 ``ingredient`` 또는 ``product``).
            classes: ``drug_classes.jsonl`` 행.
        출력: ``ConceptCatalog``.
        """
        test_map = {row["id"]: row for row in tests}
        by_loinc: dict[str, str] = {}
        by_alias: dict[str, str] = {}
        for cid, row in test_map.items():
            for code in row.get("loinc") or ():
                by_loinc.setdefault(code["code"], cid)
            for alias in [*(row.get("aliases") or ()), cid]:
                by_alias.setdefault(normalize_name(alias), cid)
        ingredients = {row["id"]: row for row in drugs if row.get("type") == "ingredient"}
        products = {row["id"]: row for row in drugs if row.get("type") == "product"}
        ko = sorted(((normalize_name(name), cid) for cid, row in ingredients.items() for name in row.get("ko") or ()), key=lambda item: -len(item[0]))
        en = TermMatcher(
            ((name.lower(), (position, cid))
             for position, (cid, row) in enumerate(ingredients.items())
             for name in {cid.replace("_", " "), *(item["name"] for item in row.get("rxnorm") or ())}),
            boundary="a-z",
        )
        ko_terms = TermMatcher((name, cid) for name, cid in ko if len(name) >= 3)
        drug_ids: dict[str, str] = {}
        for row in [*ingredients.values(), *products.values()]:
            for alias in row.get("corpus_ids") or (row["id"],):
                drug_ids.setdefault(alias, row["id"])
        return cls(test_map, ingredients, products, {row["id"]: row for row in classes},
                   by_loinc, by_alias, tuple(ko), en, ko_terms, drug_ids)

    def suggest_test(self, *names: object) -> str | None:
        """주어진 이름·코드 중 처음으로 검사 별칭과 맞는 개념 ID(사이트 맵 후보 제안용). 없으면 ``None``."""
        for name in names:
            if isinstance(name, str) and name.strip():
                found = self.test_by_alias.get(normalize_name(name))
                if found is not None:
                    return found
        return None

    def match_ingredients(self, med_name: str) -> tuple[tuple[str, ...], bool]:
        """``med_name``에 적힌 성분들과, 적힌 구성 성분이 모두 맞았는지 여부를 돌려준다.

        역할: 괄호 안 목록을 ``,·+/``와 "및"으로 나눠 조각마다 한글명(염 접두어 제거 후보 포함), 없으면 영문명으로 맞춘다.
        괄호가 없으면 괄호 밖 텍스트에서 영문 일반명을 모두 찾되, 성분 목록이 없어 다른 성분을 놓칠 수 있으므로
        완전하지 않음(``False``)으로 둔다.
        입력:
            med_name: EMR 처방명(``medName``).
        출력: (성분 ID 튜플(중복 없이 찾은 순서), 모든 조각이 맞았는지).
        """
        segments = _PARENS.findall(med_name)
        parts = [part for segment in segments for part in re.split(r"[,·+/]|및", segment) if part.strip()]
        found: list[str] = []
        complete = bool(parts)
        for part in parts:
            ingredient = self._match_korean(normalize_name(part))
            if ingredient is None:
                ingredient = self._match_english(part)
            if ingredient is None:
                complete = False
            elif ingredient not in found:
                found.append(ingredient)
        if not parts:  # 예: "Metformin 500mg tab" — 영문 일반명만 있음
            outside = _PARENS.sub(" ", med_name)
            found = [cid for cid in dict.fromkeys(self._english_all(outside))]
            complete = False  # 성분 목록이 없는 이름은 다른 구성 성분을 숨길 수 있음
        return tuple(found), complete

    def _match_korean(self, part: str) -> str | None:
        """정규화한 조각(그리고 염 접두어를 뗀 형태)이 어떤 한글 성분명으로 시작하면 그 성분(긴 이름 우선)."""
        candidates = [part] + [part[len(prefix):] for prefix in _SALT_PREFIXES if part.startswith(prefix)]
        for candidate in candidates:
            for name, cid in self.ko_names:
                if name and candidate.startswith(name):
                    return cid
        return None

    def _match_english(self, part: str) -> str | None:
        """조각에서 찾은 첫 영문 성분(카탈로그 순서). 없으면 ``None``."""
        return next(iter(self._english_all(part)), None)

    def _english_all(self, text: str) -> list[str]:
        """``text``에 영문명이 온전한 단어로 나오는 성분들(카탈로그 순서)."""
        lowered = unicodedata.normalize("NFKC", text).lower()
        found = {value for match in self.en_names.matches(lowered) for value in match.values}
        return [cid for _position, cid in sorted(found)]

    def korean_named(self, normalized_text: str) -> set[str]:
        """``normalize_name``으로 정규화한 텍스트에 한글명(세 글자 이상)이 나오는 성분들."""
        return {cid for match in self.ko_terms.matches(normalized_text) for cid in match.values}


@dataclass(frozen=True)
class SiteLab:
    """사이트 맵의 검사 코드 항목: 개념 ID와 (EMR에 단위가 없을 때 쓰는) 기본 단위.

    역할: ``SiteCodeMap.labs``의 값.
    """

    concept: str
    unit: str | None = None


@dataclass(frozen=True)
class SiteMed:
    """사이트 맵의 처방 코드 항목: 성분들, 투여 경로, 명시한 약물 계열.

    역할: ``SiteCodeMap.meds``의 값. 이 항목으로 맞춘 처방은 성분이 완전한 것으로 본다.
    """

    ingredients: tuple[str, ...]
    route: str | None = None
    classes: tuple[str, ...] = ()


@dataclass(frozen=True)
class SiteCodeMap:
    """검수된 병원 코드 → 개념 맵.

    역할: 검사 코드와 처방 코드를 카탈로그 개념으로 잇는다. 카탈로그에 없는 성분 ID도 허용하며(카탈로그 밖의 알려진
    성분), 그런 처방도 복용 약 부재를 판단할 수 있게 유지한다.
    """

    labs: Mapping[str, SiteLab] = field(default_factory=dict)
    meds: Mapping[str, SiteMed] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path, catalog: ConceptCatalog) -> SiteCodeMap:
        """JSON 파일을 읽어 ``from_dict``로 검증·생성한다."""
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls.from_dict(data, catalog)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], catalog: ConceptCatalog) -> SiteCodeMap:
        """사이트 맵 사전을 카탈로그에 대해 검증하고 맵을 만든다.

        역할: 검사 항목은 개념이 카탈로그에 있어야 한다. 처방 항목은 ``product``가 있으면 카탈로그 제품이어야 하며
        그 구성 성분을 ``ingredients``에 더하고, 성분이 하나 이상 있어야 하며 ``classes``는 모두 카탈로그 계열이어야 한다.
        오류는 모두 모아 한 번에 알린다.
        입력:
            data: ``labs``·``meds`` 목록을 가진 사이트 맵 사전.
            catalog: 검증 기준 개념 카탈로그.
        출력: ``SiteCodeMap``.
        예외: 잘못된 항목이 하나라도 있으면 ``ConceptMapError``.
        """
        errors: list[str] = []
        labs: dict[str, SiteLab] = {}
        for item in data.get("labs") or ():
            concept = item.get("concept")
            if concept not in catalog.tests:
                errors.append(f"labs {item.get('code')!r}: unknown concept {concept!r}")
                continue
            labs[str(item["code"])] = SiteLab(concept, item.get("unit"))
        meds: dict[str, SiteMed] = {}
        for item in data.get("meds") or ():
            ingredients = list(item.get("ingredients") or ())
            product = item.get("product")
            if product is not None:
                if product not in catalog.products:
                    errors.append(f"meds {item.get('code')!r}: unknown product {product!r}")
                    continue
                ingredients += catalog.products[product]["components"]
            unknown_classes = [name for name in item.get("classes") or () if name not in catalog.classes]
            if not ingredients or unknown_classes:
                errors.append(f"meds {item.get('code')!r}: needs ingredients/product; unknown classes {unknown_classes}")
                continue
            meds[str(item["code"])] = SiteMed(tuple(dict.fromkeys(ingredients)), item.get("route"), tuple(item.get("classes") or ()))
        if errors:
            raise ConceptMapError("; ".join(errors))
        return cls(labs, meds)


@dataclass(frozen=True)
class LabValue:
    """규칙에 들어갈 검사값 하나(기준 단위로 환산한 값과 원래 값·단위·출처).

    역할: ``map_emr``가 개념마다 최신 값 하나를 고르는 단위. ``matched_by``는 값의 출처다.
    """

    concept: str
    value: float
    unit: str
    raw_value: object
    raw_unit: str
    test_date: str | None
    exam_code: str | None
    matched_by: str  # site_map | loinc | emr_vital | emr_context | derived(계산값)


@dataclass(frozen=True)
class MedicationMatch:
    """EMR 처방 하나를 성분·경로·계열로 매핑한 결과.

    역할: 복용 약 사실(``PatientFacts``)과 용량 단계의 입력. ``complete``는 처방의 모든 구성 성분이 확인됐는지다.
    """

    med_code: str | None
    med_name: str | None
    ingredients: tuple[str, ...]
    route: str | None
    classes: tuple[str, ...]
    matched_by: str  # site_map | name(이름 매칭)
    complete: bool
    # EMR의 dosage/unit/frequency/interval(및 durationDays) 원값(식별자 없음) — 용량 단계만 읽는다.
    dose: Mapping[str, Any] | None = None


_DOSE_FIELDS = ("dosage", "unit", "frequency", "interval", "durationDays")


@dataclass(frozen=True)
class MappingIssue:
    """매핑하지 못했거나 버린 EMR 항목에 대한 알림.

    역할: ``kind``(예: ``lab_unmapped``, ``med_partial``)와 한국어 설명, 가능하면 사이트 맵 후보(``suggestion``)를 담는다.
    """

    kind: str
    code: str | None
    name: str | None
    detail: str
    suggestion: str | None = None


@dataclass(frozen=True)
class EmrMapping:
    """``map_emr``의 결과: 환자 사실, 채택한 검사값, 처방 매핑, 매핑 문제.

    역할: 임상 추천의 규칙 평가·용량 평가·표시가 함께 읽는 EMR 변환 결과.
    """

    facts: PatientFacts
    labs: tuple[LabValue, ...]
    medications: tuple[MedicationMatch, ...]
    issues: tuple[MappingIssue, ...]


def map_emr(source: Mapping[str, Any], catalog: ConceptCatalog, site_map: SiteCodeMap | None = None) -> EmrMapping:
    """Bridge EMR payload를 개념에 매핑해 ``PatientFacts``와 매핑 결과를 만든다.

    역할: 검사(``labResultList``)·활력징후 목록의 값을 개념마다 가장 최근 것(``testDate``가 같거나 늦으면 교체) 하나로
    모은다. ``calculationContext``의 체중·혈청 크레아티닌은 같은 개념의 검사값이 없을 때만 쓰고, 그 뒤 eGFR·CrCl을
    계산값으로 더한다. 값은 개념의 ``corpus_ids``(없으면 개념 ID) 이름으로, 나이는 ``age``로 넣는다. 처방은 성분·계열
    사실로, 진단(``diagnosisList``)은 정규화한 KCD 코드 집합으로 바꾼다(목록이 없으면 ``None`` = 알 수 없음).
    입력:
        source: Bridge EMR payload.
        catalog: 개념 카탈로그.
        site_map: 사이트 코드 맵(없으면 빈 맵).
    출력: ``EmrMapping``(환자 사실, 채택한 검사값, 처방 매핑, 매핑 문제).
    """
    site = site_map or SiteCodeMap()
    issues: list[MappingIssue] = []
    latest: dict[str, LabValue] = {}

    def keep(value: LabValue) -> None:
        previous = latest.get(value.concept)
        if previous is None or (value.test_date or "") >= (previous.test_date or ""):
            latest[value.concept] = value

    for item in _items(source.get("labResultList")):
        result = _map_lab(item, catalog, site, issues)
        if result is not None:
            keep(result)
    for value in _vital_values(source, catalog, issues):
        keep(value)

    raw_context = source.get("calculationContext")
    context: Mapping[str, Any] = raw_context if isinstance(raw_context, Mapping) else {}
    age = _bounded(context.get("age_years"), 0, 130)
    weight = _bounded(context.get("weight_kg"), 0.3, 500)
    sex = context.get("sex") if context.get("sex") in {"male", "female"} else None
    values: dict[str, float] = {}
    if age is not None:
        values["age"] = age
    if weight is not None and "weight_kg" not in latest:
        keep(LabValue("weight_kg", weight, "kg", weight, "kg", None, None, "emr_vital"))
    # Bridge가 calculationContext에 넣은 크레아티닌(검사 행 없음): 채팅 계산과 같은 단위(mg/dL).
    creatinine = _bounded(context.get("serum_creatinine_mg_dl"), 0.01, 30)
    if creatinine is not None and "SCr" not in latest:
        keep(LabValue("SCr", creatinine, "mg/dL", creatinine, "mg/dL", None, None, "emr_context"))
    for derived in _derived(latest, age, sex):
        keep(derived)
    for concept, lab in latest.items():
        for corpus_id in (catalog.tests.get(concept) or {}).get("corpus_ids") or (concept,):
            values[corpus_id] = lab.value

    medications, meds_complete = _map_medications(source, catalog, site, issues)
    med_ids, med_classes, uncertain = _medication_facts(medications, catalog)
    diagnoses = source.get("diagnosisList")
    codes = None
    if isinstance(diagnoses, list):
        codes = frozenset(
            normalize_kcd(code) for item in _items(diagnoses)
            if isinstance(code := item.get("diagnosisCode", item.get("kcdCode")), str) and code.strip()
        )
    facts = PatientFacts(
        values=values, diagnosis_codes=codes,
        medications=None if medications is None else frozenset(med_ids),
        medication_classes=None if medications is None else frozenset(med_classes),
        medications_complete=meds_complete, uncertain_classes=frozenset(uncertain),
    )
    return EmrMapping(facts, tuple(latest.values()), tuple(medications or ()), tuple(issues))


def _items(value: object) -> list[Mapping[str, Any]]:
    """목록이면 그 안의 사전 항목만, 아니면 빈 목록."""
    return [item for item in value if isinstance(item, Mapping)] if isinstance(value, list) else []


def _text(value: object) -> str | None:
    """앞뒤 공백을 뗀 비어 있지 않은 문자열, 아니면 ``None``."""
    return value.strip() if isinstance(value, str) and value.strip() else None


def _bounded(value: object, low: float, high: float) -> float | None:
    """유한한 숫자(bool 제외)이고 [low, high] 안이면 float, 아니면 ``None``."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) and low <= number <= high else None


def _number(raw: object) -> float | None:
    """검사 결과 원값을 숫자로 읽는다.

    역할: 유한한 숫자, 또는 쉼표를 뺀 뒤 (``+`` 부호를 허용하는) 정수·소수 문자열만 받아들인다. bool은 거부한다.
    입력:
        raw: EMR 결과 값.
    출력: float, 읽을 수 없으면 ``None``(예: "음성", "<5").
    """
    if isinstance(raw, bool):
        return None
    if isinstance(raw, (int, float)):
        return float(raw) if math.isfinite(raw) else None
    if isinstance(raw, str):
        text = raw.strip().replace(",", "")
        return float(text) if _NUMBER.match(text) else None
    return None


def _convert(concept: Mapping[str, Any], value: float, unit: str) -> float | None:
    """값을 개념의 기준 단위로 환산한다.

    역할: 단위가 기준 단위와 같으면 그대로, 개념의 ``convert`` 표에 있으면 ``값 × factor``(사전이면 ``+ offset``)로 바꾼다.
    단위 비교는 ``normalize_unit``으로 한다.
    입력:
        concept: 검사 개념 행(``unit``·``convert``).
        value: 원래 값.
        unit: 원래 단위.
    출력: 환산값. 기준 단위가 없거나 환산 규칙이 없으면 ``None``.
    """
    canonical = concept.get("unit")
    if canonical is None:
        return None
    if normalize_unit(unit) == normalize_unit(canonical):
        return value
    for source_unit, rule in (concept.get("convert") or {}).items():
        if normalize_unit(source_unit) == normalize_unit(unit):
            if isinstance(rule, Mapping):
                return value * rule["factor"] + rule.get("offset", 0)
            return value * rule
    return None


def _checked_value(concept_id: str, raw: object, unit: str, catalog: ConceptCatalog, issues: list[MappingIssue], code: str | None, name: str | None) -> float | None:
    """원값을 숫자로 읽고 기준 단위로 환산한 뒤 허용 범위를 검사한다.

    역할: 수치가 아님·환산 불가·범위(개념 ``range``, 양 끝 포함) 밖이면 ``issues``에 알림을 더하고 값을 버린다.
    입력:
        concept_id: 검사 개념 ID.
        raw: EMR 결과 원값.
        unit: 결과 단위.
        catalog: 개념 카탈로그.
        issues: 매핑 문제를 쌓는 목록(이 함수가 추가한다).
        code: 알림에 적을 코드.
        name: 알림에 적을 이름.
    출력: 소수 4자리로 반올림한 환산값, 버리면 ``None``.
    """
    concept = catalog.tests[concept_id]
    number = _number(raw)
    if number is None:
        issues.append(MappingIssue("lab_non_numeric", code, name, f"{concept_id}: 수치가 아닌 결과 {raw!r}는 규칙 값으로 쓰지 않음"))
        return None
    converted = _convert(concept, number, unit)
    if converted is None:
        issues.append(MappingIssue("lab_unit_unsupported", code, name, f"{concept_id}: 단위 {unit!r}를 기준 단위 {concept.get('unit')!r}로 환산할 수 없음"))
        return None
    bounds = concept.get("range")
    if bounds and not bounds[0] <= converted <= bounds[1]:
        issues.append(MappingIssue("lab_out_of_range", code, name, f"{concept_id}: {converted:g} {concept.get('unit')}는 허용 범위 {bounds} 밖 — 입력·단위 오류 의심"))
        return None
    return round(converted, 4)


def _map_lab(item: Mapping[str, Any], catalog: ConceptCatalog, site: SiteCodeMap, issues: list[MappingIssue]) -> LabValue | None:
    """검사 결과 행 하나를 ``LabValue``로 매핑한다.

    역할: ``examCode``를 사이트 맵(단위는 EMR ``unit`` 우선, 없으면 사이트 맵 단위), 없으면 LOINC 코드(EMR ``unit``만)로
    개념에 잇는다. 매핑되지 않은 코드는 이름 기반 후보 제안과 함께 알리고, 단일 검사 항목이 아닌 개념(묶음·영상)과
    단위를 모르는 결과도 알리고 버린다.
    입력:
        item: ``labResultList`` 항목.
        catalog: 개념 카탈로그.
        site: 사이트 코드 맵.
        issues: 매핑 문제를 쌓는 목록(이 함수가 추가한다).
    출력: ``LabValue``, 쓸 수 없으면 ``None``.
    """
    code, name = _text(item.get("examCode")), _text(item.get("itemName"))
    emr_unit = _text(item.get("unit"))
    entry = site.labs.get(code) if code else None
    if entry is not None:
        concept, unit, matched_by = entry.concept, emr_unit or entry.unit, "site_map"
    elif code and code in catalog.test_by_loinc:
        concept, unit, matched_by = catalog.test_by_loinc[code], emr_unit, "loinc"
    else:
        issues.append(MappingIssue("lab_unmapped", code, name, "사이트 매핑에 없는 검사 코드", catalog.suggest_test(name, code)))
        return None
    if catalog.tests[concept].get("section") != "labs":
        issues.append(MappingIssue("lab_not_single", code, name, f"{concept}는 단일 검사 항목이 아님(묶음·영상) — 구성 항목 코드로 매핑"))
        return None
    if unit is None:
        issues.append(MappingIssue("lab_unit_unknown", code, name, f"{concept}: 결과 단위를 알 수 없음 — EMR unit 또는 사이트 매핑 unit 필요"))
        return None
    value = _checked_value(concept, item.get("value"), unit, catalog, issues, code, name)
    if value is None:
        return None
    return LabValue(concept, value, catalog.tests[concept]["unit"], item.get("value"), unit, _text(item.get("testDate")), code, matched_by)


def _vital_values(source: Mapping[str, Any], catalog: ConceptCatalog, issues: list[MappingIssue]) -> list[LabValue]:
    """Bridge 활력징후 목록(혈당·HbA1c·혈압·SpO2)을 ``LabValue``로 바꾼다.

    역할: Bridge 활력징후 목록은 단위가 고정되어 있다(``main_bridge`` 참고). 카탈로그에 없는 개념과 값이 비어 있는 행은
    건너뛰고, 값은 ``_checked_value``로 검사한다. ``exam_code``에는 목록 이름을 넣는다.
    입력:
        source: Bridge EMR payload.
        catalog: 개념 카탈로그.
        issues: 매핑 문제를 쌓는 목록(이 함수가 추가한다).
    출력: ``LabValue`` 목록(``matched_by="emr_vital"``).
    """
    result: list[LabValue] = []
    for list_key, value_key, date_key, concept, unit in (
        ("bloodSugarList", "glucoseValue", "measuredAt", "Glucose", "mg/dL"),
        ("glycatedHemoglobinList", "hba1cValue", "testDate", "HbA1c", "%"),
        ("bloodPressureList", "sbp", "measuredAt", "SBP", "mmHg"),
        ("bloodPressureList", "dbp", "measuredAt", "DBP", "mmHg"),
        ("oxygenSaturationList", "spo2Value", "measuredAt", "SpO2", "%"),
    ):
        if concept not in catalog.tests:
            continue
        for item in _items(source.get(list_key)):
            if item.get(value_key) is None:
                continue  # 예: 이완기 값만 있는 혈압 행
            value = _checked_value(concept, item.get(value_key), unit, catalog, issues, list_key, value_key)
            if value is not None:
                result.append(LabValue(concept, value, unit, item.get(value_key), unit, _text(item.get(date_key)), list_key, "emr_vital"))
    return result


def _derived(latest: Mapping[str, LabValue], age: float | None, sex: str | None) -> list[LabValue]:
    """최신 크레아티닌으로 eGFR(CKD-EPI 2021)과 Cockcroft-Gault CrCl을 계산한다.

    역할: 검사실이 그 값(eGFR, CrCl 각각)을 보고하지 않았고 모든 입력이 있을 때만 계산한다. 나이·성별·크레아티닌이
    없거나 18세 미만이면 아무것도 계산하지 않으며, CrCl은 체중도 있어야 한다. 검사일은 크레아티닌의 것을 쓰고,
    ``raw_unit``에는 계산 방법 이름을 넣는다.
    입력:
        latest: 개념 ID → 최신 ``LabValue``.
        age: 나이(세).
        sex: ``male``/``female``.
    출력: 계산한 ``LabValue`` 목록(``matched_by="derived"``).
    """
    result: list[LabValue] = []
    scr = latest.get("SCr")
    if scr is None or age is None or sex is None or age < 18:
        return result
    if "eGFR" not in latest:
        egfr = round(ckd_epi_2021_egfr(age, sex, scr.value), 1)
        result.append(LabValue("eGFR", egfr, "mL/min/1.73m2", None, "CKD-EPI 2021", scr.test_date, None, "derived"))
    weight = latest.get("weight_kg")
    if weight is not None and "CrCl" not in latest:
        # 채팅 계산과 같은 Cockcroft-Gault 함수(실제 체중).
        crcl = cockcroft_gault_crcl(age, sex, weight.value, scr.value)
        result.append(LabValue("CrCl", round(crcl, 1), "mL/min", None, "Cockcroft-Gault(실제 체중)", scr.test_date, None, "derived"))
    return result


def _route(med_name: str | None, declared: str | None) -> str | None:
    """처방의 투여 경로를 정한다.

    역할: 명시한 경로가 있으면 그대로 쓰고, 없으면 괄호 밖 처방명의 제형 단어로 흡입 → 국소 → 경피 → 경구 순서로 추정한다.
    입력:
        med_name: 처방명.
        declared: 사이트 맵에 적힌 경로.
    출력: ``inhaled``·``topical``·``transdermal``·``oral`` 또는 명시 경로, 알 수 없으면 ``None``.
    """
    if declared is not None:
        return declared
    if not med_name:
        return None
    head = _PARENS.sub(" ", med_name)
    if _INHALED_FORM.search(head):
        return "inhaled"
    if _TOPICAL_FORM.search(head):
        return "topical"
    if _TRANSDERMAL_FORM.search(head):
        return "transdermal"
    if _ORAL_FORM.search(head):
        return "oral"
    return None


def _map_medications(source: Mapping[str, Any], catalog: ConceptCatalog, site: SiteCodeMap, issues: list[MappingIssue]) -> tuple[list[MedicationMatch] | None, bool]:
    """처방 목록(``medicineDtailList``)을 ``MedicationMatch``로 매핑한다.

    역할: ``medCode``가 사이트 맵에 있으면 그 성분·경로·계열을, 없으면 처방명의 성분명 매칭을 쓴다. 성분을 찾지 못한 처방은
    빼고, 일부 성분만 찾은 처방은 남기되 둘 다 알리고 목록 전체를 완전하지 않음으로 둔다. 용량 필드는 문자열·숫자 값만 남긴다.
    입력:
        source: Bridge EMR payload.
        catalog: 개념 카탈로그.
        site: 사이트 코드 맵.
        issues: 매핑 문제를 쌓는 목록(이 함수가 추가한다).
    출력: (처방 매핑 목록, 전체가 완전히 매핑됐는지). 처방 목록 자체가 없으면 (``None``, ``False``).
    """
    raw = source.get("medicineDtailList")
    if not isinstance(raw, list):
        return None, False
    complete = True
    result: list[MedicationMatch] = []
    for item in _items(raw):
        code, name = _text(item.get("medCode")), _text(item.get("medName"))
        entry = site.meds.get(code) if code else None
        if entry is not None:
            ingredients, matched_by, item_complete = entry.ingredients, "site_map", True
            route = _route(name, entry.route)
            asserted = entry.classes
        else:
            ingredients, item_complete = catalog.match_ingredients(name or "")
            matched_by, route, asserted = "name", _route(name, None), ()
        if not ingredients:
            complete = False
            issues.append(MappingIssue("med_unmapped", code, name, "성분을 확인할 수 없는 처방 — 사이트 매핑 필요"))
            continue
        if not item_complete:
            complete = False
            issues.append(MappingIssue("med_partial", code, name, f"일부 성분만 확인됨: {', '.join(ingredients)}"))
        dose = {key: item[key] for key in _DOSE_FIELDS if isinstance(item.get(key), (str, int, float)) and not isinstance(item.get(key), bool)}
        result.append(MedicationMatch(code, name, tuple(ingredients), route, tuple(asserted), matched_by, item_complete, dose or None))
    return result, complete


def _medication_facts(medications: list[MedicationMatch] | None, catalog: ConceptCatalog) -> tuple[set[str], set[str], set[str]]:
    """처방 매핑에서 복용 약 ID, 해당하는 약물 계열, 판단할 수 없는 계열을 만든다.

    역할: 카탈로그 성분마다 말뭉치 ID와 계열을 더한다. 경로로 정하는 계열(``derive_from``)은 경로를 모르면 불확실,
    경로가 계열 ``routes``에 들면 해당으로 본다. ``derivable``이 거짓인 계열은 성분에서 정하지 않고, ``systemic_only``
    계열은 국소·흡입 경로면 제외한다. 구성 성분을 모두 가진 제품의 ID·계열과 사이트 맵이 명시한 계열도 더한다.
    끝으로 성분에서 정할 수 없는 계열, 또는 카탈로그 밖 성분이 있을 때 ``exhaustive``도 ``derive_from``도 아닌 계열은
    해당으로 확인되지 않았으면 불확실로 둔다.
    입력:
        medications: 처방 매핑 목록(``None``이면 처방 정보 없음).
        catalog: 개념 카탈로그.
    출력: (복용 약 말뭉치 ID 집합, 해당 계열의 말뭉치 별칭 집합, 불확실 계열의 말뭉치 별칭 집합(해당 계열 제외)).
    """
    if medications is None:
        return set(), set(), set()
    med_ids: set[str] = set()
    present: set[str] = set()
    uncertain: set[str] = set()
    has_other = False
    for med in medications:
        known = [ingredient for ingredient in med.ingredients if ingredient in catalog.ingredients]
        has_other |= len(known) < len(med.ingredients)
        for ingredient in known:
            row = catalog.ingredients[ingredient]
            med_ids.update(row.get("corpus_ids") or (ingredient,))
            for class_id in row.get("classes") or ():
                klass = catalog.classes.get(class_id, {})
                if klass.get("derive_from"):
                    if med.route is None:
                        uncertain.add(class_id)
                    elif med.route in (klass.get("routes") or ()):
                        present.add(class_id)
                    continue
                if klass.get("derivable") is False:
                    continue
                if klass.get("systemic_only") and med.route in LOCAL_ROUTES:
                    continue
                present.add(class_id)
        components = set(med.ingredients)
        for product_id, product in catalog.products.items():
            if set(product["components"]) <= components:
                med_ids.update(product.get("corpus_ids") or (product_id,))
                present.update(product.get("classes") or ())
        present.update(med.classes)
    for class_id, klass in catalog.classes.items():
        undecidable = klass.get("derivable") is False or (has_other and not klass.get("exhaustive") and not klass.get("derive_from"))
        if undecidable and class_id not in present:
            uncertain.add(class_id)
    present_aliases = {alias for class_id in present for alias in catalog.classes.get(class_id, {}).get("corpus_ids") or (class_id,)}
    uncertain_aliases = {alias for class_id in uncertain - present for alias in catalog.classes.get(class_id, {}).get("corpus_ids") or (class_id,)}
    return med_ids, present_aliases, uncertain_aliases
