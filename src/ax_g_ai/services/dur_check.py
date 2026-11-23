"""환자의 처방을 HIRA DUR 목록(PostgreSQL 스키마 ``dur``)과 대조해 DUR 경고를 만든다.

설계상 위치: 서비스 계층의 DUR 점검 모듈. 런타임(``runtime.py``)이 ``DurChecker``를 만들고, 임상 추천
(``clinical_recommendation``)이 질문 속 약 이름 찾기(``find_drugs``)와 처방 점검(``check``)에 쓴다.
DUR 저장 계층(``dur_store.reader``)은 DB 행으로 ``DurTables``를 만든다.

역할: 활성 데이터셋은 메모리로 한 번 읽는다(``ax_g_ai.dur_store.reader``). 원천과 변환은
``ax_g_ai.dur_store.snapshot``을 참고한다.

처방은 EMR ``medCode`` = 9자리 보험 제품코드로만 맞춘다(이름 추측 없음). 맞지 않는 코드는 사이트가 고칠 수 있게
보고한다. 경고:

- 병용금기: 현재 처방 두 개의 성분이 목록의 쌍을 이룬다(목록의 비고 —
  예: "48시간 이내 병용금기", MTX 주간 용량 조건 — 는 그대로 보여 주며 평가하지 않는다).
- 연령금기: 환자 나이가 목록의 연령 조건에 해당한다(나이를 모르면 경고 없음).
- 임부금기: 임신 기록 → ``met``. 15–49세 여성의 임신 여부를 모름, 또는 메모와 질문의 임신 여부가 서로 다름
  (남성이 아니면 나이 무관) → ``possible``.
- 노인주의: 65세 이상.

채팅 질문에 나온 약(``find_drugs``)은 추가할 약으로서, 그 성분의 모든 제형 코드로 점검하며 ``proposed: true``가
붙는다. 단일 성분 코드만 이렇게 이름으로 찾는다. 제품 목록은 복합제를 하나의 성분 코드 아래 구성 성분마다 한 행으로
주므로, 행들이 둘 이상의 성분을 말하는 코드(또는 여러 성분을 나열한 이름)는 복합제이며 처방으로만 점검한다.
성분은 염·수화물 단어를 뺀 영문명이다("amlodipine besylate" → "amlodipine", "sodium alendronate" → "alendronate";
모두 염 단어인 "potassium chloride"는 그대로 둔다). 이 키, 전체 영문명, 또는 단일 성분 제품명 속 한글명
(염 접미어 제거, 예: 와파린나트륨 → 와파린)으로 찾으며, 한글명마다 그 이름을 가진 제품들의 코드에 잇는다.

용량 관련 DUR 목록(용량주의, 투여기간주의)은 아직 내려받은 파일에 없다.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ax_g_ai.services.clinical_recommendation.schema import (
    DurAlert,
    DurNotice,
    DurProduct,
    DurProposed,
    DurSource,
    DurUnmapped,
    MedicationAlerts,
    MissingInput,
    Status,
)
from ax_g_ai.services.term_matcher import TermMatcher

ELDERLY_AGE = 65
REPRODUCTIVE_AGE = (15, 49)
_WEEKS_PER_YEAR = 52.1775
_CONDITIONS = {"미만": lambda value, limit: value < limit, "이하": lambda value, limit: value <= limit,
               "이상": lambda value, limit: value >= limit, "초과": lambda value, limit: value > limit}
_GRADE_LABELS = {"1": "1등급", "2": "2등급"}
# 제품명 속 한글 성분명: "클로시드정250밀리그램(클래리트로마이신)_(0.25g/1정)". 뒤쪽 괄호에는 대신 수출명이나
# 포장이 들어 있을 수 있다: "클리신정(클로닉신리시네이트)(수출명:CLICIN Tab)_(…)",
# "트리세녹스주(삼산화비소)(바이알)_(…)". 이런 것은 건너뛴다.
_KO_PARENTHESES = re.compile(r"\(([^()]*[가-힣][^()]*)\)")
_KO_NOT_INGREDIENT = frozenset({
    "1회용", "일회용", "바이알", "앰플", "프리필드", "카트리지", "주사용", "수출용", "관납용", "군납용", "소아용", "병", "별첨",
    "유전자재조합", "재조합", "미분화", "단클론항체",
})
_KO_RECOMBINANT = re.compile(r",?유전자재조합$")  # "니볼루맙,유전자재조합": 복합제가 아니라 생물의약품
_KO_SUFFIXES = re.compile(
    r"(?:[\d.]+(?:밀리그람|밀리그램|마이크로그램|그램|mg|g)?|염산염|황산염|수화물|무수물|나트륨|칼륨|칼슘|마그네슘|베실산염|말레산염|타르타르산염"
    r"|숙신산염|메실산염|브롬화수소산염|인산염|아세트산염|푸마르산염|시트르산염|질산염|장용과립|제피과립|장용|서방)$"
)
_MIN_KO, _MIN_EN = 3, 4
# 짝이온·수화물·공형성물·입자 형태 단어: 성분을 찾을 때 뺀다("atorvastatin calcium" → "atorvastatin").
# 에스터(furoate, propionate, valerate)는 약의 일부이므로 남긴다. 이 단어들로만 된 이름
# ("sodium bicarbonate", "potassium chloride")은 그대로 둔다.
_SALT_WORDS = frozenset("""
    hydrochloride dihydrochloride hcl hydrobromide hydroiodide hydrate monohydrate dihydrate trihydrate hemihydrate
    sesquihydrate anhydrous sodium disodium monosodium trisodium potassium dipotassium calcium magnesium
    besylate besilate camsylate camsilate mesylate mesilate dimesylate maleate fumarate hemifumarate tartrate
    bitartrate succinate hemisuccinate citrate sulfate bisulfate hydrogen phosphate diphosphate dihydrogen acetate
    oxalate tosylate tosilate napadisilate napsylate xinafoate chloride bromide iodide bicarbonate carbonate
    hydroxide oxide nitrate lactate gluconate salt l-proline l-lysine lysine arginine l-arginine meglumine
    tromethamine erbumine granule granules micronized esilate esylate hemitartrate strontium tetrahydrate diluted solution
""".split())
# HIRA 목록이 두 가지로 표기한 한 성분(2026.6 목록에서 발견, 같은 유효 성분으로 검수함).
# 염 단어를 뺀 뒤 적용한다. 여기 없는 표기는 두 변형을 서로 다른 성분으로 만들 뿐이다:
# 둘을 함께 가진 코드는 그러면 복합제로 취급되어 처방으로만 점검된다.
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
    "estrogen": "conjugated estrogen",  # "estrogen(conjugated)": 목록에서 유일한 단순 "estrogen"
    "nanoliposomal irinotecan": "irinotecan",  # 같은 유효 성분. DUR 규칙은 어차피 코드별
}
# 여러 성분을 나열한 이름, 또는 목록이 복합제라 부르는 제품("복합체"는 착물로 한 약).
_COMBINATION = re.compile(r"[,+/·]|복합(?!체)|함유")
# 이름 속 함량·양("24mg", "0.16g", "1KI.U", "300unit"). 숫자로 시작하는 이름("5-fluorouracil")은 아니다.
_STRENGTH = re.compile(r"(?=[\d.,]*\d)[\d.,]+[a-zμ㎍㎎%.]*(?:/[\w.]*)?\)*", re.I)


class DurUnavailableError(RuntimeError):
    """DUR 데이터셋이 적재되지 않았거나 형태가 예상과 다르다.

    역할: DUR 저장 계층과 ``DurTables.from_rows``가 DUR 점검을 쓸 수 없음을 알리는 오류.
    """


Row = Mapping[str, Any]


@dataclass(frozen=True)
class DurTables:
    """DUR 데이터셋 하나를 제품코드·성분코드로 조회할 수 있게 색인한 표 묶음.

    역할: 제품(제품코드 → 행), 병용금기(성분 A → 규칙), 연령·임부·노인 규칙(성분코드 → 규칙)과 출처 정보를 담는다.
    """

    source: Mapping[str, str]
    products: Mapping[str, Row]
    pairs_by_a: Mapping[str, tuple[Row, ...]]
    ages: Mapping[str, tuple[Row, ...]]
    pregnancy: Mapping[str, tuple[Row, ...]]
    elderly: Mapping[str, tuple[Row, ...]]
    dataset_id: int | None = None
    # 규칙 목록에 적힌 (성분코드, 성분명): 제품 행에 없는 구성 성분명을 담을 수 있다.
    ingredient_names: tuple[tuple[str, str], ...] = ()

    @classmethod
    def from_rows(cls, source: Mapping[str, str], rows: Mapping[str, Iterable[Row]], *, dataset_id: int | None = None) -> DurTables:
        """표 이름별 DUR 행으로 색인된 ``DurTables``를 만든다.

        역할: ``product``는 제품코드로, ``pair_rule``은 ``ingredient_a``로, ``age_rule``·``pregnancy_rule``·``elderly_rule``은
        ``ingredient_code``로 묶는다. 규칙 목록의 (성분코드, 성분명)도 모은다(병용금기는 A·B 양쪽).
        입력:
            source: 데이터셋 출처 정보.
            rows: 표 이름 → 행들.
            dataset_id: DB의 데이터셋 ID.
        출력: ``DurTables``.
        예외: 제품 행이 없으면 ``DurUnavailableError``.
        """
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
    """질문에 나온 약 하나: 그 성분의 모든 단일 성분 DUR 코드(제형).

    역할: ``DurChecker.find_drugs``의 결과이자 ``check``의 ``proposed`` 입력.
    """

    name: str  # 질문에 적힌 그대로(한글 성분명 또는 영문명)
    ingredient_name: str
    codes: frozenset[str]
    # 성분 키(``_ingredient_key``. 목록이 한 약을 두 가지로 표기하면 두 개): 이미 처방된 약은
    # 처방으로 점검한다.
    keys: frozenset[str] = frozenset()


def _bare_words(name: str) -> list[str]:
    """괄호(중첩 포함)와 함량을 뺀 이름의 단어들(소문자). 구분자(",", "+", "/")는 남긴다."""
    text = name.lower()
    while (stripped := re.sub(r"\([^()]*\)", " ", text)) != text:
        text = stripped
    return [word for word in text.split() if not _STRENGTH.fullmatch(word)]


def _english_name(name: str) -> str:
    """'Amlodipine Besylate (as amlodipine 5mg)' → 'amlodipine besylate': 괄호·함량·여분 공백 없음(``ext``는 ``extract``로)."""
    words = [word.strip(".,()[]") for word in _bare_words(name)]
    return " ".join("extract" if word == "ext" else word for word in words if word)


def _is_combination_name(name: str) -> bool:
    """여러 성분을 나열한 이름인지: 'linagliptin, metformin hydrochloride'는 두 성분, 'plerixafor 24mg(20mg/mL)'은 한 성분."""
    return _COMBINATION.search(" ".join(_bare_words(name))) is not None


def _salt_free(name: str) -> str:
    """염·수화물 단어를 뺀 영문명. 남는 단어가 없으면 전체 이름."""
    full = _english_name(name)
    kept = [word for word in full.split() if word not in _SALT_WORDS]
    return " ".join(kept) if kept else full


def _ingredient_key(name: str) -> str:
    """영문명이 가리키는 성분 키(``_salt_free`` 후 ``_SAME_INGREDIENT``로 표기 통일)."""
    key = _salt_free(name)
    return _SAME_INGREDIENT.get(key, key)


def _distinct_ingredients(keys: Iterable[str]) -> frozenset[str]:
    """서로 다른 성분을 가리키는 키만 남긴다: 다른 키로 시작하는 키("clonixin lysinate", "clonixin")는 같은 성분으로 본다."""
    ordered = sorted(set(keys), key=len)
    kept: list[str] = []
    for key in ordered:
        if not any(key.startswith(shorter) for shorter in kept):
            kept.append(key)
    return frozenset(kept)


def _korean_ingredient(product_name: str) -> str | None:
    """제품명 속 한글 성분명(공백 제거, 마지막 해당 괄호). 없으면 ``None``."""
    head = product_name.split("_(", 1)[0]
    candidates = []
    for group in _KO_PARENTHESES.findall(head):
        name = _KO_RECOMBINANT.sub("", re.sub(r"\s+", "", group))
        if name and ":" not in name and name not in _KO_NOT_INGREDIENT:
            candidates.append(name)
    return candidates[-1] if candidates else None


def _ko_bases(name: str) -> set[str]:
    """한글명과, 끝의 함량·염·제형 접미어를 차례로 뗀 형태들 중 세 글자 이상인 것."""
    names = {name}
    while (stripped := _KO_SUFFIXES.sub("", name)) != name and stripped:
        name = stripped
        names.add(name)
    return {item for item in names if len(item) >= _MIN_KO}


class _DrugNames:
    """제품 목록으로 만든 한글·영문 성분명 → 성분 키 → 단일 성분 코드 색인.

    역할: 질문 텍스트에서 약 이름을 찾고(``find``), 성분코드의 구성 성분 키(``components``)를 알려 준다.
    """

    def __init__(self, products: Iterable[Row], rule_names: Iterable[tuple[str, str]] = ()) -> None:
        """제품 행과 규칙 목록 성분명으로 이름 색인을 만든다.

        역할: 성분코드마다 구성 성분 키를 정하고, 키가 하나가 아니거나 복합 이름이 있는 코드는 복합제로 빼고 나머지 코드의
        영문 별칭(키·전체 영문명·염 제거명)과 제품명 속 한글명(및 접미어 제거형)을 키에 잇는다. 키마다 대표 표기를 고르고,
        다른 키와 겹치지 않는 첫 단어도 별칭으로 더한 뒤 한글·영문 매처를 만든다.
        입력:
            products: DUR 제품 행.
            rule_names: 규칙 목록의 (성분코드, 성분명). 모든 제품 행이 한 성분만 말하는 복합제의 다른 구성 성분명을
                담을 수 있다.
        출력: 없음(색인 생성).
        """
        rows = list(products)
        raw_names: dict[str, set[str]] = defaultdict(set)
        for row in rows:
            raw_names[row["ingredient_code"]].add(str(row.get("ingredient_name") or "").strip())
        for code, name in rule_names:
            if code in raw_names and name:
                raw_names[code].add(name.strip())
        # 코드마다의 성분 키: 단일 성분은 하나, 복합제는 구성 성분마다 하나.
        self._components = {
            code: _distinct_ingredients(_ingredient_key(name) for name in names if _english_name(name))
            for code, names in raw_names.items()
        }
        combinations = {code for code, names in raw_names.items()
                        if len(self._components[code]) != 1 or any(_is_combination_name(name) for name in names)}
        codes: dict[str, set[str]] = defaultdict(set)
        labels: dict[str, str] = {}
        label_rank: dict[str, tuple[bool, int, str]] = {}
        english: dict[str, str] = {}  # 전체 영문명 또는 성분 키 → 성분 키
        korean: dict[str, set[str]] = defaultdict(set)  # 한글명(과 염을 뗀 형태) → 성분 키들
        for row in rows:
            code = row["ingredient_code"]
            if code in combinations:
                continue
            name = str(row.get("ingredient_name") or "").strip()
            key = _ingredient_key(name)
            if not key:  # 영문 성분명이 없음: 이름으로 찾을 수 없다
                continue
            codes[key].add(code)
            # 대표 표기는 통상 표기 먼저, 그다음 가장 짧은 형태: "aspirin(enteric coated) (as aspirin)"이 아니라 "aspirin".
            rank = (_salt_free(name) != key, len(name), name)
            if key not in labels or rank < label_rank[key]:
                labels[key], label_rank[key] = name, rank
            for alias in (key, _english_name(name), _salt_free(name)):  # 목록의 다른 표기("diphenylhydantoin")도
                english.setdefault(alias, key)
            korean_name = _korean_ingredient(str(row.get("product_name") or ""))
            # 한글 성분 여러 개를 나열한 제품명("A/B")은 단독 코드여도 복합제다.
            if korean_name and not _COMBINATION.search(korean_name):
                for base in _ko_bases(korean_name):
                    korean[base].add(key)
        # 첫 단어만("clobetasol propionate"의 "clobetasol")으로는 그 단어로 시작하는 다른 성분이 없을 때만 성분을
        # 가리킨다: "ethyl", "insulin"이나 염 단어는 여러 성분을 가리키므로 매칭하지 않는다.
        first_words: dict[str, set[str]] = defaultdict(set)
        for key in codes:
            first_words[key.split()[0]].add(key)
        for word, keys in first_words.items():
            if len(keys) == 1 and word not in _SALT_WORDS and word not in english:
                english[word] = next(iter(keys))
        self._codes = {key: frozenset(items) for key, items in codes.items()}
        self._labels = labels
        # 가장 왼쪽·가장 긴 비중첩 일치: "와파린나트륨"은 "와파린" + 조각이 아니라 한 약이다.
        self._ko_names = TermMatcher((name, key) for name, keys in korean.items() for key in sorted(keys))
        self._en_names = TermMatcher(
            ((alias, key) for alias, key in english.items() if len(alias) >= _MIN_EN and re.fullmatch(r"[a-z0-9][a-z0-9 -]*[a-z]", alias)),
            boundary="a-z",
        )

    def components(self, code: str) -> frozenset[str]:
        """성분코드의 성분 키들(복합제는 여러 개)."""
        return self._components.get(code, frozenset())

    def find(self, text: str) -> tuple[ProposedDrug, ...]:
        """``text``에 나온 약들을 찾는다.

        역할: 공백을 지운 텍스트에서 한글명을, 소문자 텍스트에서 영문명을 각각 가장 왼쪽·가장 긴 일치로 찾고, 이미 찾은
        성분 키와 겹치는 일치는 버린다. 목록이 영문으로 두 가지로 표기한 한글명("류프로렐린": leuprolide, leuprorelin)은
        두 키의 코드를 모두 가진 한 약이다. 대표 성분명은 키들의 대표 표기 중 가장 짧은 것이다.
        입력:
            text: 질문이나 메모 텍스트.
        출력: ``ProposedDrug`` 튜플(한글 일치 먼저, 그다음 영문 일치).
        """
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
    """DUR 데이터셋 하나로 처방·추가 예정 약을 점검하는 검사기.

    역할: ``DurTables``와 이름 색인을 가지고 질문 속 약을 찾으며(``find_drugs``), 병용·연령·임부·노인 경고를 만든다
    (``check``). 생성 후 바뀌지 않는다.
    """

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
        """``text``(채팅 질문이나 메모)에 나온 약을 DUR 성분코드에 맞춰 돌려준다."""
        return self._names.find(text) if text else ()

    def check(
        self, medications: Sequence[tuple[str | None, str | None]], *, age_years: float | None, sex: str | None,
        pregnant: bool | None, proposed: Sequence[ProposedDrug] = (), pregnancy_conflict: bool = False,
    ) -> MedicationAlerts:
        """payload ``assessment.medication_alerts``를 만든다.

        역할: 처방을 9자리 제품코드로 DUR 제품에 맞추고(못 맞추면 ``unmapped``), 질문 속 약 중 이미 처방된 성분(복합제
        구성 성분 포함)이 아닌 것을 모든 제형 코드로 추가한다. 병용·연령·임부·노인 규칙을 적용한 뒤 내용이 같은
        경고는 하나로 합친다.
        입력:
            medications: (EMR ``medCode``, ``medName``) 목록.
            age_years: 나이(세), 모르면 ``None``.
            sex: ``male``/``female``, 모르면 ``None``.
            pregnant: 임신 여부, 모르면 ``None``.
            proposed: 질문에서 찾은 추가 예정 약(``find_drugs`` 결과).
            pregnancy_conflict: 메모와 질문의 임신 여부가 서로 다름(이때 ``pregnant``는 ``None``).
        출력: ``MedicationAlerts``(경고, 매핑 안 된 처방, 출처, 인식한 추가 예정 약이 있으면 ``proposed``).
        """
        prescribed: list[dict[str, Any]] = []
        unmapped: list[DurUnmapped] = []
        for code, name in medications:
            row = self._tables.products.get(code.strip()) if code and code.strip() else None
            if row is None:
                unmapped.append(DurUnmapped(med_code=code, med_name=name, detail="DUR 목록에 없는 제품코드(9자리 보험코드) — DUR 점검 제외"))
                continue
            prescribed.append({"med_code": row["product_code"], "med_name": name or row["product_name"], "product_name": row["product_name"],
                               "ingredient_code": row["ingredient_code"], "ingredient_name": row["ingredient_name"]})
        # 질문에 나온 추가 예정 약: 모든 제형 코드로 처방처럼 점검한다. 이미 처방된 약은
        # 처방으로 한 번만 점검한다.
        # "이미 처방됨"은 성분 기준이다: 처방된 복합제의 구성 성분도 포함한다.
        current = {key for item in prescribed for key in self._names.components(item["ingredient_code"])}
        recognized: list[DurProposed] = []
        for drug in proposed:
            if current.intersection(drug.keys):
                continue
            current.update(drug.keys)
            recognized.append(DurProposed(name=drug.name, ingredient_name=drug.ingredient_name))
            prescribed.extend({"med_code": None, "med_name": drug.name, "product_name": None, "ingredient_code": code,
                               "ingredient_name": drug.ingredient_name, "proposed": True} for code in sorted(drug.codes))
        alerts = [
            *self._pairs(prescribed),
            *self._ages(prescribed, age_years),
            *self._pregnancy(prescribed, age_years, sex, pregnant, pregnancy_conflict),
            *self._elderly(prescribed, age_years),
        ]
        # 소견 하나에 경고 하나: 추가 예정 약의 제형들은 같은 규칙을 공유한다.
        unique = {json.dumps(alert.model_dump(mode="json"), ensure_ascii=False, sort_keys=True): alert for alert in alerts}
        return MedicationAlerts(
            alerts=list(unique.values()), unmapped=unmapped, source=DurSource.model_validate(self.source),
            **({"proposed": recognized} if recognized else {}),
        )

    @staticmethod
    def _product(item: Mapping[str, Any]) -> DurProduct:
        """점검 항목을 경고에 넣을 ``DurProduct``로 바꾼다(추가 예정 약이면 ``proposed=True``)."""
        return DurProduct(
            med_code=item["med_code"], med_name=item["med_name"], ingredient_name=item["ingredient_name"],
            **({"proposed": True} if item.get("proposed") else {}),
        )

    @staticmethod
    def _notice(rule: Row) -> DurNotice:
        """규칙의 고시 번호·일자."""
        return DurNotice(no=rule.get("notice_no"), date=rule.get("notice_date"))

    def _pairs(self, prescribed: list[dict[str, Any]]) -> Iterable[DurAlert]:
        """병용금기 경고를 만든다.

        역할: 점검 항목을 성분코드로 묶고, 성분 A의 규칙 중 상대 성분 B도 있는 쌍마다 경고를 낸다(같은 성분끼리 제외,
        같은 쌍·같은 내용은 한 번). 비고는 ``condition_note``로 그대로 싣는다.
        입력:
            prescribed: 점검 항목(처방 + 추가 예정 약).
        출력: ``DurAlert``(``병용금기``, ``met``)를 내는 생성기.
        """
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
                note: dict[str, Any] = {"condition_note": rule["note"]} if rule.get("note") else {}
                yield DurAlert(
                    type="병용금기", status="met",
                    products=[self._product(item) for item in (*by_ingredient[a], *by_ingredient[b])],
                    detail=rule["detail"], **note, notice=self._notice(rule),
                )

    def _ages(self, prescribed: list[dict[str, Any]], age_years: float | None) -> Iterable[DurAlert]:
        """연령금기 경고를 만든다.

        역할: 나이를 규칙 단위(세·개월·주)로 바꿔 조건(미만·이하·이상·초과)을 검사한다. 나이를 모르거나 단위·조건을
        모르는 규칙은 경고하지 않는다.
        입력:
            prescribed: 점검 항목.
            age_years: 나이(세).
        출력: ``DurAlert``(``연령금기``, ``met``)를 내는 생성기.
        """
        if age_years is None:
            return
        for item in prescribed:
            for rule in self._tables.ages.get(item["ingredient_code"], ()):
                unit, age = rule["unit"], float(rule["age"])
                value = age_years if unit == "세" else age_years * 12 if unit == "개월" else age_years * _WEEKS_PER_YEAR if unit == "주" else None
                test = _CONDITIONS.get(rule["condition"])
                if value is None or test is None or not test(value, age):
                    continue
                yield DurAlert(type="연령금기", status="met", products=[self._product(item)],
                               age_condition=f"{age:g}{unit} {rule['condition']}", detail=rule["detail"],
                               notice=self._notice(rule))

    def _pregnancy(
        self, prescribed: list[dict[str, Any]], age_years: float | None, sex: str | None, pregnant: bool | None,
        conflict: bool = False,
    ) -> Iterable[DurAlert]:
        """임부금기 경고를 만든다.

        역할: 임신이 아님이 확인됐거나 남성이면 경고하지 않는다. 임신 기록이 있으면 ``met``, 임신 여부를 모르면 가임기
        (15–49세) 여성일 때만, 진술이 서로 다르면 나이와 관계없이 ``possible``로 확인할 입력과 함께 낸다.
        입력:
            prescribed: 점검 항목.
            age_years: 나이(세).
            sex: ``male``/``female``.
            pregnant: 임신 여부.
            conflict: 메모와 질문의 임신 여부가 서로 다름.
        출력: ``DurAlert``(``임부금기``)를 내는 생성기.
        """
        if pregnant is False or sex == "male":
            return
        if pregnant is None and not conflict:
            # 임신 여부 모름: 가임기 여성만 확인 질문으로 표시한다.
            if sex != "female" or age_years is None or not REPRODUCTIVE_AGE[0] <= age_years <= REPRODUCTIVE_AGE[1]:
                return
        # 진술이 서로 다르면 나이와 관계없이 표시한다: 누군가 임신을 언급했으므로 금기를 놓치는 것이
        # 질문 하나 더 하는 것보다 나쁘다.
        status: Status = "met" if pregnant else "possible"
        label = "임신 여부(진료 메모와 질문이 서로 다름 — 확인 필요)" if conflict else "임신 여부(가임기 여성)"
        # ``possible`` 경고만 확인할 입력을 적는다.
        missing: dict[str, Any] = (
            {"missing": [MissingInput(kind="finding", key="pregnancy", label=label)]} if status == "possible" else {}
        )
        for item in prescribed:
            for rule in self._tables.pregnancy.get(item["ingredient_code"], ()):
                grade = rule.get("grade")
                yield DurAlert(type="임부금기", status=status, products=[self._product(item)],
                               grade=_GRADE_LABELS.get(str(grade), grade or "미표기"), detail=rule["detail"],
                               notice=self._notice(rule), **missing)

    def _elderly(self, prescribed: list[dict[str, Any]], age_years: float | None) -> Iterable[DurAlert]:
        """65세 이상이면 노인주의 경고를 만든다(나이를 모르면 없음)."""
        if age_years is None or age_years < ELDERLY_AGE:
            return
        for item in prescribed:
            for rule in self._tables.elderly.get(item["ingredient_code"], ()):
                yield DurAlert(type="노인주의", status="met", category=rule["category"], products=[self._product(item)],
                               detail=rule["detail"], notice=self._notice(rule),
                               note="노인 기준 65세 이상은 DUR 운영 기준(보도 확인) — 공고 원문 대조 필요")
