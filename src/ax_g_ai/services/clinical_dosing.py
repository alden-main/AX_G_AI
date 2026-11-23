"""3단계(용량) 보조 도구: 말뭉치 ``dose_rules``의 결정적 용량 산술(LLM·``eval`` 없음).

설계상 위치: 서비스 계층의 임상 추천 3단계(용량 평가) 산술 모듈. ``clinical_recommendation/doses.py``가
공식 계산·신기능 구간 선택·현재 처방 1일량 계산에 쓴다.

역할:
- ``evaluate_formula``: ``dose_rules.calculation``의 작은 공식 언어를 계산한다 —
  숫자, 변수, ``+ - × * /``(``÷``도 허용), 괄호, 단항 ``-``, ``min``/``max``와 범위 리터럴
  ``[low, high]``(예: ``weight_kg × [0.1, 0.2]``). 모르는 문법은 ``DoseFormulaError``,
  환자 값이 없는 변수는 ``MissingInput``을 일으킨다.
- ``select_band``: 값이 들어가는 신기능 구간(``range.min``/``range.max``)을 고른다.
- ``prescribed_daily_mg``: EMR 처방의 현재 1일량(mg). 단위가 질량 단위이고 투여 간격이 1일 기준
  (또는 미기재)일 때만 계산한다(자유 텍스트에서 추측하지 않음).

결과는 검토 보조값이다. 항상 사용한 입력값·공식과 함께 전달되며, 추천기는 이를 의료진이 확인해야 하는
참고값으로 표시한다.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ax_g_ai.services.clinical_corpus import CorpusRuleError

Value = float | tuple[float, float]

_TOKEN = re.compile(r"\s*(?:(?P<number>\d+(?:\.\d+)?)|(?P<name>[A-Za-z_][A-Za-z0-9_]*)|(?P<op>[-+*/×÷(),\[\]]))")
_FUNCTIONS = {"min": min, "max": max}
# 레코드에 단위가 없을 때 계산 키의 접미사로 추정하는 단위(bag1_mg → mg).
_KEY_UNITS = (("_mcg_day", "mcg/day"), ("_mg_day", "mg/day"), ("_u_per_day", "U/day"), ("_mcg", "mcg"), ("_mg", "mg"),
              ("_g", "g"), ("_ml", "mL"), ("_u", "U"))
_MASS_UNITS = {"mg": 1.0, "밀리그램": 1.0, "g": 1000.0, "그램": 1000.0, "mcg": 0.001, "ug": 0.001, "µg": 0.001, "μg": 0.001,
               "마이크로그램": 0.001}
_DAILY_INTERVALS = {"1일", "일", "매일", "하루", "1day", "day", "daily", "qd", "/일", "per day", "perday", "d"}


class DoseFormulaError(CorpusRuleError):
    """용량 공식이 이 계산기가 지원하지 않는 문법을 쓴다.

    역할: 말뭉치 규칙 오류(``CorpusRuleError``)의 한 종류로, 공식 해석·계산 실패(0으로 나누기, 무한값 포함)를 알린다.
    """


class MissingInput(LookupError):
    """공식이 읽는 변수에 환자 값이 없다.

    역할: ``name``에 빠진 변수 이름을 담아, 호출자가 필요한 입력으로 안내할 수 있게 한다.
    """

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.name = name


# ── 공식 ────────────────────────────────────────────────────────────────
def formula_variables(expression: str) -> list[str]:
    """공식이 읽는 변수 이름(함수 이름 제외)을 처음 나온 순서대로 돌려준다."""
    names: list[str] = []
    for kind, text in _tokens(expression):
        if kind == "name" and text not in _FUNCTIONS and text not in names:
            names.append(text)
    return names


def evaluate_formula(expression: str, variables: Mapping[str, Value]) -> Value:
    """용량 공식을 계산한다.

    역할: 토큰화 후 재귀 하강 파서로 계산하고, 남은 토큰이 있거나 결과가 유한하지 않으면 거부한다.
    입력:
        expression: ``dose_rules.calculation``의 공식 문자열.
        variables: 변수 이름 → 값(숫자 또는 (하한, 상한) 범위).
    출력: 숫자 또는 (하한, 상한) 범위.
    예외: 지원하지 않는 문법·0으로 나누기·무한값이면 ``DoseFormulaError``, 값이 없는 변수면 ``MissingInput``.
    """
    parser = _Parser(_tokens(expression), variables)
    value = parser.expression()
    if parser.position != len(parser.tokens):
        raise DoseFormulaError(f"unexpected token in formula: {expression!r}")
    if any(not math.isfinite(part) for part in _parts(value)):
        raise DoseFormulaError(f"non-finite result: {expression!r}")
    return value


def _tokens(expression: str) -> list[tuple[str, str]]:
    """공식을 (종류, 텍스트) 토큰 목록으로 나눈다.

    역할: NFKC 정규화 후 ``×``·``÷``를 ``*``·``/``로 바꾸고, 숫자(``number``)·이름(``name``)·연산자(``op``) 토큰으로 자른다.
    입력:
        expression: 공식 문자열.
    출력: (``number``|``name``|``op``, 토큰 텍스트) 목록.
    예외: 토큰으로 읽을 수 없는 문자가 있으면 ``DoseFormulaError``.
    """
    text = unicodedata.normalize("NFKC", expression).replace("×", "*").replace("÷", "/")
    tokens: list[tuple[str, str]] = []
    position = 0
    while position < len(text):
        if text[position:].strip() == "":
            break
        match = _TOKEN.match(text, position)
        if match is None or match.end() == position:
            raise DoseFormulaError(f"unsupported character in formula: {expression!r}")
        kind = match.lastgroup or "op"
        tokens.append((kind, match.group(kind)))
        position = match.end()
    return tokens


def _parts(value: Value) -> tuple[float, ...]:
    """숫자는 1개짜리, 범위는 (하한, 상한) 튜플로 돌려준다."""
    return value if isinstance(value, tuple) else (value,)


def _combine(left: Value, right: Value, operation) -> Value:
    """이항 연산. 한쪽이라도 범위면 양 끝점의 모든 조합 결과의 (최솟값, 최댓값) 범위를 돌려준다."""
    if not isinstance(left, tuple) and not isinstance(right, tuple):
        return operation(left, right)
    results = [operation(a, b) for a in _parts(left) for b in _parts(right)]
    return (min(results), max(results))


class _Parser:
    """용량 공식용 재귀 하강 파서: expression := term (('+'|'-') term)*; term := unary (('*'|'/') unary)*.

    역할: 토큰 목록과 변수 값을 받아 위치(``position``)를 옮겨 가며 식을 바로 계산한다. 공식마다 새로 만든다.
    """

    def __init__(self, tokens: list[tuple[str, str]], variables: Mapping[str, Value]) -> None:
        self.tokens, self.variables, self.position = tokens, variables, 0

    def peek(self) -> tuple[str, str] | None:
        """현재 토큰(끝이면 ``None``)을 소비하지 않고 본다."""
        return self.tokens[self.position] if self.position < len(self.tokens) else None

    def take(self, op: str | None = None) -> tuple[str, str]:
        """현재 토큰을 소비한다. ``op``를 주면 그 연산자여야 하며, 아니면 ``DoseFormulaError``."""
        token = self.peek()
        if token is None or (op is not None and token != ("op", op)):
            raise DoseFormulaError(f"expected {op or 'a value'}")
        self.position += 1
        return token

    def expression(self) -> Value:
        """덧셈·뺄셈 수준의 식을 계산한다."""
        value = self.term()
        while (token := self.peek()) in (("op", "+"), ("op", "-")):
            self.take()
            right = self.term()
            value = _combine(value, right, (lambda a, b: a + b) if token[1] == "+" else (lambda a, b: a - b))
        return value

    def term(self) -> Value:
        value = self.unary()
        while (token := self.peek()) in (("op", "*"), ("op", "/")):
            self.take()
            right = self.unary()
            if token[1] == "/" and 0 in _parts(right):
                raise DoseFormulaError("division by zero")
            value = _combine(value, right, (lambda a, b: a * b) if token[1] == "*" else (lambda a, b: a / b))
        return value

    def unary(self) -> Value:
        """단항 ``-``를 처리하고 나머지는 ``atom``에 넘긴다."""
        if self.peek() == ("op", "-"):
            self.take()
            return _combine(-1.0, self.unary(), lambda a, b: a * b)
        return self.atom()

    def atom(self) -> Value:
        """숫자·변수·함수 호출·괄호식·범위 리터럴 하나를 계산한다.

        역할: ``min``/``max`` 호출은 인자 중 범위가 있으면 하한끼리·상한끼리 따로 적용해 범위를 만든다.
        범위 리터럴 ``[low, high]``는 두 값을 정렬해 (작은 값, 큰 값)으로 만들고, 범위 안의 범위는 거부한다.
        입력: 없음(파서의 현재 토큰을 읽는다).
        출력: 숫자 또는 (하한, 상한) 범위.
        예외: 모르는 함수·중첩 범위·예상 밖 토큰이면 ``DoseFormulaError``, 값이 없는 변수면 ``MissingInput``.
        """
        kind, text = self.take()
        if kind == "number":
            return float(text)
        if kind == "name":
            if self.peek() == ("op", "("):
                function = _FUNCTIONS.get(text)
                if function is None:
                    raise DoseFormulaError(f"unsupported function: {text}")
                self.take("(")
                args = [self.expression()]
                while self.peek() == ("op", ","):
                    self.take()
                    args.append(self.expression())
                self.take(")")
                if any(isinstance(arg, tuple) for arg in args):
                    lows = [_parts(arg)[0] for arg in args]
                    highs = [_parts(arg)[-1] for arg in args]
                    return (function(lows), function(highs))
                return function(args)  # type: ignore[arg-type]
            if text not in self.variables:
                raise MissingInput(text)
            return self.variables[text]
        if text == "(":
            value = self.expression()
            self.take(")")
            return value
        if text == "[":
            low = self.expression()
            self.take(",")
            high = self.expression()
            self.take("]")
            if isinstance(low, tuple) or isinstance(high, tuple):
                raise DoseFormulaError("nested range")
            return (min(low, high), max(low, high))
        raise DoseFormulaError(f"unexpected token {text!r}")


def unit_for(key: str, declared: object) -> str | None:
    """레코드에 적힌 단위, 없으면 계산 키 접미사(``_KEY_UNITS``)로 추정한 단위, 둘 다 없으면 ``None``."""
    if isinstance(declared, str) and declared.strip():
        return declared.strip()
    lowered = key.lower()
    return next((unit for suffix, unit in _KEY_UNITS if lowered.endswith(suffix)), None)


def rounded(value: Value) -> float | list[float]:
    """표시용 반올림: 절댓값 1 이상은 소수 1자리, 미만은 3자리. 범위는 [하한, 상한] 목록으로 돌려준다."""
    def one(number: float) -> float:
        return round(number, 1) if abs(number) >= 1 else round(number, 3)

    return [one(part) for part in value] if isinstance(value, tuple) else one(value)


# ── 신기능 구간 ────────────────────────────────────────────────────────────
def select_band(bands: object, value: float) -> Mapping[str, Any] | None:
    """값이 들어가는 첫 번째 신기능 구간을 고른다.

    역할: 목록 순서대로 ``range``의 ``min``·``max``(둘 다 포함 경계)를 검사한다. ``range``가 사전이 아닌 구간은 건너뛴다.
    입력:
        bands: 구간 사전 목록(목록이 아니면 빈 목록으로 본다).
        value: 신기능 값(eGFR 또는 CrCl).
    출력: 해당 구간 사전, 없으면 ``None``.
    """
    for band in bands if isinstance(bands, list) else ():
        limits = band.get("range") if isinstance(band, Mapping) else None
        if not isinstance(limits, Mapping):
            continue
        if ("min" in limits and value < limits["min"]) or ("max" in limits and value > limits["max"]):
            continue
        return band
    return None


def range_label(limits: Mapping[str, Any], unit: str | None = None) -> str:
    """구간 경계를 한국어 범위 문구로 바꾼다: {'min': 30, 'max': 49.999} → '30 이상 50 미만'; {'min': 60.001} → '60 초과'.

    역할: 정수에 0.0011 이내로 가까운 비정수 경계(49.999, 60.001)는 그 정수의 배타 경계(미만·초과)로,
    나머지는 포함 경계(이상·이하)로 표시한다. 경계가 없으면 '전 구간'.
    입력:
        limits: ``min``·``max`` 키를 가질 수 있는 경계 사전.
        unit: 문구 끝에 붙일 단위(없으면 붙이지 않음).
    출력: 범위 문구.
    """

    def bound(number: float) -> tuple[float, bool]:
        """(표시 숫자, 정확 여부) — 49.999는 50(배타), 60.001은 60(배타)으로 표시한다."""
        nearest = round(number)
        return (nearest, False) if number != nearest and abs(number - nearest) <= 0.0011 else (number, True)

    parts: list[str] = []
    if "min" in limits:
        number, exact = bound(float(limits["min"]))
        parts.append(f"{number:g} {'이상' if exact else '초과'}")
    if "max" in limits:
        number, exact = bound(float(limits["max"]))
        parts.append(f"{number:g} {'이하' if exact else '미만'}")
    text = " ".join(parts) or "전 구간"
    return f"{text} {unit}" if unit else text


def band_daily_max_mg(band: Mapping[str, Any]) -> float | None:
    """구간에 적힌 숫자 1일 최대량(``max_daily_mg`` 또는 ``daily_mg.max``, 앞의 것 우선). 없으면 ``None``."""
    for value in (band.get("max_daily_mg"), (band.get("daily_mg") or {}).get("max") if isinstance(band.get("daily_mg"), Mapping) else None):
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    return None


# ── 처방 ─────────────────────────────────────────────────────────
@dataclass(frozen=True)
class DailyDose:
    """처방에서 계산한 1일량(mg)과 그 계산 근거 문구.

    역할: ``prescribed_daily_mg``의 결과. ``basis``는 응답에 그대로 보여 줄 계산 설명이다.
    """

    mg_per_day: float
    basis: str


def prescribed_daily_mg(dose: Mapping[str, Any] | None) -> DailyDose | None:
    """처방의 1일량 ``dosage × frequency``(mg)를 계산한다. 단위·횟수·간격이 명시되지 않으면 ``None``.

    역할: ``dosage``·``frequency``는 양수여야 하고, ``unit``은 질량 단위(mg·g·mcg와 한글 표기)여야 한다.
    ``interval``이 비어 있으면 1일 투여로 가정해 계산하고, 근거 문구(``basis``) 맨 앞에 그 가정을 밝힌다.
    1일 기준 표기(``_DAILY_INTERVALS``)면 그대로 계산하며, 주 단위 등 다른 간격이면 1일량을 내지 않는다.
    근거 문구에는 계산식과 mg 환산 결과를 함께 적는다(예: ``투여 간격 미기재 — 1일 투여로 가정해 계산:
    1회 0.5 g × 1일 2회 = 1000 mg/일``).
    입력:
        dose: EMR 처방의 ``dosage``·``unit``·``frequency``·``interval`` 사전(``MedicationMatch.dose``).
    출력: ``DailyDose``(mg/일, 소수 3자리 반올림) 또는 ``None``.
    """
    if not isinstance(dose, Mapping):
        return None
    dosage, frequency = _positive(dose.get("dosage")), _positive(dose.get("frequency"))
    unit = dose.get("unit")
    factor = _MASS_UNITS.get(re.sub(r"\s+", "", unicodedata.normalize("NFKC", unit)).lower()) if isinstance(unit, str) else None
    if dosage is None or frequency is None or factor is None:
        return None
    interval = dose.get("interval")
    if interval in (None, ""):
        assumption = "투여 간격 미기재 — 1일 투여로 가정해 계산: "  # 의료진이 가정을 먼저 보도록 맨 앞에 둔다
    elif isinstance(interval, str) and re.sub(r"\s+", "", interval).lower() in {re.sub(r"\s+", "", item) for item in _DAILY_INTERVALS}:
        assumption = ""
    else:
        return None  # 주 단위 등 다른 일정: 1일량 없음
    daily = round(dosage * frequency * factor, 3)
    return DailyDose(daily, f"{assumption}1회 {dosage:g} {unit} × 1일 {frequency:g}회 = {daily:g} mg/일")


def _positive(value: object) -> float | None:
    """처방 수치를 양수 float로 읽는다.

    역할: 숫자, 또는 쉼표를 뺀 뒤 정수·소수 형태인 문자열을 받아들인다. bool·0 이하·무한값은 거부한다.
    입력:
        value: EMR의 ``dosage``·``frequency`` 값.
    출력: 유한한 양수 float, 아니면 ``None``.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, str):
        text = value.strip().replace(",", "")
        value = float(text) if re.fullmatch(r"\d+(?:\.\d+)?", text) else None
    if isinstance(value, (int, float)) and math.isfinite(value) and value > 0:
        return float(value)
    return None
