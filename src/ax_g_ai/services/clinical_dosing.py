"""Stage 3 helpers: deterministic dose arithmetic for corpus ``dose_rules`` (no LLM, no ``eval``).

- ``evaluate_formula``: the small formula language of ``dose_rules.calculation`` —
  numbers, variables, ``+ - × * /``, parentheses, ``min``/``max`` and a range literal
  ``[low, high]`` (e.g. ``weight_kg × [0.1, 0.2]``). Unknown syntax raises
  ``DoseFormulaError``; a variable without a patient value raises ``MissingInput``.
- ``select_band``: the renal-function band (``range.min``/``range.max``) holding a value.
- ``prescribed_daily_mg``: the current daily amount from an EMR prescription, only when the
  unit is a mass unit and the frequency is per day (never guessed from free text).

The results are review aids: they always travel with the inputs used and the formula, and
the recommender labels them as reference values that a clinician must confirm.
"""

from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Mapping

from ax_g_ai.services.clinical_corpus import CorpusRuleError

Value = float | tuple[float, float]

_TOKEN = re.compile(r"\s*(?:(?P<number>\d+(?:\.\d+)?)|(?P<name>[A-Za-z_][A-Za-z0-9_]*)|(?P<op>[-+*/×÷(),\[\]]))")
_FUNCTIONS = {"min": min, "max": max}
# Unit implied by a calculation key when the record does not state one (bag1_mg → mg).
_KEY_UNITS = (("_mcg_day", "mcg/day"), ("_mg_day", "mg/day"), ("_u_per_day", "U/day"), ("_mcg", "mcg"), ("_mg", "mg"),
              ("_g", "g"), ("_ml", "mL"), ("_u", "U"))
_MASS_UNITS = {"mg": 1.0, "밀리그램": 1.0, "g": 1000.0, "그램": 1000.0, "mcg": 0.001, "ug": 0.001, "µg": 0.001, "μg": 0.001,
               "마이크로그램": 0.001}
_DAILY_INTERVALS = {"1일", "일", "매일", "하루", "1day", "day", "daily", "qd", "/일", "per day", "perday", "d"}


class DoseFormulaError(CorpusRuleError):
    """A dose formula uses syntax this evaluator does not support."""


class MissingInput(LookupError):
    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.name = name


# ── formula ────────────────────────────────────────────────────────────────
def formula_variables(expression: str) -> list[str]:
    """Variable names a formula reads (functions excluded), in order of appearance."""
    names: list[str] = []
    for kind, text in _tokens(expression):
        if kind == "name" and text not in _FUNCTIONS and text not in names:
            names.append(text)
    return names


def evaluate_formula(expression: str, variables: Mapping[str, Value]) -> Value:
    parser = _Parser(_tokens(expression), variables)
    value = parser.expression()
    if parser.position != len(parser.tokens):
        raise DoseFormulaError(f"unexpected token in formula: {expression!r}")
    if any(not math.isfinite(part) for part in _parts(value)):
        raise DoseFormulaError(f"non-finite result: {expression!r}")
    return value


def _tokens(expression: str) -> list[tuple[str, str]]:
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
    return value if isinstance(value, tuple) else (value,)


def _combine(left: Value, right: Value, operation) -> Value:
    if not isinstance(left, tuple) and not isinstance(right, tuple):
        return operation(left, right)
    results = [operation(a, b) for a in _parts(left) for b in _parts(right)]
    return (min(results), max(results))


class _Parser:
    """Recursive descent: expression := term (('+'|'-') term)*; term := unary (('*'|'/') unary)*."""

    def __init__(self, tokens: list[tuple[str, str]], variables: Mapping[str, Value]) -> None:
        self.tokens, self.variables, self.position = tokens, variables, 0

    def peek(self) -> tuple[str, str] | None:
        return self.tokens[self.position] if self.position < len(self.tokens) else None

    def take(self, op: str | None = None) -> tuple[str, str]:
        token = self.peek()
        if token is None or (op is not None and token != ("op", op)):
            raise DoseFormulaError(f"expected {op or 'a value'}")
        self.position += 1
        return token

    def expression(self) -> Value:
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
        if self.peek() == ("op", "-"):
            self.take()
            return _combine(-1.0, self.unary(), lambda a, b: a * b)
        return self.atom()

    def atom(self) -> Value:
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
    if isinstance(declared, str) and declared.strip():
        return declared.strip()
    lowered = key.lower()
    return next((unit for suffix, unit in _KEY_UNITS if lowered.endswith(suffix)), None)


def rounded(value: Value) -> float | list[float]:
    def one(number: float) -> float:
        return round(number, 1) if abs(number) >= 1 else round(number, 3)

    return [one(part) for part in value] if isinstance(value, tuple) else one(value)


# ── renal bands ────────────────────────────────────────────────────────────
def select_band(bands: object, value: float) -> Mapping[str, Any] | None:
    for band in bands if isinstance(bands, list) else ():
        limits = band.get("range") if isinstance(band, Mapping) else None
        if not isinstance(limits, Mapping):
            continue
        if ("min" in limits and value < limits["min"]) or ("max" in limits and value > limits["max"]):
            continue
        return band
    return None


def range_label(limits: Mapping[str, Any], unit: str | None = None) -> str:
    """{'min': 30, 'max': 49.999} → '30 이상 50 미만'; {'min': 60.001} → '60 초과'."""

    def bound(number: float) -> tuple[float, bool]:
        """(display number, exact) — 49.999 shows as 50 (exclusive), 60.001 as 60 (exclusive)."""
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
    """A numeric daily maximum stated by a band (``max_daily_mg`` or ``daily_mg.max``)."""
    for value in (band.get("max_daily_mg"), (band.get("daily_mg") or {}).get("max") if isinstance(band.get("daily_mg"), Mapping) else None):
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    return None


# ── prescriptions ─────────────────────────────────────────────────────────
@dataclass(frozen=True)
class DailyDose:
    mg_per_day: float
    basis: str


def prescribed_daily_mg(dose: Mapping[str, Any] | None) -> DailyDose | None:
    """``dosage × frequency`` per day in mg, or None when the unit, count or interval is not explicit."""
    if not isinstance(dose, Mapping):
        return None
    dosage, frequency = _positive(dose.get("dosage")), _positive(dose.get("frequency"))
    unit = dose.get("unit")
    factor = _MASS_UNITS.get(re.sub(r"\s+", "", unicodedata.normalize("NFKC", unit)).lower()) if isinstance(unit, str) else None
    if dosage is None or frequency is None or factor is None:
        return None
    interval = dose.get("interval")
    if interval in (None, ""):
        basis = f"1회 {dosage:g} {unit} × {frequency:g}회(frequency를 1일 횟수로 봄, interval 미기재)"
    elif isinstance(interval, str) and re.sub(r"\s+", "", interval).lower() in {re.sub(r"\s+", "", item) for item in _DAILY_INTERVALS}:
        basis = f"1회 {dosage:g} {unit} × 1일 {frequency:g}회"
    else:
        return None  # weekly or other schedules: no daily amount
    return DailyDose(round(dosage * frequency * factor, 3), basis)


def _positive(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, str):
        text = value.strip().replace(",", "")
        value = float(text) if re.fullmatch(r"\d+(?:\.\d+)?", text) else None
    if isinstance(value, (int, float)) and math.isfinite(value) and value > 0:
        return float(value)
    return None
