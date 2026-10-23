"""Deterministic clinical calculations used only as review aids."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping


@dataclass(frozen=True)
class CalculationResult:
    name: str
    value: float
    unit: str
    method: str
    review_notice: str = "입력값·적용 체중·임상 적합성은 의료진이 최종 확인해야 합니다."

    def as_dict(self) -> dict[str, object]:
        return {"name": self.name, "value": self.value, "unit": self.unit, "method": self.method, "review_notice": self.review_notice}


def calculate_review_aids(context: object) -> tuple[tuple[CalculationResult, ...], tuple[str, ...]]:
    """Return supported calculation values; never infer missing inputs or doses."""
    source = context if isinstance(context, Mapping) else {}
    age = _number(source.get("age_years"), 18, 120)
    weight = _number(source.get("weight_kg"), 0.1, 500)
    height = _number(source.get("height_cm"), 30, 300)
    creatinine = _number(source.get("serum_creatinine_mg_dl"), 0.01, 30)
    sex = source.get("sex") if source.get("sex") in {"male", "female"} else None
    results: list[CalculationResult] = []
    needed: list[str] = []
    if weight is not None and height is not None:
        height_m = height / 100
        results.append(CalculationResult("BMI", round(weight / (height_m * height_m), 1), "kg/m²", "BMI"))
        results.append(CalculationResult("체표면적", round(math.sqrt((height * weight) / 3600), 2), "m²", "Mosteller"))
    else:
        needed.append("BMI·체표면적 계산에는 체중(kg)과 키(cm) 입력이 필요합니다.")
    missing = [label for label, value in (("연령", age), ("성별(male/female)", sex), ("체중(kg)", weight), ("혈청 크레아티닌(mg/dL)", creatinine)) if value is None]
    if not missing:
        factor = 0.85 if sex == "female" else 1.0
        crcl = ((140 - age) * weight / (72 * creatinine)) * factor
        results.append(CalculationResult("추정 CrCl", round(crcl, 1), "mL/min", "Cockcroft–Gault (제공된 실제 체중 사용)"))
    else:
        needed.append("Cockcroft–Gault CrCl 계산에는 " + ", ".join(missing) + " 입력이 필요합니다.")
    return tuple(results), tuple(needed)


def _number(value: object, minimum: float, maximum: float) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) and minimum <= number <= maximum else None
