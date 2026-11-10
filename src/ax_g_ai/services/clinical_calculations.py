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
    category: str | None = None

    def as_dict(self) -> dict[str, object]:
        result: dict[str, object] = {"name": self.name, "value": self.value, "unit": self.unit, "method": self.method}
        if self.category is not None:
            result["category"] = self.category
        result["review_notice"] = self.review_notice
        return result


# CKD-EPI 2021 creatinine equation without a race coefficient (Inker et al.,
# N Engl J Med 2021). KDIGO 2024 Rec 1.1.2.1 / PP 1.2.4.2: eGFRcr, no race.
_CKD_EPI_2021 = {"female": (0.7, -0.241, 1.012), "male": (0.9, -0.302, 1.0)}
# KDIGO GFR categories (lower bound inclusive), mL/min/1.73 m².
_GFR_CATEGORIES = ((90, "G1"), (60, "G2"), (45, "G3a"), (30, "G3b"), (15, "G4"), (0, "G5"))
_EGFR_NOTICE = (
    "IDMS 표준화 크레아티닌이 안정 상태일 때만 유효합니다. 병원 검사실 보고 eGFR 식과 다를 수 있으며, "
    "단일 측정으로 CKD를 진단하지 않습니다. 의료진이 최종 확인해야 합니다."
)


def ckd_epi_2021_egfr(age_years: float, sex: str, serum_creatinine_mg_dl: float) -> float:
    """Return CKD-EPI 2021 creatinine eGFR (mL/min/1.73 m²), unrounded."""
    kappa, alpha, sex_factor = _CKD_EPI_2021[sex]
    ratio = serum_creatinine_mg_dl / kappa
    return 142 * min(ratio, 1) ** alpha * max(ratio, 1) ** -1.200 * 0.9938 ** age_years * sex_factor


def gfr_category(egfr: float) -> str:
    return next(label for lower, label in _GFR_CATEGORIES if egfr >= lower)


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
    egfr_missing = [label for label, value in (("연령", age), ("성별(male/female)", sex), ("혈청 크레아티닌(mg/dL)", creatinine)) if value is None]
    if not egfr_missing:
        egfr = round(ckd_epi_2021_egfr(age, sex, creatinine), 1)
        results.append(CalculationResult(
            "추정 GFR", egfr, "mL/min/1.73m²", "CKD-EPI 2021 크레아티닌(인종 계수 없음)",
            review_notice=_EGFR_NOTICE, category=gfr_category(egfr),
        ))
    else:
        needed.append("CKD-EPI 2021 eGFR 계산에는 " + ", ".join(egfr_missing) + " 입력이 필요합니다.")
    return tuple(results), tuple(needed)


def _number(value: object, minimum: float, maximum: float) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) and minimum <= number <= maximum else None
