"""검토 보조용으로만 쓰는 결정적(deterministic) 임상 계산(BMI·체표면적·CrCl·eGFR).

설계상 위치: 서비스 계층의 임상 계산 도구. Bridge 어댑터(``adapters/main_bridge.py``)가 채팅 응답의 계산값을 만들 때
``calculate_review_aids``를 쓰고, EMR 매핑(``emr_mapping``)과 임상 추천(``clinical_recommendation``)이
``ckd_epi_2021_egfr``·``cockcroft_gault_crcl``·``body_mass_index``를 같은 식으로 재사용한다.

역할: 주어진 입력값만으로 공식을 계산한다. 빠진 입력은 추정하지 않고 필요한 입력 안내 문구로 돌려주며,
용량은 계산하지 않는다. 결과는 의료진이 최종 확인해야 하는 참고값이다.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class CalculationResult:
    """계산값 하나와 그 단위·방법·확인 안내.

    역할: ``calculate_review_aids``의 결과 단위. ``category``는 eGFR의 KDIGO GFR 범주처럼 값에 딸린 분류이며,
    ``as_dict``로 응답 payload 형태의 사전이 된다.
    """

    name: str
    value: float
    unit: str
    method: str
    review_notice: str = "입력값·적용 체중·임상 적합성은 의료진이 최종 확인해야 합니다."
    category: str | None = None

    def as_dict(self) -> dict[str, object]:
        """응답용 사전으로 바꾼다(``category``는 있을 때만 넣고, ``review_notice``는 마지막에 둔다)."""
        result: dict[str, object] = {"name": self.name, "value": self.value, "unit": self.unit, "method": self.method}
        if self.category is not None:
            result["category"] = self.category
        result["review_notice"] = self.review_notice
        return result


# 인종 계수가 없는 CKD-EPI 2021 크레아티닌 식(Inker et al.,
# N Engl J Med 2021). KDIGO 2024 Rec 1.1.2.1 / PP 1.2.4.2: eGFRcr, 인종 계수 없음.
# 성별마다 (kappa, alpha, 성별 계수).
_CKD_EPI_2021 = {"female": (0.7, -0.241, 1.012), "male": (0.9, -0.302, 1.0)}
# KDIGO GFR 범주(하한 포함), mL/min/1.73 m².
_GFR_CATEGORIES = ((90, "G1"), (60, "G2"), (45, "G3a"), (30, "G3b"), (15, "G4"), (0, "G5"))
_EGFR_NOTICE = (
    "IDMS 표준화 크레아티닌이 안정 상태일 때만 유효합니다. 병원 검사실 보고 eGFR 식과 다를 수 있으며, "
    "단일 측정으로 CKD를 진단하지 않습니다. 의료진이 최종 확인해야 합니다."
)


def ckd_epi_2021_egfr(age_years: float, sex: str, serum_creatinine_mg_dl: float) -> float:
    """CKD-EPI 2021 크레아티닌 eGFR(mL/min/1.73 m²)을 반올림하지 않고 돌려준다(``sex``는 ``"male"``/``"female"``)."""
    kappa, alpha, sex_factor = _CKD_EPI_2021[sex]
    ratio = serum_creatinine_mg_dl / kappa
    return 142 * min(ratio, 1) ** alpha * max(ratio, 1) ** -1.200 * 0.9938 ** age_years * sex_factor


def cockcroft_gault_crcl(age_years: float, sex: str, weight_kg: float, serum_creatinine_mg_dl: float) -> float:
    """주어진(실제) 체중으로 Cockcroft-Gault 크레아티닌 청소율(mL/min)을 반올림하지 않고 돌려준다(여성은 × 0.85)."""
    return (140 - age_years) * weight_kg / (72 * serum_creatinine_mg_dl) * (0.85 if sex == "female" else 1.0)


def body_mass_index(weight_kg: float, height_cm: float) -> float:
    """BMI(kg/m²)를 반올림하지 않고 돌려준다."""
    height_m = height_cm / 100
    return weight_kg / (height_m * height_m)


def gfr_category(egfr: float) -> str:
    """eGFR 값이 속하는 KDIGO GFR 범주(G1–G5)."""
    return next(label for lower, label in _GFR_CATEGORIES if egfr >= lower)


def calculate_review_aids(context: object) -> tuple[tuple[CalculationResult, ...], tuple[str, ...]]:
    """입력이 갖춰진 계산값과, 갖춰지지 않은 계산의 필요 입력 안내를 돌려준다(빠진 입력·용량은 추정하지 않음).

    역할: 나이(18–120세)·체중(0.1–500 kg)·키(30–300 cm)·혈청 크레아티닌(0.01–30 mg/dL)·성별(``male``/``female``)을
    범위 검사한 뒤 BMI·체표면적(Mosteller), Cockcroft–Gault CrCl(실제 체중), CKD-EPI 2021 eGFR(KDIGO 범주 포함)을
    계산한다. 범위 밖이거나 숫자가 아닌 값은 없는 값으로 본다.
    입력:
        context: 계산 입력 사전(``age_years``·``weight_kg``·``height_cm``·``serum_creatinine_mg_dl``·``sex``).
            ``Mapping``이 아니면 빈 입력으로 본다.
    출력: (``CalculationResult`` 튜플, 필요한 입력 안내 문구 튜플).
    """
    source = context if isinstance(context, Mapping) else {}
    age = _number(source.get("age_years"), 18, 120)
    weight = _number(source.get("weight_kg"), 0.1, 500)
    height = _number(source.get("height_cm"), 30, 300)
    creatinine = _number(source.get("serum_creatinine_mg_dl"), 0.01, 30)
    sex = source.get("sex") if source.get("sex") in {"male", "female"} else None
    results: list[CalculationResult] = []
    needed: list[str] = []
    if weight is not None and height is not None:
        results.append(CalculationResult("BMI", round(body_mass_index(weight, height), 1), "kg/m²", "BMI"))
        results.append(CalculationResult("체표면적", round(math.sqrt((height * weight) / 3600), 2), "m²", "Mosteller"))
    else:
        needed.append("BMI·체표면적 계산에는 체중(kg)과 키(cm) 입력이 필요합니다.")
    if age is not None and sex is not None and weight is not None and creatinine is not None:
        crcl = cockcroft_gault_crcl(age, sex, weight, creatinine)
        results.append(CalculationResult("추정 CrCl", round(crcl, 1), "mL/min", "Cockcroft–Gault (제공된 실제 체중 사용)"))
    else:
        missing = _missing_labels(("연령", age), ("성별(male/female)", sex), ("체중(kg)", weight), ("혈청 크레아티닌(mg/dL)", creatinine))
        needed.append("Cockcroft–Gault CrCl 계산에는 " + ", ".join(missing) + " 입력이 필요합니다.")
    if age is not None and sex is not None and creatinine is not None:
        egfr = round(ckd_epi_2021_egfr(age, sex, creatinine), 1)
        results.append(CalculationResult(
            "추정 GFR", egfr, "mL/min/1.73m²", "CKD-EPI 2021 크레아티닌(인종 계수 없음)",
            review_notice=_EGFR_NOTICE, category=gfr_category(egfr),
        ))
    else:
        egfr_missing = _missing_labels(("연령", age), ("성별(male/female)", sex), ("혈청 크레아티닌(mg/dL)", creatinine))
        needed.append("CKD-EPI 2021 eGFR 계산에는 " + ", ".join(egfr_missing) + " 입력이 필요합니다.")
    return tuple(results), tuple(needed)


def _missing_labels(*inputs: tuple[str, object]) -> list[str]:
    """(표시 이름, 값) 쌍 중 값이 ``None``인 항목의 표시 이름."""
    return [label for label, value in inputs if value is None]


def _number(value: object, minimum: float, maximum: float) -> float | None:
    """유한한 숫자(bool 제외)이고 [minimum, maximum] 안이면 float, 아니면 ``None``."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) and minimum <= number <= maximum else None
