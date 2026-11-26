"""AI 답변의 문장 단위 결정적 검증: 답변의 날짜·수치가 이 요청에 실린 자료에 있는지 코드로 확인한다.

설계상 위치: 서비스 계층. ``ClinicalChatService``가 Provider 스트림을 문장으로 나눠(``SentenceSplitter``) 내보내기 전에
문장마다 ``check_sentence``로 검사한다. 벤치마크 출시 게이트(``benchmarks.gate``)도 같은 검사를 쓴다.

역할: 프롬프트 규칙은 모델이 지키지 않을 수 있다(작은 모델의 날짜 차용, 예시 수치 누출, 측정 시점 단정). 그래서 코드로 검사할 수
있는 성질은 생성 뒤에 검사해 하한선을 보장한다. 검사하는 것은 다음뿐이다.

- ``unsupported_date``: 이 요청의 자료(EMR·규칙 평가·근거·질문)에 없는 날짜(일까지 쓴 날짜만).
- ``unsupported_value``: 임상 단위가 붙은 수치(mg/dL, mmHg, mL/min …)가 자료에 없는 값. 기준값 표현(``<30``, ``30 미만``)과
  변화율 같은 일반 ``%``는 검사하지 않는다.
- ``untimed_glucose_timing``: 측정 시점이 기록되지 않은 혈당 값에 공복·식후·비공복을 붙임.
- ``borrowed_prescription_date``: 처방일이 없는 처방에 날짜를 붙임.
- ``prescription_directive``: 처방 중단·변경 등을 명령형으로 지시함('검토할 수 있습니다'는 허용).

임상 판단의 적절성과 누락은 검사하지 않는다(평가셋·의료진 검토의 몫).
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

Violation = Literal[
    "unsupported_date", "unsupported_value", "untimed_glucose_timing", "borrowed_prescription_date", "prescription_directive",
]

# 일까지 쓴 날짜만 검사한다("2025년 12월 8.8%"처럼 월 뒤의 수치를 날짜로 읽지 않도록).
_DATE = re.compile(r"(20\d\d)\s*(?:년\s*(\d{1,2})\s*월\s*(\d{1,2})\s*일|[-./](\d{1,2})[-./](\d{1,2})(?!\d))")
_SOURCE_DATE = re.compile(r"(20\d\d)-(\d\d)-(\d\d)")
_NUMBER = re.compile(r"(?<![\d.])\d+(?:\.\d+)?")
_UNIT = (r"mg/dL|mmHg|mL/min(?:/1\.73\s*m(?:²|2))?|mmol/L|mEq/L|g/dL|[uμµ]IU/mL|ng/mL|pg/mL|U/L|IU/L|10\^3/[uμµ]L|"
         r"10\^6/[uμµ]L|kg/m(?:²|2)|kg|cm|bpm|회/분|%")
_VALUE = re.compile(rf"(?<![\d.])(\d+(?:\.\d+)?)(?:\s*/\s*(\d+(?:\.\d+)?))?\s*({_UNIT})")
_THRESHOLD_BEFORE = re.compile(r"[<>≤≥=~∼]\s*$|이상\s*$|이하\s*$")
# 모델이 계산한 차이("약 18 mmHg 상승", "0.07 mg/dL 감소")와 일반 기준값("정상 상한(100 mg/dL)")은 환자 기록값이
# 아니므로 검사하지 않는다(환자 값을 지어내는 것을 막는 것이 목적이다).
_REFERENCE_BEFORE = re.compile(r"(?:정상|기준|상한|하한|범위|목표|진단|컷오프|cut-?off)[^\d\n]{0,12}$", re.I)
_RANGE_START = re.compile(r"(?<![\d.])\d+(?:\.\d+)?\s*[–\-~∼]\s*$")
_DIFFERENCE_AFTER = re.compile(r"^\s*(?:상승|감소|증가|하락|차이|높|낮|올|내려|떨어)")
_THRESHOLD_AFTER = re.compile(rf"^\s*(?:{_UNIT})?\s*(?:미만|이하|이상|초과|전후|정도의\s*기준)")
# '%'는 수치 기록(당화혈색소·산소포화도)일 때만 검사한다. 변화율("약 7% 감소")은 모델이 계산한 값이다.
_PERCENT_RECORD = re.compile(r"(?:HbA1c|A1c|당화혈색소|산소포화도|SpO2|SpO₂|O2\s*sat|O₂\s*sat)[^.\n]{0,25}$", re.I)
_TIMING = re.compile(r"공복|식후|식전|비공복")
_TIMING_NEGATED = re.compile(r"기록\s*없|미기재|미상|확인되지|알 수 없|불명|단정")
_DIRECTIVE = re.compile(
    r"(?:중단|대체|교체|변경|감량|증량|처방|투여|복용)\s*(?:하세요|하십시오|해야\s*(?:합니다|한다)|하여야\s*합니다|이\s*필요합니다|가\s*필요합니다)"
)
_RECORD_SPLIT = re.compile(r"[,;/|)]|\.\s")
_DATE_KEYS = ("prescriptionDate", "prescription_date", "startDate", "start_date", "orderDate", "order_date")

OMISSION_NOTE = "(일부 내용은 EMR과 대조되지 않아 생략했습니다.)"


@dataclass(frozen=True)
class Facts:
    """검사 기준: 이 요청에 실린 자료의 날짜·수치와, 측정 시점 없는 혈당 값·처방일 없는 약 이름."""

    dates: frozenset[str]
    numbers: frozenset[float]
    untimed_glucose: frozenset[float]
    undated_drugs: tuple[str, ...]


def build_facts(snapshot: Mapping[str, Any], *others: object) -> Facts:
    """EMR 요약과 그 밖의 자료(규칙 평가·근거·질문·참고 기록)로 검사 기준을 만든다.

    역할: 날짜와 수치는 자료 전체(JSON 문자열)에서 모으고, 혈당 측정 시점과 처방일은 EMR 요약의 행에서 읽는다.
        같은 값이 측정 시점이 있는 혈당에도 있으면 시점 단정 검사에서 뺀다(어느 행인지 문장만으로 알 수 없음).
    """
    text = json.dumps([snapshot, *others], ensure_ascii=False, default=str)
    dates = frozenset("-".join(parts) for parts in _SOURCE_DATE.findall(text))
    numbers = frozenset(float(number) for number in _NUMBER.findall(text))
    readings = [row for row in snapshot.get("blood_sugar") or () if isinstance(row, Mapping)]
    timed = {_float(row.get("value")) for row in readings if row.get("timing_type")}
    untimed = frozenset(value for row in readings if not row.get("timing_type")
                        if (value := _float(row.get("value"))) is not None and value not in timed)
    names: list[str] = []
    for row in snapshot.get("prescriptions") or ():
        if isinstance(row, Mapping) and not any(row.get(key) not in (None, "") for key in _DATE_KEYS):
            names.extend(_drug_stems(str(row.get("med_name") or "")))
    return Facts(dates, numbers, untimed, tuple(dict.fromkeys(name for name in names if len(name) >= 3)))


def check_sentence(sentence: str, facts: Facts) -> Violation | None:
    """문장 하나를 검사해 첫 위반 종류를 돌려준다(없으면 ``None``)."""
    for match in _DATE.finditer(sentence):
        if _iso(match) not in facts.dates:
            return "unsupported_date"
    for match in _VALUE.finditer(sentence):
        unit = match.group(3)
        before, after = sentence[max(0, match.start() - 4):match.start()], sentence[match.end():match.end() + 8]
        if _THRESHOLD_BEFORE.search(before) or _THRESHOLD_AFTER.match(after):
            continue
        # 범위("4.0–4.5 uIU/mL")는 범위 앞 글자로 기준값인지 본다.
        start = match.start()
        if (lower := _RANGE_START.search(sentence[:start])) is not None:
            start = lower.start()
        if _DIFFERENCE_AFTER.match(after) or _REFERENCE_BEFORE.search(sentence[max(0, start - 20):start]):
            continue
        if unit == "%" and not _PERCENT_RECORD.search(sentence[max(0, match.start() - 40):match.start()]):
            continue
        values = [match.group(1), *([match.group(2)] if match.group(2) else [])]
        if any(float(value) not in facts.numbers for value in values):
            return "unsupported_value"
    for match in _NUMBER.finditer(sentence):
        if float(match.group(0)) in facts.untimed_glucose:
            window = sentence[max(0, match.start() - 14):match.end() + 14]
            if "mg" in window or "혈당" in window:
                if _TIMING.search(window) and not _TIMING_NEGATED.search(window):
                    return "untimed_glucose_timing"
    for name in facts.undated_drugs:
        for match in re.finditer(re.escape(name), sentence):
            head = _RECORD_SPLIT.split(sentence[max(0, match.start() - 25):match.start()])[-1]
            tail = _RECORD_SPLIT.split(sentence[match.end():match.end() + 60])[0]
            if _DATE.search(head) or _DATE.search(tail):
                return "borrowed_prescription_date"
    if _DIRECTIVE.search(sentence):
        return "prescription_directive"
    return None


class SentenceSplitter:
    """스트림 조각을 이어 받아 완성된 문장(줄·문장 끝까지, 뒤 공백 포함)을 내놓는다.

    역할: 줄바꿈, 또는 마침표·물음표·느낌표 뒤 공백을 문장 끝으로 본다. 숫자 바로 뒤의 마침표("0.92", "2026. 6. 23.")는
        끝으로 보지 않는다. 끝이 없이 ``limit`` 글자를 넘으면 그대로 내보낸다.
    """

    _END = re.compile(r"\n+|(?<!\d)[.!?。][ \t]+")

    def __init__(self, limit: int = 600) -> None:
        self._buffer = ""
        self._limit = limit

    def feed(self, text: str) -> list[str]:
        """``text``를 더하고 완성된 문장들을 돌려준다."""
        self._buffer += text
        sentences: list[str] = []
        while match := self._END.search(self._buffer):
            sentences.append(self._buffer[:match.end()])
            self._buffer = self._buffer[match.end():]
        if len(self._buffer) >= self._limit:
            sentences.append(self._buffer)
            self._buffer = ""
        return sentences

    def flush(self) -> list[str]:
        """남은 내용(마지막 문장)을 돌려준다."""
        rest, self._buffer = self._buffer, ""
        return [rest] if rest else []


def _iso(match: re.Match[str]) -> str:
    year, month, day = match.group(1), match.group(2) or match.group(4), match.group(3) or match.group(5)
    return f"{year}-{int(month):02d}-{int(day):02d}"


def _float(value: object) -> float | None:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _drug_stems(name: str) -> Iterable[str]:
    """약품명에서 답변에 나올 만한 이름 조각: 괄호 앞 제품명과 괄호 안 성분명의 앞 네 글자."""
    product = re.split(r"[(\d]", name, maxsplit=1)[0].strip()
    yield product[:4]
    for ingredient in re.findall(r"\(([^)]+)\)", name):
        yield ingredient.strip()[:4]
