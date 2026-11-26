"""Provider 답변의 결정적 점검과 답변에 덧붙이는 DUR 점검 블록.

설계상 위치: 서비스 계층. ``ClinicalChatService``가 Provider 답변을 사용자에게 보내기 전에 이 모듈로 점검하고,
채팅·추천 답변 끝에 규칙 평가의 DUR 점검 결과를 코드로 만든 문장으로 붙인다.

역할: 작은 자체 모델에서 확인된 실패(질문을 그대로 되풀이, 내부 JSON 출력)를 스트림 앞부분에서 찾는다. 질문 되풀이는
잘라 내고, 쓸 수 없는 답변(되풀이뿐이거나 JSON)은 호출자가 다시 요청하도록 알린다. DUR 경고는 모델이 빠뜨리거나
약하게 쓰거나 뒤집는 일이 있어, 본문에서는 DUR을 판단하지 않게 하고(프롬프트) 이 블록으로만 표시한다.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

# 답변을 그대로 쓸 수 없는 문제(호출자가 다시 요청한다).
Problem = Literal["echo_only", "raw_json"]
_RAW_JSON_START = ("{", "[", "`")


@dataclass(frozen=True)
class Opening:
    """답변 앞부분 점검 결과.

    역할: ``state``가 ``wait``이면 질문 되풀이인지 아직 알 수 없어 더 받아야 하고, ``ok``이면 ``text``(되풀이를 뺀 앞부분)를
    내보내도 되며, ``bad``이면 ``problem`` 때문에 쓸 수 없다.
    """

    state: Literal["wait", "ok", "bad"]
    text: str = ""
    problem: Problem | None = None


def check_opening(text: str, question: str, *, final: bool = False) -> Opening:
    """답변 앞부분이 질문 되풀이·JSON인지 본다(공백은 무시하고 비교).

    역할: 스트리밍에서는 앞부분이 질문과 갈라지는 순간 판정하므로, 되풀이가 아니면 첫 조각을 바로 내보낸다.
    입력:
        text: 지금까지 받은 답변(앞부분).
        question: 프롬프트의 질문(되풀이 비교 대상).
        final: 답변이 끝났으면 True. 이때는 ``wait``을 돌려주지 않는다.
    출력: ``Opening``.
    """
    rest = _strip_echo(text, question)
    echoed = rest is not None
    if rest is None:
        if not final and _could_be_echo(text, question):
            return Opening("wait")
        rest = text
    body = rest.lstrip()
    if not body:
        if not final:
            return Opening("wait")
        return Opening("bad", problem="echo_only")
    if body.startswith(_RAW_JSON_START):
        return Opening("bad", problem="raw_json")
    return Opening("ok", body if echoed else rest)


# DUR 점검만 묻는 질문에 서버가 쓰는 본문(그 뒤에 DUR 점검 블록이 붙는다).
DUR_ONLY_ANSWER = "처방 점검 결과는 아래와 같습니다."


def fallback_answer(*, has_notice: bool) -> str:
    """Provider 답변이 두 번 모두 쓸 수 없을 때 오류 대신 돌려줄 본문.

    역할: DUR 점검 블록이 뒤에 붙으면 그 결과는 보여 줄 수 있음을 알리고, 없으면 다시 시도를 안내한다.
    입력:
        has_notice: 답변 끝에 DUR 점검 블록이 붙는지.
    출력: 고정 안내 문장.
    """
    if has_notice:
        return "AI 답변 본문을 만들지 못했습니다. 처방 점검 결과는 아래와 같습니다."
    return "AI 답변을 만들지 못했습니다. 질문을 조금 바꾸거나 다시 시도해 주세요."


def dur_notice(assessment: Mapping[str, Any] | None, *, prescribed: bool = True) -> str | None:
    """규칙 평가의 DUR 점검 결과를 답변 끝에 붙일 고정 문장으로 만든다(DUR 점검 결과가 없으면 ``None``).

    역할: 모델은 본문에서 DUR을 판단하지 않으므로(``prompts.clinical.DUR_RULE``) 이 블록이 DUR의 유일한 표시다. 경고마다
        유형(등급·조건)·제품·사유·비고를 한 줄로 쓰고, 경고가 없으면 없다고 쓴다. 경고가 없는 추가 예정 약과 DUR 점검에서
        빠진 처방(목록에 없는 제품코드)도 한 줄씩 밝힌다. 앞에 빈 줄을 둔다.
    입력:
        assessment: 평가 페이로드(``medication_alerts``를 읽는다).
        prescribed: EMR에 현재 처방이 있는지. 없으면 '현재 처방 경고 없음' 줄을 쓰지 않는다(점검한 처방이 없으므로).
    출력: 덧붙일 문자열 또는 ``None``(점검 결과가 없거나 밝힐 내용이 없을 때).
    """
    checked = (assessment or {}).get("medication_alerts")
    if not isinstance(checked, Mapping):
        return None
    alerts = [alert for alert in checked.get("alerts") or () if isinstance(alert, Mapping)]
    lines = [_alert_line(alert) for alert in alerts]
    flagged = {_name(product) for alert in alerts for product in alert.get("products") or () if isinstance(product, Mapping)}
    proposed = [_name(drug) for drug in checked.get("proposed") or () if isinstance(drug, Mapping)]
    lines += [f"• 추가 예정 약 {name}: 현재 DUR 목록상 해당 경고 없음" for name in proposed if name and name not in flagged]
    if not alerts and prescribed:
        lines.insert(0, "• 현재 처방: DUR 목록상 해당 경고 없음")
    lines += [f"• DUR 점검 제외(목록에 없는 제품코드): {_name(item)}" for item in checked.get("unmapped") or () if isinstance(item, Mapping)]
    if not lines:
        return None
    return "\n\n[처방 점검 — 심평원 DUR 목록]\n" + "\n".join(lines)


_DATE_KEYS = ("prescriptionDate", "prescription_date", "startDate", "start_date", "orderDate", "order_date")


def record_notice(snapshot: Mapping[str, Any]) -> str | None:
    """요약 끝에 붙일 기록 블록: EMR의 진단·처방을 코드가 나열한다(R-F29, 둘 다 없으면 ``None``).

    역할: 사고를 끈 요약이 진단 줄을 빠뜨리거나, 위반 문장을 빼면서 같은 문장의 진단이 함께 빠지는 일이 있었다. 반드시 들어가야
        하는 기록 사실은 모델에 맡기지 않는다. 빈 목록으로 기록된 항목(``recorded_none``)은 '기록 없음', 전달되지 않은 항목은
        줄을 쓰지 않는다. 앞에 빈 줄을 둔다.
    """
    recorded_none = set(snapshot.get("recorded_none") or ())
    lines = []
    diagnoses = [row for row in snapshot.get("diagnoses") or () if isinstance(row, Mapping)]
    if diagnoses:
        lines.append("• 진단: " + ", ".join(_diagnosis(row) for row in diagnoses))
    elif "진단" in recorded_none:
        lines.append("• 진단: 기록 없음")
    prescriptions = [row for row in snapshot.get("prescriptions") or () if isinstance(row, Mapping)]
    if prescriptions:
        lines.append("• 처방: " + "; ".join(_prescription(row) for row in prescriptions))
    elif "처방" in recorded_none:
        lines.append("• 처방: 기록 없음")
    return "\n\n[기록 — 진단·처방]\n" + "\n".join(lines) if lines else None


def _diagnosis(row: Mapping[str, Any]) -> str:
    name, code = row.get("name_ko"), row.get("code")
    return f"{name}({code})" if name and code else str(name or code or "")


def _prescription(row: Mapping[str, Any]) -> str:
    parts = [str(row.get("med_name") or row.get("med_code") or "")]
    if row.get("dosage") not in (None, ""):
        parts.append(f"{_number(row['dosage'])}{row.get('unit') or ''}")
    if row.get("frequency") not in (None, ""):
        parts.append(f"1일 {_number(row['frequency'])}회")
    date = next((row[key] for key in _DATE_KEYS if row.get(key) not in (None, "")), None)
    return " ".join(part for part in parts if part) + (f"(처방일 {str(date)[:10]})" if date else "(처방일 기록 없음)")


def _number(value: object) -> str:
    return f"{value:g}" if isinstance(value, float) else str(value)


def _name(item: Mapping[str, Any]) -> str:
    return str(item.get("med_name") or item.get("name") or item.get("ingredient_name") or item.get("med_code") or "")


def _alert_line(alert: Mapping[str, Any]) -> str:
    qualifiers = [value for value in (alert.get("grade"), alert.get("category"), alert.get("age_condition")) if value]
    if alert.get("status") == "possible":
        qualifiers.append("임신 여부 확인 필요" if alert.get("type") == "임부금기" else "조건 확인 필요")
    label = f"{alert.get('type')}({', '.join(qualifiers)})" if qualifiers else str(alert.get("type"))
    names = [f"{_name(product)}{' (추가 시)' if product.get('proposed') else ''}"
             for product in alert.get("products") or () if isinstance(product, Mapping)]
    line = f"• {label}: {' + '.join(names)}"
    if alert.get("detail"):
        line += f" — {alert['detail']}"
    if alert.get("condition_note"):
        line += f" (비고: {alert['condition_note']})"
    return line


def _compact(text: str) -> str:
    return "".join(text.split())


def _strip_echo(text: str, question: str) -> str | None:
    """``text``가 공백을 무시하고 ``question``으로 시작하면 그 뒤의 원문을, 아니면 ``None``을 돌려준다."""
    target = _compact(question)
    if not target:
        return None
    matched = 0
    for index, char in enumerate(text):
        if char.isspace():
            continue
        if char != target[matched]:
            return None
        matched += 1
        if matched == len(target):
            return text[index + 1:]
    return None


def _could_be_echo(text: str, question: str) -> bool:
    """지금까지 받은 ``text``가 질문의 앞부분과 같아 되풀이일 수 있는지."""
    compact = _compact(text)
    return not compact or _compact(question).startswith(compact)
