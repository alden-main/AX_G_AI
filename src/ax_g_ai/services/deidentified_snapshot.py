"""Provider 입력 전 EMR 값에서 직접 식별자를 제거하는 단일 경계.

설계상 위치: 서비스 계층의 비식별 도구. ``adapters.main_bridge``가 원문 EMR을 문맥으로 바꿀 때 값마다 적용한다.

역할: 이름·생년월일·연락처·주소·주민등록번호·진료/문서/사용자 ID 같은 식별자 키(대소문자와 영숫자 외 문자는 무시,
``identifier``로 끝나는 키 포함)를 재귀적으로 빼고, 비게 된 dict·목록과 지원하지 않는 타입도 버린다.
"""

from __future__ import annotations

from collections.abc import Mapping

_IDENTIFIER_KEYS = frozenset({
    "patient", "patientid", "patientname", "name", "firstname", "lastname", "fullname",
    "birthdate", "dateofbirth", "phone", "mobile", "email", "address", "zipcode", "ssn", "rrn",
    "nationalid", "medicalrecordnumber", "mrn", "encounterid", "visitid", "recordid", "documentid",
    "providerid", "doctorid", "userid",
})


def deidentify(value: object) -> object | None:
    """임상 값은 보존하되 식별자 키·빈 값·지원하지 않는 타입은 재귀적으로 제거한다.

    역할: 문자열·숫자·불리언은 그대로 둔다(빈 문자열도 유지). Mapping은 식별자가 아닌 문자열 키만, 목록은 남는 항목만
    재귀로 남기고, 비게 된 Mapping·목록과 그 밖의 타입은 ``None``으로 버린다.
    입력:
        value: EMR에서 온 임의의 JSON 값.
    출력: 식별자를 뺀 값. 남는 것이 없으면 ``None``.
    """
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        result = {
            key: safe_value
            for key, raw_value in value.items()
            if isinstance(key, str)
            and not is_identifier_key(key)
            and (safe_value := deidentify(raw_value)) is not None
        }
        return result or None
    if isinstance(value, list):
        items = [safe_value for item in value if (safe_value := deidentify(item)) is not None]
        return items or None
    return None


def is_identifier_key(key: str) -> bool:
    """키를 소문자 영숫자로 정규화해 식별자 키 목록에 있거나 ``identifier``로 끝나면 True."""
    normalized = "".join(character for character in key.casefold() if character.isalnum())
    return normalized in _IDENTIFIER_KEYS or normalized.endswith("identifier")
