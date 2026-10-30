"""Provider 입력 전 EMR 값에서 직접 식별자를 제거하는 단일 경계."""

from __future__ import annotations

from typing import Mapping


_IDENTIFIER_KEYS = frozenset({
    "patient", "patientid", "patientname", "name", "firstname", "lastname", "fullname",
    "birthdate", "dateofbirth", "phone", "mobile", "email", "address", "zipcode", "ssn", "rrn",
    "nationalid", "medicalrecordnumber", "mrn", "encounterid", "visitid", "recordid", "documentid",
    "providerid", "doctorid", "userid",
})


def deidentify(value: object) -> object | None:
    """임상 값은 보존하되 식별자 키·빈 값·지원하지 않는 타입은 재귀적으로 제거한다."""
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
        result = [safe_value for item in value if (safe_value := deidentify(item)) is not None]
        return result or None
    return None


def is_identifier_key(key: str) -> bool:
    normalized = "".join(character for character in key.casefold() if character.isalnum())
    return normalized in _IDENTIFIER_KEYS or normalized.endswith("identifier")
