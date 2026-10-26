"""Provider 입력 전 EMR snapshot의 직접 식별자를 제거한다."""

from __future__ import annotations

from typing import Any, Mapping


_IDENTIFIER_KEYS = frozenset({
    "patient", "patientid", "patientname", "name", "firstname", "lastname", "fullname",
    "birthdate", "dateofbirth", "phone", "mobile", "email", "address", "zipcode", "ssn", "rrn",
    "nationalid", "medicalrecordnumber", "mrn", "encounterid", "visitid", "recordid", "documentid",
    "providerid", "doctorid", "userid",
})


def deidentified_snapshot(snapshot: Mapping[str, Any]) -> dict[str, object]:
    """모든 임상 필드를 보존하되 직접 식별자와 지원하지 않는 값은 재귀적으로 제거한다."""
    return {
        key: safe_value
        for key, raw_value in snapshot.items()
        if not is_identifier_key(key)
        and (safe_value := _deidentify_value(raw_value)) is not None
    }


def _deidentify_value(value: object) -> object | None:
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
            and (safe_value := _deidentify_value(raw_value)) is not None
        }
        return result or None
    if isinstance(value, list):
        result = [safe_value for item in value if (safe_value := _deidentify_value(item)) is not None]
        return result or None
    return None


def is_identifier_key(key: str) -> bool:
    normalized = "".join(character for character in key.casefold() if character.isalnum())
    return normalized in _IDENTIFIER_KEYS or normalized.endswith("identifier")
