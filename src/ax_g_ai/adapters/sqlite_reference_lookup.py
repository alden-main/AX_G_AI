"""SQLite 참조 데이터 DB(약품·상병명)의 읽기 전용 조회(``services.ports.ReferenceLookup`` 구현).

설계상 위치: 어댑터 계층. 참조 DB 경로가 설정되면 ``runtime``이 만들어 EMR 매퍼(처방·진단 보강)와 임상 채팅 서비스
(질문에 나온 약품·상병 참조)에 넘긴다.

역할: 약품 코드·상병 코드로 한 건을 조회하고, 질문에 이름이 직접 나온 약품·상병의 참조 항목을 출처 메타데이터와 함께
돌려준다. DB를 쓸 수 없거나 손상됐으면 빈 결과로 대신한다.
"""

from __future__ import annotations

import sqlite3
from collections import Counter
from collections.abc import Sequence
from contextlib import closing
from pathlib import Path

from ax_g_ai.services.ports import MedicineReference
from ax_g_ai.services.term_matcher import TermMatcher

_MAX_MATCHES = 5
_MIN_NAME_LENGTH = 2
_REFERENCE_TEXT_LIMIT = 600


class SqliteReferenceLookup:
    """환자 데이터를 쓰지 않고 승인된 로컬 참조 데이터를 조회한다.

    역할: ``ReferenceLookup`` 구현. ``instr(question, name_ko)``는 인덱스를 쓸 수 없어 질문마다 340MB 테이블을 훑는 데
    약 1초가 걸렸다. 그래서 상병명·약품명을 한 번만 읽어 메모리에서 매칭하고(수 ms), 매칭된 행의 긴 본문만 SQLite에서 읽는다.
    """

    def __init__(self, database_path: Path) -> None:
        """읽기 전용 URI를 정하고 상병명·약품명 매처와 메타데이터를 미리 읽는다.

        역할: 2자 이상인 이름만 매처에 넣고, 같은 약품 코드가 여러 원본 행에 대응하면 그 코드는 약품 매처에서 뺀다.
        입력:
            database_path: SQLite 참조 DB 파일 경로(``mode=ro``로 연다).
        출력: 없음(조회기 생성).
        """
        self._uri = f"file:{database_path}?mode=ro"
        self._diseases: TermMatcher[str] = TermMatcher(
            (name, code)
            for code, name in self._rows("SELECT code, name_ko FROM diseases")
            if len(name) >= _MIN_NAME_LENGTH
        )
        medicines = self._rows("SELECT kd_code, name_ko FROM medicines")
        # 정규화된 옛 코드 하나가 여러 원본 레코드에 대응할 수 있다. 그때는 임의로 고르지 않는다.
        code_counts = Counter(code for code, _name in medicines)
        self._medicines: TermMatcher[str] = TermMatcher(
            (name, code)
            for code, name in medicines
            if code_counts[code] == 1 and len(name) >= _MIN_NAME_LENGTH
        )
        self._metadata: dict[str, str] = dict(self._rows("SELECT key, value FROM metadata"))

    def medicine(self, code: object) -> MedicineReference | None:
        """약품 코드 하나로 참조 약품을 조회한다.

        역할: 코드 앞뒤 공백을 빼고 ``kd_code``로 조회해, 정확히 한 행이 대응할 때만 결과로 삼는다.
        입력:
            code: EMR의 약품 코드(문자열이 아니거나 비었으면 조회하지 않는다).
        출력: ``MedicineReference``. 없거나 여러 행이면 ``None``.
        """
        if not isinstance(code, str) or not code.strip():
            return None
        rows = self._rows(
            "SELECT name_ko, efficacy, precautions FROM medicines WHERE kd_code = ?", (code.strip(),)
        )
        if len(rows) != 1:
            return None
        return MedicineReference(*rows[0])

    def disease(self, code: object) -> str | None:
        """상병 코드의 한글명. 없으면 ``None``."""
        if not isinstance(code, str) or not code.strip():
            return None
        rows = self._rows("SELECT name_ko FROM diseases WHERE code = ?", (code.strip(),))
        return str(rows[0][0]) if rows else None

    def search_question(self, question: str) -> list[dict[str, str]]:
        """질문에 이름이 명시된 레코드를 개수 제한을 두고 찾는다(환자 텍스트는 검색하지 않는다).

        역할: 메모리 매처로 질문에 나온 상병명·약품명을 각각 최대 ``_MAX_MATCHES``개 찾고, 약품은 효능·주의사항을 한 번의
        쿼리로 읽어 ``_REFERENCE_TEXT_LIMIT``자까지 자른다. 항목마다 출처·SHA-256·생성 시각 메타데이터를 붙인다.
        입력:
            question: 사용자 질문 원문.
        출력: 상병 참조 항목 뒤에 약품 참조 항목이 오는 dict 목록. 질문이 비었으면 빈 목록.
        """
        if not question.strip():
            return []
        diseases = _named_in(self._diseases, question)
        medicines = _named_in(self._medicines, question)
        details: dict[str, tuple[str | None, str | None]] = {}
        if medicines:
            codes = [code for code, _name in medicines]
            details = {
                code: (efficacy, precautions)
                for code, efficacy, precautions in self._rows(
                    f"SELECT kd_code, efficacy, precautions FROM medicines WHERE kd_code IN ({','.join('?' * len(codes))})",
                    codes,
                )
            }
        metadata = self._metadata
        results = [
            {"type": "상병명 참조", "code": code, "name": name,
             "source": metadata.get("disease_source", "출처 미상"),
             "source_sha256": metadata.get("disease_sha256", ""),
             "generated_at": metadata.get("generated_at", "")}
            for code, name in diseases
        ]
        results.extend(
            {"type": "약품 일반 참조", "code": code, "name": name,
             "efficacy": (details.get(code, (None, None))[0] or "")[:_REFERENCE_TEXT_LIMIT],
             "precautions": (details.get(code, (None, None))[1] or "")[:_REFERENCE_TEXT_LIMIT],
             "source": metadata.get("medicine_source", "출처 미상"),
             "source_sha256": metadata.get("medicine_sha256", ""),
             "generated_at": metadata.get("generated_at", "")}
            for code, name in medicines
        )
        return results

    def _rows(self, sql: str, parameters: Sequence[object] = ()) -> list[tuple]:
        # 참조 보강은 선택 기능이다. 로컬 DB를 쓸 수 없거나 손상돼도 환자 요청을 막으면 안 된다.
        try:
            with closing(sqlite3.connect(self._uri, uri=True)) as connection:
                return connection.execute(sql, parameters).fetchall()
        except sqlite3.Error:
            return []


def _named_in(names: TermMatcher[str], question: str) -> list[tuple[str, str]]:
    """질문에 나온 이름 중 가장 긴 것들의 (코드, 이름). 예전 SQL과 같은 순서(이름 길이 내림차순, 같으면 코드 순)다."""
    found = {match.term: match.values for match in names.matches(question)}
    matches = [(code, name) for name, codes in found.items() for code in codes]
    return sorted(matches, key=lambda entry: (-len(entry[1]), entry[0]))[:_MAX_MATCHES]
