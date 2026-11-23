"""자유 텍스트에서 사전 용어(약품·검사·질환·개념 이름)를 찾는 공용 매처.

설계상 위치: 서비스 계층의 공용 도구. 질문·진료 메모·처방명에서 이름을 찾는 모든 곳(근거 검색의 개념 보너스,
EMR 처방명 → 성분 매핑, DUR 약 이름 찾기, 참조 DB 질문 매칭)이 이 모듈을 쓴다.

역할: 사전마다 Aho–Corasick 오토마톤(``pyahocorasick``) 하나를 만들어, 사전 크기(수만 개 이름)와 관계없이
텍스트를 한 번만 훑어 모든 출현 위치를 찾는다. 이름을 하나씩 검사하는 방식보다 훨씬 빠르다.

호출자는 용어와 텍스트를 같은 방식으로 정규화해서 넘긴다(예: NFKC + 소문자, 한국어는 공백 제거).
``boundary``는 일치 구간의 앞뒤에 붙어 있으면 안 되는 문자들(정규식 문자 클래스 본문)이다. ``"a-z"``이면
"aspirinate" 속의 "aspirin"은 일치로 보지 않는다. 한국어 이름은 조사가 붙으므로("아스피린을") 경계를 두지 않는다.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable
from typing import NamedTuple

import ahocorasick


class TermMatch[V](NamedTuple):
    """텍스트 안에서 찾은 용어 하나의 위치와 값.

    역할: ``TermMatcher.matches``·``longest``의 결과 단위. ``values``는 같은 용어로 등록된 모든 값이다
    (예: 같은 한글명을 가진 성분 여러 개).
    """

    start: int
    end: int  # 끝 위치(이 위치의 문자는 포함하지 않음)
    term: str
    values: tuple[V, ...]  # 이 용어로 추가된 모든 값(추가한 순서)


class TermMatcher[V]:
    """사전 하나에 대한 다중 패턴 문자열 검색기.

    역할: 생성 시 (용어, 값) 쌍으로 오토마톤을 만들고, 이후 텍스트마다 겹치는 일치 전체(``matches``),
    겹치지 않는 가장 긴 일치(``longest``), 등장한 용어 집합(``terms_in``)을 돌려준다. 생성 후에는 바뀌지 않으므로
    여러 스레드에서 함께 써도 된다.
    """

    def __init__(self, terms: Iterable[tuple[str, V]], *, boundary: str | None = None) -> None:
        """용어 사전으로 오토마톤을 만든다.

        역할: 같은 용어의 값들을 하나로 묶어 등록하고, 경계 문자 검사용 정규식을 준비한다.
        입력:
            terms: (정규화된 용어, 값) 쌍. 빈 용어는 무시한다.
            boundary: 일치 구간 앞뒤에 올 수 없는 문자들(정규식 문자 클래스 본문). ``None``이면 검사하지 않는다.
        출력: 없음(검색기 생성).
        """
        grouped: dict[str, list[V]] = defaultdict(list)
        for term, value in terms:
            if term:
                grouped[term].append(value)
        self._automaton = ahocorasick.Automaton()
        for term, values in grouped.items():
            self._automaton.add_word(term, (term, tuple(values)))
        if grouped:
            self._automaton.make_automaton()
        self._size = len(grouped)
        self._boundary = re.compile(f"[{boundary}]") if boundary else None

    def __len__(self) -> int:
        return self._size

    def matches(self, text: str) -> list[TermMatch[V]]:
        """텍스트에 나온 모든 용어의 모든 출현(겹치는 것 포함)을 돌려준다.

        역할: 오토마톤으로 한 번 훑은 뒤 경계 조건을 어기는 일치를 버리고 정렬한다.
        입력:
            text: 용어와 같은 방식으로 정규화한 텍스트.
        출력: ``TermMatch`` 목록. 시작 위치 순, 같은 위치에서는 긴 것부터.
        """
        if not self._size or not text:
            return []
        found = []
        for last, (term, values) in self._automaton.iter(text):
            start, end = last - len(term) + 1, last + 1
            if self._touches_boundary(text, start, end):
                continue
            found.append(TermMatch(start, end, term, values))
        found.sort(key=lambda match: (match.start, match.start - match.end))
        return found

    def longest(self, text: str) -> list[TermMatch[V]]:
        """서로 겹치지 않는, 가장 왼쪽·가장 긴 일치들을 텍스트 순서로 돌려준다(긴 것 우선 정규식 대안과 같은 결과).

        역할: "와파린나트륨"을 "와파린" + 나머지 조각이 아니라 한 약으로 읽게 한다.
        입력:
            text: 정규화한 텍스트.
        출력: 겹치지 않는 ``TermMatch`` 목록(텍스트 순서).
        """
        chosen: list[TermMatch[V]] = []
        end = 0
        for match in self.matches(text):
            if match.start >= end:
                chosen.append(match)
                end = match.end
        return chosen

    def terms_in(self, text: str) -> set[str]:
        """텍스트에 나온 서로 다른 용어의 집합."""
        return {match.term for match in self.matches(text)}

    def _touches_boundary(self, text: str, start: int, end: int) -> bool:
        """일치 구간 바로 앞이나 뒤의 문자가 경계 문자이면 True(단어 일부로 보고 버린다)."""
        if self._boundary is None:
            return False
        before = text[start - 1] if start > 0 else ""
        after = text[end] if end < len(text) else ""
        return bool((before and self._boundary.match(before)) or (after and self._boundary.match(after)))
