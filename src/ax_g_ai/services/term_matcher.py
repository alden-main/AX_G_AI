"""Find dictionary terms (drug, test, disease and concept names) in free text.

Every name lookup in a question, memo or prescription name goes through ``TermMatcher``: one
Aho–Corasick automaton (``pyahocorasick``) per dictionary, so a question is scanned once whatever
the dictionary size (tens of thousands of names), instead of testing each name in turn.

The caller normalizes both the terms and the text the same way (e.g. NFKC + lower case, or
whitespace removed for Korean). ``boundary`` lists the characters that may not touch a match on
either side, as a regex character class body: ``"a-z"`` keeps "aspirin" out of "aspirinate"; Korean
names are matched without a boundary because particles attach to them ("아스피린을").
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable
from typing import NamedTuple

import ahocorasick


class TermMatch[V](NamedTuple):
    start: int
    end: int  # exclusive
    term: str
    values: tuple[V, ...]  # every value added for this term, in the order added


class TermMatcher[V]:
    def __init__(self, terms: Iterable[tuple[str, V]], *, boundary: str | None = None) -> None:
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
        """Every occurrence of every term, overlapping ones included, ordered by start then longest first."""
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
        """Leftmost-longest matches that do not overlap, in text order (what a longest-first regex alternation finds)."""
        chosen: list[TermMatch[V]] = []
        end = 0
        for match in self.matches(text):
            if match.start >= end:
                chosen.append(match)
                end = match.end
        return chosen

    def terms_in(self, text: str) -> set[str]:
        """The distinct terms that occur in ``text``."""
        return {match.term for match in self.matches(text)}

    def _touches_boundary(self, text: str, start: int, end: int) -> bool:
        if self._boundary is None:
            return False
        before = text[start - 1] if start > 0 else ""
        after = text[end] if end < len(text) else ""
        return bool((before and self._boundary.match(before)) or (after and self._boundary.match(after)))
