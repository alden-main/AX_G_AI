from __future__ import annotations

import unittest

from ax_g_ai.services.term_matcher import TermMatch, TermMatcher


class TermMatcherTest(unittest.TestCase):
    def test_all_occurrences_overlapping_included_ordered_by_start_then_longest(self) -> None:
        matcher = TermMatcher([("와파린", "w"), ("와파린나트륨", "ws"), ("나트륨", "na")])
        self.assertEqual(
            [(match.start, match.term) for match in matcher.matches("와파린나트륨과 와파린")],
            [(0, "와파린나트륨"), (0, "와파린"), (3, "나트륨"), (8, "와파린")],
        )

    def test_longest_is_leftmost_longest_without_overlap(self) -> None:
        matcher = TermMatcher([("와파린", "w"), ("와파린나트륨", "ws"), ("나트륨", "na")])
        self.assertEqual([match.term for match in matcher.longest("와파린나트륨과 와파린")], ["와파린나트륨", "와파린"])

    def test_boundary_rejects_a_match_inside_a_word_and_falls_back_to_a_shorter_term(self) -> None:
        matcher = TermMatcher([("aspirin", "a"), ("war", "w"), ("warfarin", "x")], boundary="a-z")
        self.assertEqual(matcher.terms_in("aspirinate, warfarins"), set())
        self.assertEqual(matcher.terms_in("aspirin500mg (warfarin)"), {"aspirin", "warfarin"})
        self.assertEqual([match.term for match in matcher.longest("war-warfarin")], ["war", "warfarin"])

    def test_values_of_a_term_added_twice_are_kept_in_order(self) -> None:
        matcher = TermMatcher([("aspirin", "a"), ("aspirin", "b")])
        self.assertEqual(matcher.matches("aspirin"), [TermMatch(0, 7, "aspirin", ("a", "b"))])
        self.assertEqual(len(matcher), 1)

    def test_empty_dictionary_text_and_terms(self) -> None:
        self.assertEqual(TermMatcher([]).matches("anything"), [])
        self.assertEqual(TermMatcher([("", "x"), ("a", "y")]).matches(""), [])
        self.assertEqual(len(TermMatcher([("", "x")])), 0)


if __name__ == "__main__":
    unittest.main()
