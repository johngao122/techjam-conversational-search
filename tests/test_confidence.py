from __future__ import annotations

import math
import unittest

from src.confidence.confidence import compute_confidence
from src.confidence.fallback import safe_decide
from src.confidence.session_ledger import SessionLedger
from src.confidence.policy import (
    DEFAULT_THETA,
    TOPIC_PRIORITY,
    TURN_CUTOFF,
    decide,
    decide_specific_attribute,
    missing_topics,
    next_unasked_topic,
)
from src.ledger.ledger import ATTRIBUTE_PRIORITY
from src.reranker.types import RankResult


class FakeRanker:
    """Canned reranker returning a fixed RankResult (no retrieval dependency)."""

    def __init__(self, result: RankResult) -> None:
        self.result = result
        self.calls = 0

    def __call__(self) -> RankResult:
        self.calls += 1
        return self.result


def _rank(pool_size=100, max_coverage=1, crowd=10, ranked=None) -> RankResult:
    ranked = ranked if ranked is not None else [f"B{i:09d}" for i in range(10)]
    return RankResult(
        ranked=ranked,
        pool_size=pool_size,
        max_coverage=max_coverage,
        top_tier_crowd=crowd,
    )


class ConfidenceFunctionTest(unittest.TestCase):
    def test_score_in_unit_interval(self) -> None:
        score, _ = compute_confidence(_rank(max_coverage=4, crowd=1), 4)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_full_coverage_low_crowd_is_high(self) -> None:
        score, _ = compute_confidence(_rank(max_coverage=4, crowd=1), 4)
        self.assertGreater(score, DEFAULT_THETA)

    def test_high_crowd_lowers_confidence(self) -> None:
        low_crowd, _ = compute_confidence(_rank(max_coverage=2, crowd=1), 4)
        high_crowd, _ = compute_confidence(_rank(max_coverage=2, crowd=200), 4)
        self.assertGreater(low_crowd, high_crowd)

    def test_empty_pool_is_zero(self) -> None:
        score, reason = compute_confidence(_rank(pool_size=0), 3)
        self.assertEqual(score, 0.0)
        self.assertIn("no candidates", reason)


class PolicyMappingTest(unittest.TestCase):
    def test_zero_info_forces_clarify(self) -> None:
        ledger = SessionLedger("s", turn=1)
        payload = decide(_rank(max_coverage=4, crowd=1), ledger)
        self.assertTrue(payload.clarify)
        self.assertEqual(payload.ask_attribute, "other")

    def test_low_confidence_clarifies(self) -> None:
        ledger = SessionLedger("s", turn=2, constraints_known=["cotton", "black"])
        payload = decide(_rank(max_coverage=1, crowd=200), ledger, theta=0.9)
        self.assertTrue(payload.clarify)
        self.assertEqual(payload.ask_attribute, "other")

    def test_high_confidence_recommends_only(self) -> None:
        ledger = SessionLedger("s", turn=2, constraints_known=["cotton", "black", "size 10", "budget"])
        payload = decide(_rank(max_coverage=4, crowd=1), ledger, theta=0.3)
        self.assertFalse(payload.clarify)
        self.assertIsNone(payload.ask_attribute)


class EdgeCaseTest(unittest.TestCase):
    def test_override_resets_exhausted_and_resumes_clarify(self) -> None:
        ledger = SessionLedger("s", constraints_known=["cotton"])
        ledger.observe("I don't have an additional preference for color.", turn=2)
        self.assertTrue(ledger.exhausted)
        # Override arrives.
        ledger.observe("Actually, ignore my earlier preference. What I need is: leather.", turn=3)
        self.assertFalse(ledger.exhausted)
        self.assertTrue(ledger.override_seen)
        payload = decide(_rank(max_coverage=1, crowd=200), ledger, theta=0.9)
        self.assertTrue(payload.clarify)

    def test_boundary_brushoff_does_not_exhaust_and_clarify_continues(self) -> None:
        # Boundary brush-off ("no preference for THIS attribute; use your
        # judgment") must NOT latch exhaustion -- the customer may still have
        # other constraints, so clarification continues next turn.
        ledger = SessionLedger("s", turn=2, constraints_known=["cotton"])
        ledger.observe("I don't have a preference for color; please use your judgment.", turn=2)
        self.assertFalse(ledger.exhausted, "boundary brush-off must not exhaust")
        # Low-confidence rank -> policy should still clarify (not gated off).
        payload = decide(_rank(max_coverage=1, crowd=200), ledger, theta=0.9)
        self.assertTrue(payload.clarify)
        self.assertEqual(payload.ask_attribute, "other")

    def test_boundary_brushoff_does_not_count_as_no_progress(self) -> None:
        # A boundary brush-off is neutral: it should not advance no_progress_turns
        # (otherwise repeated brush-offs could trip the late-turn exhaustion rule).
        ledger = SessionLedger("s", turn=2, constraints_known=["cotton"])
        before = ledger.no_progress_turns
        ledger.observe("I don't have a preference for color; please use your judgment.", turn=2)
        self.assertEqual(ledger.no_progress_turns, before)

    def test_late_turn_no_progress_uses_shared_cutoff(self) -> None:
        # The late-turn exhaustion rule in the ledger must key off the same
        # TURN_CUTOFF the policy gate uses (single source of truth).
        ledger = SessionLedger("s", turn=2, constraints_known=["cotton"])
        ledger.observe("hmm", turn=TURN_CUTOFF)      # 1st no-progress at cutoff
        ledger.observe("still hmm", turn=TURN_CUTOFF) # 2nd -> exhausts
        self.assertTrue(ledger.exhausted)

    def test_exhausted_message_stops_clarify_forever(self) -> None:
        ledger = SessionLedger("s", constraints_known=["cotton", "black"])
        ledger.observe("I don't have an additional preference for material.", turn=4)
        self.assertTrue(ledger.exhausted)
        payload = decide(_rank(max_coverage=1, crowd=200), ledger, theta=0.9)
        self.assertFalse(payload.clarify)
        # Later turn, still exhausted.
        ledger.observe("still nothing", turn=5)
        self.assertTrue(ledger.exhausted)
        payload2 = decide(_rank(max_coverage=1, crowd=200), ledger, theta=0.9)
        self.assertFalse(payload2.clarify)

    def test_empty_pool_fallback_fires(self) -> None:
        ledger = SessionLedger("s", turn=1, constraints_known=["cotton"])
        fallback = [f"B{i:09d}" for i in range(10)]
        payload, recs = safe_decide(FakeRanker(_rank(pool_size=0, ranked=[])), ledger, fallback, DEFAULT_THETA)
        self.assertEqual(payload.score, 0.0)
        self.assertTrue(payload.clarify)
        self.assertEqual(recs, fallback)

    def test_exception_in_rank_no_raise(self) -> None:
        def boom() -> RankResult:
            raise ValueError("reranker exploded")

        ledger = SessionLedger("s", turn=1, constraints_known=["cotton"])
        fallback = [f"B{i:09d}" for i in range(10)]
        payload, recs = safe_decide(boom, ledger, fallback, DEFAULT_THETA)
        self.assertTrue(payload.clarify)
        self.assertEqual(recs, fallback)

    def test_determinism_identical_inputs(self) -> None:
        rank = _rank(max_coverage=2, crowd=37)
        a = decide(rank, SessionLedger("s", turn=3, constraints_known=["cotton", "black"]))
        b = decide(rank, SessionLedger("s", turn=3, constraints_known=["cotton", "black"]))
        self.assertEqual((a.score, a.clarify, a.ask_attribute), (b.score, b.clarify, b.ask_attribute))
        self.assertFalse(math.isnan(a.score))


class AttributeCyclePolicyTest(unittest.TestCase):
    """decide_specific_attribute: asks a specific, unasked attribute each
    turn and never repeats one already asked or already disclosed."""

    def test_first_turn_asks_first_priority_attribute(self) -> None:
        ledger = SessionLedger("s", turn=1)
        payload = decide_specific_attribute(_rank(), ledger)
        self.assertTrue(payload.clarify)
        self.assertEqual(payload.ask_attribute, ATTRIBUTE_PRIORITY[0])

    def test_does_not_repeat_an_already_asked_attribute(self) -> None:
        ledger = SessionLedger("s", turn=2)
        ledger.note_ask(ATTRIBUTE_PRIORITY[0])
        payload = decide_specific_attribute(_rank(), ledger)
        self.assertEqual(payload.ask_attribute, ATTRIBUTE_PRIORITY[1])
        self.assertNotEqual(payload.ask_attribute, ATTRIBUTE_PRIORITY[0])

    def test_no_preference_reply_still_marks_attribute_asked_and_moves_on(self) -> None:
        # The example scenario: agent asks "color", customer says "no
        # preference", that must not make the agent ask "color" again.
        ledger = SessionLedger("s", turn=1)
        first = decide_specific_attribute(_rank(), ledger)
        ledger.note_ask(first.ask_attribute)
        ledger.turn = 2
        ledger.observe(f"I don't have a preference for {first.ask_attribute}; please use your judgment.", turn=2)
        second = decide_specific_attribute(_rank(), ledger)
        self.assertNotEqual(second.ask_attribute, first.ask_attribute)

    def test_skips_attributes_already_disclosed_as_constraints(self) -> None:
        ledger = SessionLedger("s", turn=1)
        known = {ATTRIBUTE_PRIORITY[0], ATTRIBUTE_PRIORITY[1]}
        payload = decide_specific_attribute(_rank(), ledger, known_attrs=known)
        self.assertNotIn(payload.ask_attribute, known)

    def test_stops_once_every_attribute_covered(self) -> None:
        ledger = SessionLedger("s", turn=1)
        for attr in ATTRIBUTE_PRIORITY:
            ledger.note_ask(attr)
        payload = decide_specific_attribute(_rank(), ledger)
        self.assertFalse(payload.clarify)
        self.assertIsNone(payload.ask_attribute)

    def test_exhausted_stops_regardless_of_unasked_attributes(self) -> None:
        ledger = SessionLedger("s", turn=2)
        ledger.observe("I don't have an additional preference for color.", turn=2)
        self.assertTrue(ledger.exhausted)
        payload = decide_specific_attribute(_rank(), ledger)
        self.assertFalse(payload.clarify)

    def test_turn_cutoff_stops_regardless_of_unasked_attributes(self) -> None:
        ledger = SessionLedger("s", turn=TURN_CUTOFF)
        payload = decide_specific_attribute(_rank(), ledger)
        self.assertFalse(payload.clarify)

    def test_never_repeats_across_a_full_session_cycle(self) -> None:
        # Simulate a whole session: every ask must be a distinct attribute.
        ledger = SessionLedger("s", turn=1)
        asked: list[str] = []
        for turn in range(1, TURN_CUTOFF + 1):
            ledger.turn = turn
            payload = decide_specific_attribute(_rank(), ledger)
            if not payload.clarify:
                break
            self.assertNotIn(payload.ask_attribute, asked, "repeated an already-asked attribute")
            asked.append(payload.ask_attribute)
            ledger.note_ask(payload.ask_attribute)
        self.assertEqual(len(asked), len(set(asked)))

    def test_safe_decide_wires_attribute_cycle_policy(self) -> None:
        ledger = SessionLedger("s", turn=1)
        ranker = FakeRanker(_rank())
        payload, recs = safe_decide(
            ranker, ledger, [f"B{i:09d}" for i in range(10)],
            DEFAULT_THETA, policy="attribute_cycle",
        )
        self.assertEqual(payload.ask_attribute, ATTRIBUTE_PRIORITY[0])
        self.assertEqual(ranker.calls, 1)


class NextUnaskedTopicTest(unittest.TestCase):
    """next_unasked_topic: message-phrasing-only helper, safe to use under
    always_ask (never touches the real ask_attribute the contract returns).
    Ordered by TOPIC_PRIORITY (measured constraint-type frequency), not the
    ledger's ATTRIBUTE_PRIORITY."""

    def test_first_call_returns_category_first(self) -> None:
        # category is placed first regardless of frequency ranking -- see
        # TOPIC_PRIORITY's docstring (natural opener when genuinely unknown).
        ledger = SessionLedger("s", turn=1)
        self.assertEqual(next_unasked_topic(ledger), "category")

    def test_frequency_order_after_category(self) -> None:
        # feature(404) > material(302) > color(60) > style(19) > size(11) >
        # use_case(4) -- the team's measured constraint-type breakdown.
        ledger = SessionLedger("s", turn=1)
        ledger.note_ask("category")
        for expected in ("feature", "material", "color", "style", "size", "use_case"):
            self.assertEqual(next_unasked_topic(ledger), expected)
            ledger.note_ask(expected)

    def test_budget_and_brand_come_last(self) -> None:
        # budget "effectively never survives the card's 4-slot cut"; brand
        # was not observed at all -- both are lowest priority.
        ledger = SessionLedger("s", turn=1)
        for attr in ("category", "feature", "material", "color", "style", "size", "use_case"):
            ledger.note_ask(attr)
        self.assertEqual(next_unasked_topic(ledger), "budget")
        ledger.note_ask("budget")
        self.assertEqual(next_unasked_topic(ledger), "brand")

    def test_never_suggests_the_same_topic_twice(self) -> None:
        ledger = SessionLedger("s", turn=1)
        seen: set[str] = set()
        for _ in range(len(TOPIC_PRIORITY)):
            topic = next_unasked_topic(ledger)
            self.assertIsNotNone(topic)
            self.assertNotIn(topic, seen)
            seen.add(topic)
            ledger.note_ask(topic)

    def test_returns_none_once_every_specific_attribute_suggested(self) -> None:
        ledger = SessionLedger("s", turn=1)
        for attr in TOPIC_PRIORITY:
            ledger.note_ask(attr)
        self.assertIsNone(next_unasked_topic(ledger))

    def test_never_suggests_other_as_a_topic(self) -> None:
        # "other" is always in asked_attributes (from the real always_ask
        # payload) but must never itself be offered as a phrasing topic.
        ledger = SessionLedger("s", turn=1)
        ledger.note_ask("other")
        self.assertNotEqual(next_unasked_topic(ledger), "other")
        self.assertNotIn("other", TOPIC_PRIORITY)

    def test_skips_attributes_already_disclosed_as_constraints(self) -> None:
        ledger = SessionLedger("s", turn=1)
        known = {"category"}
        self.assertNotEqual(next_unasked_topic(ledger, known_attrs=known), "category")

    def test_example_scenario_no_preference_reply_does_not_repeat_topic(self) -> None:
        # The exact requested scenario: agent's message asks about a topic,
        # customer says no preference, that topic must not be suggested
        # again -- while the real ask_attribute stays "other" throughout
        # (always_ask never changes).
        ledger = SessionLedger("s", turn=1)
        topic = next_unasked_topic(ledger)
        self.assertEqual(topic, "category")
        ledger.note_ask(topic)  # message referenced this topic
        ledger.note_ask("other")  # the real (unchanged) ask_attribute
        ledger.observe(f"I don't have a preference for {topic}; please use your judgment.", turn=2)
        ledger.turn = 2
        next_topic = next_unasked_topic(ledger)
        self.assertNotEqual(next_topic, topic)


class MissingTopicsTest(unittest.TestCase):
    """missing_topics: message-phrasing-only helper, safe to use under
    always_ask (never touches the real ask_attribute the contract returns).
    Pure function of known_attrs only -- no ledger/session state -- ordered
    by TOPIC_PRIORITY (measured constraint-type frequency), not the ledger's
    ATTRIBUTE_PRIORITY."""

    def test_nothing_known_returns_everything_in_priority_order(self) -> None:
        self.assertEqual(missing_topics(), list(TOPIC_PRIORITY))

    def test_priority_order_matches_measured_frequency(self) -> None:
        # category first (natural opener), then feature(404) > material(302)
        # > color(60) > style(19) > size(11) > use_case(4) > budget > brand
        # -- the team's measured constraint-type breakdown.
        self.assertEqual(
            missing_topics(),
            ["category", "feature", "material", "color", "style", "size", "use_case", "budget", "brand"],
        )

    def test_known_attributes_are_excluded(self) -> None:
        known = {"category", "color"}
        result = missing_topics(known_attrs=known)
        self.assertNotIn("category", result)
        self.assertNotIn("color", result)
        self.assertEqual(len(result), len(TOPIC_PRIORITY) - 2)

    def test_shrinks_as_more_becomes_known(self) -> None:
        known: set[str] = set()
        previous_len = len(missing_topics(known_attrs=known))
        for attr in TOPIC_PRIORITY:
            known.add(attr)
            current_len = len(missing_topics(known_attrs=known))
            self.assertLess(current_len, previous_len)
            previous_len = current_len

    def test_empty_once_everything_known(self) -> None:
        known = set(TOPIC_PRIORITY)
        self.assertEqual(missing_topics(known_attrs=known), [])

    def test_never_includes_other(self) -> None:
        self.assertNotIn("other", missing_topics())
        self.assertNotIn("other", TOPIC_PRIORITY)

    def test_a_missing_attribute_stays_missing_across_calls_until_known(self) -> None:
        # No per-session "already asked" state: an attribute the customer
        # hasn't revealed keeps showing up in the bundle every turn (unlike
        # the old one-shot next_unasked_topic), until it's actually known.
        known: set[str] = set()
        first = missing_topics(known_attrs=known)
        second = missing_topics(known_attrs=known)  # simulates a later turn, nothing new disclosed
        self.assertEqual(first, second)
        self.assertIn("color", second)


if __name__ == "__main__":
    unittest.main()
