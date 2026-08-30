from __future__ import annotations

import unittest

from src.confidence.payload import ConfidencePayload
from src.output.followup import (
    FollowUpContext,
    build_all_missing_ask_message,
    build_ask_message,
    build_recommend_message,
)
from src.output.formatter import OutputFormatter


def _ctx(**overrides) -> FollowUpContext:
    defaults = dict(
        scenario="buying",
        n_constraints_known=1,
        exhausted=False,
        turn=2,
        override_seen=False,
    )
    defaults.update(overrides)
    return FollowUpContext(**defaults)


class BuildAskMessageTest(unittest.TestCase):
    def test_no_context_returns_generic_default(self) -> None:
        self.assertTrue(build_ask_message(None))

    def test_never_bundles_multiple_attribute_names_in_one_question(self) -> None:
        # One thing asked per turn, always -- never "color, size, material,
        # or budget?" in a single sentence, whether via a specific topic or
        # any situational fallback branch.
        keywords = ("color", "material", "size", "style", "brand", "budget")
        contexts = [
            _ctx(scenario="intent_override"),
            _ctx(scenario="boundary"),
            _ctx(scenario="browsing", n_constraints_known=0),
            _ctx(scenario="buying", n_constraints_known=2, turn=2),
            _ctx(scenario="buying", n_constraints_known=2, turn=9),
            _ctx(topic="color"),
            _ctx(scenario="boundary", topic="brand"),
            _ctx(scenario="intent_override", topic="size"),
        ]
        for context in contexts:
            msg = build_ask_message(context).lower()
            hits = [kw for kw in keywords if kw in msg]
            self.assertLessEqual(len(hits), 1, f"bundled multiple attributes in one message: {msg!r} ({hits})")

    def test_intent_override_gets_its_own_message(self) -> None:
        msg = build_ask_message(_ctx(scenario="intent_override"))
        default = build_ask_message(_ctx(scenario="buying"))
        self.assertNotEqual(msg, default)

    def test_boundary_gets_its_own_message(self) -> None:
        msg = build_ask_message(_ctx(scenario="boundary"))
        default = build_ask_message(_ctx(scenario="buying"))
        self.assertNotEqual(msg, default)

    def test_zero_info_gets_its_own_message(self) -> None:
        msg = build_ask_message(_ctx(scenario="browsing", n_constraints_known=0))
        default = build_ask_message(_ctx(scenario="browsing", n_constraints_known=2))
        self.assertNotEqual(msg, default)

    def test_late_turn_gets_its_own_message(self) -> None:
        msg = build_ask_message(_ctx(turn=8, n_constraints_known=3))
        default = build_ask_message(_ctx(turn=2, n_constraints_known=3))
        self.assertNotEqual(msg, default)

    def test_scenario_takes_priority_over_zero_info(self) -> None:
        # An intent override on turn 1 with nothing else known yet should
        # still get the override phrasing, not the zero-info phrasing --
        # the customer just told us something (a new intent), it's not vague.
        msg = build_ask_message(_ctx(scenario="intent_override", n_constraints_known=0))
        self.assertEqual(msg, build_ask_message(_ctx(scenario="intent_override", n_constraints_known=5)))

    def test_topic_gets_its_own_question(self) -> None:
        color_q = build_ask_message(_ctx(topic="color"))
        material_q = build_ask_message(_ctx(topic="material"))
        self.assertNotEqual(color_q, material_q)
        self.assertIn("color", color_q.lower())

    def test_topic_overrides_situational_default(self) -> None:
        with_topic = build_ask_message(_ctx(topic="color", n_constraints_known=2, turn=2))
        without_topic = build_ask_message(_ctx(topic=None, n_constraints_known=2, turn=2))
        self.assertNotEqual(with_topic, without_topic)

    def test_topic_still_layers_boundary_lead_in(self) -> None:
        msg = build_ask_message(_ctx(scenario="boundary", topic="size"))
        self.assertIn("judgment", msg.lower())
        self.assertIn("size", msg.lower())

    def test_topic_still_layers_override_lead_in(self) -> None:
        msg = build_ask_message(_ctx(scenario="intent_override", topic="brand"))
        self.assertIn("updating", msg.lower())
        self.assertIn("brand", msg.lower())

    def test_all_variants_are_distinct_strings(self) -> None:
        variants = {
            build_ask_message(_ctx(scenario="intent_override")),
            build_ask_message(_ctx(scenario="boundary")),
            build_ask_message(_ctx(scenario="browsing", n_constraints_known=0)),
            build_ask_message(_ctx(scenario="buying", n_constraints_known=2, turn=8)),
            build_ask_message(_ctx(scenario="buying", n_constraints_known=2, turn=2)),
        }
        self.assertEqual(len(variants), 5, "expected 5 distinct hardcoded messages")


class BuildRecommendMessageTest(unittest.TestCase):
    def test_no_context_returns_generic_default(self) -> None:
        self.assertTrue(build_recommend_message(None))

    def test_exhausted_gets_its_own_message(self) -> None:
        msg = build_recommend_message(_ctx(exhausted=True))
        default = build_recommend_message(_ctx(exhausted=False, turn=2))
        self.assertNotEqual(msg, default)

    def test_late_turn_not_exhausted_gets_its_own_message(self) -> None:
        msg = build_recommend_message(_ctx(exhausted=False, turn=9))
        default = build_recommend_message(_ctx(exhausted=False, turn=2))
        self.assertNotEqual(msg, default)

    def test_exhausted_takes_priority_over_late_turn(self) -> None:
        msg = build_recommend_message(_ctx(exhausted=True, turn=9))
        self.assertEqual(msg, build_recommend_message(_ctx(exhausted=True, turn=2)))


class OutputFormatterTest(unittest.TestCase):
    def setUp(self) -> None:
        self.formatter = OutputFormatter()

    def test_clarify_payload_without_context_uses_legacy_static_message(self) -> None:
        payload = ConfidencePayload(score=0.1, clarify=True, ask_attribute="other", reason="r")
        result = self.formatter.format(payload, ["B001"])
        self.assertEqual(result["message"], "Anything else that would help me narrow this down?")
        self.assertEqual(result["ask_attribute"], "other")

    def test_clarify_payload_with_context_uses_situational_message(self) -> None:
        # OutputFormatter calls build_all_missing_ask_message (the bundled
        # missing-attributes builder), not build_ask_message (the older,
        # still-available single-topic builder) -- see formatter.py.
        payload = ConfidencePayload(score=0.1, clarify=True, ask_attribute="other", reason="r")
        result = self.formatter.format(payload, ["B001"], context=_ctx(scenario="boundary"))
        self.assertEqual(result["message"], build_all_missing_ask_message(_ctx(scenario="boundary")))
        self.assertNotEqual(result["message"], "Anything else that would help me narrow this down?")

    def test_no_clarify_without_context_uses_legacy_static_message(self) -> None:
        payload = ConfidencePayload(score=0.9, clarify=False, ask_attribute=None, reason="r")
        result = self.formatter.format(payload, ["B001"])
        self.assertEqual(result["message"], "Here are the closest matches I found.")
        self.assertIsNone(result["ask_attribute"])

    def test_no_clarify_with_context_uses_situational_message(self) -> None:
        payload = ConfidencePayload(score=0.9, clarify=False, ask_attribute=None, reason="r")
        result = self.formatter.format(payload, ["B001"], context=_ctx(exhausted=True))
        self.assertEqual(result["message"], build_recommend_message(_ctx(exhausted=True)))
        self.assertIsNone(result["ask_attribute"])

    def test_contract_shape_unaffected_by_context(self) -> None:
        # message phrasing must never change ask_attribute or recommendations --
        # those are what the evaluator actually scores.
        payload = ConfidencePayload(score=0.1, clarify=True, ask_attribute="other", reason="r")
        without = self.formatter.format(payload, ["B001", "B002"])
        with_ctx = self.formatter.format(payload, ["B001", "B002"], context=_ctx())
        self.assertEqual(without["ask_attribute"], with_ctx["ask_attribute"])
        self.assertEqual(without["recommendations"], with_ctx["recommendations"])

    def test_recommendations_and_usage_shape_preserved(self) -> None:
        payload = ConfidencePayload(score=0.5, clarify=False, ask_attribute=None, reason="r")
        result = self.formatter.format(payload, ["B001", "B002"], usage={"prompt_tokens": 5, "completion_tokens": 2})
        self.assertEqual(result["recommendations"], [{"parent_asin": "B001"}, {"parent_asin": "B002"}])
        self.assertEqual(result["usage"], {"prompt_tokens": 5, "completion_tokens": 2})


if __name__ == "__main__":
    unittest.main()
