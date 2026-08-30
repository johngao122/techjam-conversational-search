from __future__ import annotations

import unittest

from src.agent import Agent


class TestUnionSearchKey(unittest.TestCase):
    """_union_search_key: text fields accumulate (union), numeric fields update."""

    def test_text_fields_union_accumulates_values(self) -> None:
        prev = {"material": ["cotton"], "color": ["black"]}
        curr = {"material": ["cotton", "linen"], "style": ["casual"]}
        merged = Agent._union_search_key(prev, curr)
        # material accumulates, de-duped, order preserved
        self.assertEqual(merged["material"], ["cotton", "linen"])
        # field only in previous is retained
        self.assertEqual(merged["color"], ["black"])
        # field only in current is added
        self.assertEqual(merged["style"], ["casual"])

    def test_text_union_dedupes_and_preserves_order(self) -> None:
        prev = {"type": ["jacket", "coat"]}
        curr = {"type": ["coat", "parka"]}
        merged = Agent._union_search_key(prev, curr)
        self.assertEqual(merged["type"], ["jacket", "coat", "parka"])

    def test_numeric_field_updates_to_current(self) -> None:
        prev = {"price": [{"lte": 50.0}]}
        curr = {"price": [{"lte": 30.0}]}
        merged = Agent._union_search_key(prev, curr)
        # latest budget wins (update, not union)
        self.assertEqual(merged["price"], [{"lte": 30.0}])

    def test_numeric_field_kept_when_absent_in_current(self) -> None:
        prev = {"price": [{"lte": 50.0}], "material": ["wool"]}
        curr = {"material": ["wool"]}
        merged = Agent._union_search_key(prev, curr)
        self.assertEqual(merged["price"], [{"lte": 50.0}])
        self.assertEqual(merged["material"], ["wool"])

    def test_numeric_field_added_from_current(self) -> None:
        prev = {"material": ["wool"]}
        curr = {"material": ["wool"], "price": [{"gte": 20.0}]}
        merged = Agent._union_search_key(prev, curr)
        self.assertEqual(merged["price"], [{"gte": 20.0}])

    def test_empty_inputs(self) -> None:
        self.assertEqual(Agent._union_search_key({}, {}), {})
        self.assertEqual(Agent._union_search_key({}, {"color": ["red"]}), {"color": ["red"]})
        self.assertEqual(Agent._union_search_key({"color": ["red"]}, {}), {"color": ["red"]})

    def test_mixed_text_and_numeric(self) -> None:
        prev = {"material": ["cotton"], "price": [{"lte": 40.0}]}
        curr = {"material": ["cotton", "denim"], "color": ["blue"], "price": [{"lte": 25.0}]}
        merged = Agent._union_search_key(prev, curr)
        self.assertEqual(merged["material"], ["cotton", "denim"])
        self.assertEqual(merged["color"], ["blue"])
        self.assertEqual(merged["price"], [{"lte": 25.0}])
_PROFILE = {
    "purchase_frequency": "3-4 prior purchases",
    "average_prior_rating": 4.2,
    "rating_style": "usually positive",
    "preference_tags": ["comfort", "fit"],
    "summary": "Prior purchases emphasize comfort and fit.",
}


class FollowUpMissingAttributesTest(unittest.TestCase):
    """A provided attribute must never keep showing up in the follow-up
    question as still-missing."""

    @classmethod
    def setUpClass(cls) -> None:
        # Building the agent (FTS5 index + catalog load) is the same
        # regardless of test case; build once per class, not per test.
        cls.agent = Agent()

    def setUp(self) -> None:
        self.agent = self.__class__.agent

    def test_dollar_sign_less_budget_is_recognised_as_known(self) -> None:
        # Regression: _parse_price_constraint (agent.py's own price regex,
        # used for search filtering) accepts "under 50" with no "$", but
        # extract_attributes()'s own budget regex requires a literal "$" --
        # so the two disagreed on whether budget was "known", and the
        # follow-up question kept asking about a budget the customer had
        # already given. Fixed by also counting price_constraint as known.
        session_id = "budget-regression"
        self.agent.reset(session_id, dict(_PROFILE))
        first = self.agent.respond(session_id, "i want a shirt", 1, 10)
        self.assertIn("budget", first["message"].lower())

        second = self.agent.respond(session_id, "under 50", 2, 10)
        self.assertNotIn("budget", second["message"].lower())

    def test_dollar_sign_budget_still_recognised_as_known(self) -> None:
        # Non-regression: the ordinary "$" phrasing must keep working too.
        session_id = "budget-dollar-sign"
        self.agent.reset(session_id, dict(_PROFILE))
        self.agent.respond(session_id, "i want a shirt", 1, 10)
        second = self.agent.respond(session_id, "under $50", 2, 10)
        self.assertNotIn("budget", second["message"].lower())

    def test_provided_material_and_color_drop_out_of_the_question(self) -> None:
        session_id = "material-color-regression"
        self.agent.reset(session_id, dict(_PROFILE))
        first = self.agent.respond(session_id, "i want a shirt", 1, 10)
        self.assertIn("material", first["message"].lower())
        self.assertIn("color", first["message"].lower())

        second = self.agent.respond(session_id, "black cotton", 2, 10)
        self.assertNotIn("material", second["message"].lower())
        self.assertNotIn("color", second["message"].lower())


if __name__ == "__main__":
    unittest.main()
