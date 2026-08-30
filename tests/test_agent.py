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


if __name__ == "__main__":
    unittest.main()
