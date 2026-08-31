from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.reranker import build_reranker
from src.retrieval.buckets import BucketIndex, fragment_type_tokens


def _catalog_file(rows: list[dict]) -> str:
    directory = tempfile.mkdtemp()
    path = Path(directory) / "catalog.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return str(path)


ROWS = [
    {
        "parent_asin": "DRESS0001",
        "title": "Blue satin cocktail dress",
        "features": ["satin", "sleeveless"],
        "details": {"department": "womens"},
        "description": ["a flattering blue satin dress"],
        "categories": ["Clothing, Shoes & Jewelry", "Women", "Clothing", "Dresses"],
        "store": "DressCo",
        "average_rating": 4.5,
        "rating_number": 200,
        "price": 89.0,
    },
    {
        "parent_asin": "DRESS0002",
        "title": "Green floral summer dress",
        "features": ["floral", "cotton"],
        "details": {"department": "womens"},
        "description": ["a light summer dress"],
        "categories": ["Clothing, Shoes & Jewelry", "Women", "Clothing", "Dresses"],
        "store": "DressCo",
        "average_rating": 4.2,
        "rating_number": 80,
        "price": 45.0,
    },
    {
        "parent_asin": "HEEL00001",
        "title": "Blue satin high heel",
        "features": ["satin", "3 inch heel"],
        "details": {"department": "womens"},
        "description": ["an elegant blue satin heel"],
        "categories": ["Clothing, Shoes & Jewelry", "Women", "Shoes", "Heels"],
        "store": "ShoeCo",
        "average_rating": 4.6,
        "rating_number": 500,
        "price": 60.0,
    },
    {
        "parent_asin": "JEANS0001",
        "title": "Slim fit blue jean",
        "features": ["denim"],
        "details": {"department": "mens"},
        "description": ["a classic pair of jeans"],
        "categories": ["Clothing, Shoes & Jewelry", "Men", "Clothing", "Jeans"],
        "store": "DenimCo",
        "average_rating": 4.1,
        "rating_number": 300,
        "price": 55.0,
    },
]


class BucketResolveTest(unittest.TestCase):
    def setUp(self) -> None:
        self.index = BucketIndex(ROWS)

    def test_singular_query_resolves_plural_bucket(self) -> None:
        key, how = self.index.resolve("i want a blue satin dress")
        self.assertIsNotNone(key, "bucket resolution should not fall back to None")
        self.assertIn("overlap", how)
        pool = self.index.get(key)
        self.assertIn("DRESS0001", pool)
        self.assertIn("DRESS0002", pool)
        self.assertNotIn("HEEL00001", pool)
        self.assertNotIn("JEANS0001", pool)

    def test_another_plural_noun_resolves(self) -> None:
        key, how = self.index.resolve("i want a blue jean")
        self.assertIsNotNone(key)
        pool = self.index.get(key)
        self.assertIn("JEANS0001", pool)
        self.assertNotIn("DRESS0001", pool)

    def test_gibberish_still_unresolved(self) -> None:
        key, how = self.index.resolve("xk qz flarn blorp")
        self.assertIsNone(key)
        self.assertEqual(how, "unresolved")


# Reproduces the real catalog shape that exposed the containment tie-break
# bug: a pure "Jackets" leaf category (short bucket key) vs. a combined
# "Coats, Jackets & Vests" -> "Vests" leaf category, whose coarse_category
# ("jackets & vests vests") is a much longer string that happens to contain
# the word "jackets" as a raw substring.
JACKET_VEST_ROWS = [
    {
        "parent_asin": "JACKET0001",
        "title": "Men's Cycling Jacket",
        "features": ["windproof"],
        "details": {"department": "mens"},
        "description": ["a warm cycling jacket"],
        "categories": ["Clothing, Shoes & Jewelry", "Cycling", "Men", "Jackets"],
        "store": "OuterCo",
        "average_rating": 4.3,
        "rating_number": 150,
        "price": 90.0,
    },
    {
        "parent_asin": "VEST0001",
        "title": "Women's Puffer Vest",
        "features": ["sleeveless"],
        "details": {"department": "womens"},
        "description": ["a puffer vest"],
        "categories": [
            "Clothing, Shoes & Jewelry", "Women", "Clothing",
            "Coats, Jackets & Vests", "Vests",
        ],
        "store": "VestCo",
        "average_rating": 4.1,
        "rating_number": 900,
        "price": 55.0,
    },
]


class BucketContainmentTieBreakTest(unittest.TestCase):
    """Regression: a short fragment like "jackets" must resolve to the
    category-pure bucket, not a longer combined-category bucket that merely
    contains the word as a substring (e.g. a vest bucket whose coarse
    category is "Coats, Jackets & Vests" -> "jackets & vests vests")."""

    def setUp(self) -> None:
        self.index = BucketIndex(JACKET_VEST_ROWS)

    def test_short_fragment_resolves_to_pure_bucket_not_longest_containing_key(self) -> None:
        key, how = self.index.resolve("I'm looking for jackets, but I'm still exploring.")
        self.assertEqual(how, "containment-reverse")
        pool = self.index.get(key)
        self.assertIn("JACKET0001", pool)
        self.assertNotIn("VEST0001", pool)


class FragmentTypeTokensTest(unittest.TestCase):
    def test_extracts_singularized_type_words(self) -> None:
        tokens = fragment_type_tokens("i want a blue satin dress")
        self.assertIn("dress", tokens)
        self.assertIn("blue", tokens)
        self.assertIn("satin", tokens)

    def test_empty_message(self) -> None:
        self.assertEqual(fragment_type_tokens(""), set())


# A dress and a heel filed under a made-up, unresolvable category taxonomy
# (so bucket resolution genuinely fails for both) but whose titles still
# plainly say what they are -- isolates the rung-3 title-relevance gate from
# bucket resolution behavior.
UNBUCKETABLE_ROWS = [
    {
        "parent_asin": "DRESS0001",
        "title": "Blue satin cocktail dress",
        "features": ["satin", "sleeveless"],
        "details": {"department": "womens"},
        "description": ["a flattering blue satin dress"],
        "categories": ["Clothing, Shoes & Jewelry", "Miscellaneous Apparel"],
        "store": "DressCo",
        "average_rating": 4.5,
        "rating_number": 200,
        "price": 89.0,
    },
    {
        "parent_asin": "HEEL00001",
        "title": "Blue satin high heel",
        "features": ["satin", "3 inch heel"],
        "details": {"department": "womens"},
        "description": ["an elegant blue satin heel"],
        "categories": ["Clothing, Shoes & Jewelry", "Miscellaneous Apparel"],
        "store": "ShoeCo",
        "average_rating": 4.6,
        "rating_number": 500,
        "price": 60.0,
    },
]


class TitleRelevanceGateTest(unittest.TestCase):
    """Covers the whole-catalog fallback's title-relevance gate end to end."""

    def setUp(self) -> None:
        self.rr = build_reranker(_catalog_file(ROWS))

    def test_gate_excludes_off_type_matches_on_unresolved_bucket(self) -> None:
        # Both rows share the same made-up category, so bucket resolution
        # can't place either -- confirm that first, then confirm rung 3's
        # gate still keeps the heel out of a dress query on head-noun alone.
        unbucketable = build_reranker(_catalog_file(UNBUCKETABLE_ROWS))
        key, _ = unbucketable.bucket_index.resolve("i want a blue satin dress")
        self.assertIsNone(key, "fixture should be genuinely unresolvable")

        res = unbucketable.rank_bucket(
            opening_message="i want a blue satin dress",
            constraints=[],
            transcript="i want a blue satin dress. budget around $500",
        )
        self.assertIn("DRESS0001", res.ranked)
        self.assertNotIn("HEEL00001", res.ranked, "heel must not leak into a dress query")

    def test_relevant_ids_prefix_matches_plural(self) -> None:
        ids = self.rr.retriever.title_relevant_ids({"dress"})
        self.assertIn("DRESS0001", ids)
        self.assertIn("DRESS0002", ids)
        self.assertNotIn("HEEL00001", ids)

    def test_relevant_ids_empty_terms_returns_empty(self) -> None:
        self.assertEqual(self.rr.retriever.title_relevant_ids(set()), set())


if __name__ == "__main__":
    unittest.main()
