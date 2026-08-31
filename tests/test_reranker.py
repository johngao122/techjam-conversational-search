from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.reranker import Reranker, build_reranker, default_query
from src.reranker import rank as rank_module
from src.reranker.coverage import Product, covers, coverage_count
from src.reranker.rank import _hydrate_products
from src.catalog.catalog import Catalog
from src.retrieval.retrieval import Retriever


def _catalog_file(rows: list[dict]) -> str:
    directory = tempfile.mkdtemp()
    path = Path(directory) / "catalog.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return str(path)


ROWS = [
    {
        "parent_asin": "B000000001",
        "title": "Blue cotton running shoe",
        "features": ["cotton", "breathable"],
        "details": {"department": "womens"},
        "description": ["comfortable running shoe"],
        "categories": ["Clothing", "Shoes"],
        "store": "Acme",
        "average_rating": 4.5,
        "rating_number": 100,
        "price": 40.0,
    },
    {
        "parent_asin": "B000000002",
        "title": "Blue cotton casual shoe",
        "features": ["cotton"],
        "details": {"department": "womens"},
        "description": ["casual shoe"],
        "categories": ["Clothing", "Shoes"],
        "store": "Acme",
        "average_rating": 4.0,
        "rating_number": 500,
        "price": 35.0,
    },
    {
        "parent_asin": "B000000003",
        "title": "Red leather boot",
        "features": ["leather"],
        "details": {"department": "mens"},
        "description": ["winter boot"],
        "categories": ["Clothing", "Boots"],
        "store": "BootCo",
        "average_rating": 4.8,
        "rating_number": 20,
        "price": 120.0,
    },
]


class CoverageTest(unittest.TestCase):
    def setUp(self) -> None:
        self.cat = Catalog(_catalog_file(ROWS))
        self.products = _hydrate_products(
            self.cat, ["B000000001", "B000000002", "B000000003"]
        )

    def test_single_token_match(self) -> None:
        p = self.products["B000000001"]
        self.assertTrue(covers(p, "cotton"))
        self.assertFalse(covers(p, "leather"))

    def test_color_label_prefix_stripped(self) -> None:
        p = self.products["B000000001"]
        self.assertTrue(covers(p, "color: blue"))

    def test_budget_numeric_within_tolerance(self) -> None:
        cheap = self.products["B000000002"]  # $35
        pricey = self.products["B000000003"]  # $120
        self.assertTrue(covers(cheap, "budget around $40"))
        self.assertFalse(covers(pricey, "budget around $40"))

    def test_coverage_count(self) -> None:
        p = self.products["B000000001"]
        n = coverage_count(p, ["cotton", "color: blue", "leather"])
        self.assertEqual(n, 2)

    # --- Phase 1: content-token matching fidelity ---

    def test_percentage_noise_ignored(self) -> None:
        # "100% Polyester": the numeric "100" must not block matching on the
        # content token "polyester".
        p = Product("Z", "made of polyester, imported", None, 5, 4.0)
        self.assertTrue(covers(p, "100% Polyester"))

    def test_long_soft_preference_matches_on_content(self) -> None:
        # A long marketing sentence should match when its distinctive content
        # tokens are present in the product text.
        p = Product("Z", "high quality mesh for maximum breathability to keep you cool", None, 5, 4.0)
        self.assertTrue(covers(p, "High quality mesh for maximum breathability to keep you cool"))

    def test_long_soft_preference_not_matched_when_content_absent(self) -> None:
        p = Product("Z", "a plain leather boot", None, 5, 4.0)
        self.assertFalse(covers(p, "High quality mesh for maximum breathability to keep you cool"))

    def test_size_label_token_stripped(self) -> None:
        # "size 10": bare "size" is a structural label (noise); "10" is numeric
        # noise. With no content tokens the constraint carries no signal.
        p = Product("Z", "shirt size 10 cotton", None, 5, 4.0)
        self.assertFalse(covers(p, "size"))

    def test_budget_variants(self) -> None:
        cheap = Product("C", "shirt", 30.0, 5, 4.0)
        pricey = Product("P", "shirt", 120.0, 5, 4.0)
        for phrase in ("under $40", "under 40", "less than $40", "<= 40", "40 dollars", "budget 40"):
            self.assertTrue(covers(cheap, phrase), f"cheap should satisfy {phrase!r}")
            self.assertFalse(covers(pricey, phrase), f"pricey should not satisfy {phrase!r}")

    def test_budget_no_price_not_covered(self) -> None:
        p = Product("Z", "shirt", None, 5, 4.0)
        self.assertFalse(covers(p, "under $40"))


class RankResultTest(unittest.TestCase):
    def setUp(self) -> None:
        self.rr = build_reranker(_catalog_file(ROWS))

    def test_coverage_orders_top(self) -> None:
        res = self.rr.rank(default_query(["cotton", "color: blue"]), ["cotton", "color: blue"])
        # Both cotton+blue shoes cover 2; leather boot covers 0.
        self.assertEqual(res.max_coverage, 2)
        self.assertEqual(res.top_tier_crowd, 2)
        self.assertIn("B000000001", res.ranked[:2])
        self.assertIn("B000000002", res.ranked[:2])

    def test_pool_size_and_fields(self) -> None:
        res = self.rr.rank("shoe", ["cotton"])
        self.assertGreater(res.pool_size, 0)
        self.assertGreaterEqual(res.top_tier_crowd, 1)
        self.assertLessEqual(len(res.ranked), 10)

    def test_empty_query_empty_result(self) -> None:
        res = self.rr.rank("", [])
        self.assertEqual(res.pool_size, 0)
        self.assertEqual(res.ranked, [])
        self.assertEqual(res.max_coverage, 0)

    def test_determinism(self) -> None:
        a = self.rr.rank(default_query(["cotton"]), ["cotton"])
        b = self.rr.rank(default_query(["cotton"]), ["cotton"])
        self.assertEqual(a, b)

    def test_tie_break_by_rating_number(self) -> None:
        # cotton only: both shoes cover 1; B...02 has more ratings -> ranks first.
        res = self.rr.rank(default_query(["cotton"]), ["cotton"])
        cotton_ids = [r for r in res.ranked if r in ("B000000001", "B000000002")]
        self.assertEqual(cotton_ids[0], "B000000002")


# Catalog designed so that pure keyword retrieval and constraint-coverage
# disagree: the "distractor" is popular and keyword-heavy (retrieval favors it),
# but only the "target" actually satisfies the full constraint set. A correct
# reranker must promote the target as context accumulates.
CONTEXT_ROWS = [
    {
        "parent_asin": "DISTRACT001",
        "title": "shirt shirt shirt cotton comfortable popular",
        "features": ["cotton"],
        "details": {"department": "mens"},
        "description": ["a very popular cotton shirt everyone buys"],
        "categories": ["Clothing", "Shirts"],
        "store": "BigBrand",
        "average_rating": 4.9,
        "rating_number": 100000,  # hugely popular -> wins ties / retrieval bias
        "price": 20.0,
    },
    {
        "parent_asin": "TARGET0001",
        "title": "shirt cotton",
        "features": ["cotton", "blue", "long sleeve"],
        "details": {"department": "mens"},
        "description": ["blue long sleeve cotton shirt"],
        "categories": ["Clothing", "Shirts"],
        "store": "SmallBrand",
        "average_rating": 4.0,
        "rating_number": 5,  # unpopular -> loses every tiebreak
        "price": 45.0,
    },
    {
        "parent_asin": "NOISE00001",
        "title": "shirt cotton red short sleeve",
        "features": ["cotton", "red"],
        "details": {"department": "womens"},
        "description": ["red short sleeve cotton shirt"],
        "categories": ["Clothing", "Shirts"],
        "store": "MidBrand",
        "average_rating": 4.5,
        "rating_number": 500,
        "price": 30.0,
    },
]


class ContextualRerankTest(unittest.TestCase):
    """The reranker must reorder as contextual constraints accumulate."""

    def setUp(self) -> None:
        self.rr = build_reranker(_catalog_file(CONTEXT_ROWS))

    def test_context_promotes_target_to_top(self) -> None:
        # With only "cotton", all three products cover exactly 1 constraint, so
        # they sit in the same coverage tier (order decided by retrieval/rating).
        base = self.rr.rank(default_query(["cotton"]), ["cotton"])
        self.assertEqual(base.max_coverage, 1)
        self.assertEqual(base.top_tier_crowd, 3)

        # Add context: blue + long sleeve. Only the target covers all three, so
        # coverage-first reranking MUST put it at #1 regardless of popularity.
        ctx = self.rr.rank(
            default_query(["cotton", "blue", "long sleeve"]),
            ["cotton", "blue", "long sleeve"],
        )
        self.assertEqual(ctx.ranked[0], "TARGET0001",
                         "target must be #1 once context distinguishes it")
        self.assertEqual(ctx.max_coverage, 3)

    def test_coverage_beats_popularity(self) -> None:
        # DISTRACT001 is far more popular (100k ratings) but only covers cotton.
        # TARGET0001 (5 ratings) covers cotton+blue. Coverage must win over
        # popularity: the less popular but better-matching product ranks higher.
        res = self.rr.rank(default_query(["cotton", "blue"]), ["cotton", "blue"])
        self.assertEqual(res.ranked[0], "TARGET0001")
        self.assertGreater(
            res.ranked.index("DISTRACT001"), res.ranked.index("TARGET0001")
        )

    def test_crowd_shrinks_as_context_accumulates(self) -> None:
        c1 = self.rr.rank(default_query(["cotton"]), ["cotton"])
        c2 = self.rr.rank(default_query(["cotton", "blue"]), ["cotton", "blue"])
        c3 = self.rr.rank(
            default_query(["cotton", "blue", "long sleeve"]),
            ["cotton", "blue", "long sleeve"],
        )
        # Top-tier crowd should monotonically shrink (ambiguity decreasing).
        self.assertGreaterEqual(c1.top_tier_crowd, c2.top_tier_crowd)
        self.assertGreaterEqual(c2.top_tier_crowd, c3.top_tier_crowd)
        self.assertEqual(c3.top_tier_crowd, 1)  # target uniquely at max coverage

    def test_max_coverage_rises_with_context(self) -> None:
        c1 = self.rr.rank(default_query(["cotton"]), ["cotton"])
        c3 = self.rr.rank(
            default_query(["cotton", "blue", "long sleeve"]),
            ["cotton", "blue", "long sleeve"],
        )
        self.assertLess(c1.max_coverage, c3.max_coverage)

    def test_irrelevant_constraint_does_not_promote(self) -> None:
        # A constraint no product satisfies must not change the coverage tier.
        res = self.rr.rank(
            default_query(["cotton", "waterproof"]),
            ["cotton", "waterproof"],
        )
        # "waterproof" matches nothing -> max coverage stays at 1 (cotton).
        self.assertEqual(res.max_coverage, 1)

    def test_more_specific_context_narrows_to_correct_product(self) -> None:
        # "red" should surface the red shirt, not the blue target.
        res = self.rr.rank(default_query(["cotton", "red"]), ["cotton", "red"])
        self.assertEqual(res.ranked[0], "NOISE00001")
        self.assertEqual(res.max_coverage, 2)


# Both rows satisfy the plain "jacket" content-token constraint equally (the
# distractor's title/description mention "jacket" incidentally), but only
# JACKET0001 is structurally categorized as "Jackets" -- the exact vest-vs-
# jacket problem CATEGORY_MATCH_BONUS exists to nudge.
CATEGORY_BONUS_ROWS = [
    {
        "parent_asin": "JACKET0001",
        "title": "Outdoor Coat",
        "features": ["water resistant"],
        "details": {},
        "description": ["warm outer layer, great jacket for winter"],
        "categories": ["Clothing, Shoes & Jewelry, Jackets"],
        "store": "Acme",
        "average_rating": 4.0,
        "rating_number": 50,
        "price": 60.0,
    },
    {
        "parent_asin": "VEST0001",
        "title": "Jacket-style Puffer Vest",
        "features": ["jacket look", "sleeveless"],
        "details": {},
        "description": ["pairs well with any jacket"],
        "categories": ["Clothing, Shoes & Jewelry, Vests"],
        "store": "Acme",
        "average_rating": 4.0,
        "rating_number": 50,
        "price": 55.0,
    },
]


class CategoryBonusTest(unittest.TestCase):
    """CATEGORY_MATCH_BONUS nudges a structurally-categorized match above a
    text-overlapping but wrong-category distractor, without acting as a hard
    filter -- both rows still satisfy the "jacket" content-token constraint,
    the bonus only breaks the tie."""

    def setUp(self) -> None:
        self.rr = build_reranker(_catalog_file(CATEGORY_BONUS_ROWS))
        self._prev_bonus = rank_module.CATEGORY_MATCH_BONUS

    def tearDown(self) -> None:
        rank_module.CATEGORY_MATCH_BONUS = self._prev_bonus

    def test_disabled_keeps_retrieval_order(self) -> None:
        rank_module.CATEGORY_MATCH_BONUS = 0.0
        # Distractor listed first in the candidate pool; equal coverage (both
        # match "jacket") and equal rating mean retrieval order alone decides
        # with the bonus off, so the distractor stays on top.
        res = self.rr.score_by_coverage(
            ["VEST0001", "JACKET0001"], ["jacket"], category_constraints=["jackets"],
        )
        self.assertEqual(res.ranked[0], "VEST0001")

    def test_enabled_promotes_correct_category(self) -> None:
        rank_module.CATEGORY_MATCH_BONUS = 0.5
        res = self.rr.score_by_coverage(
            ["VEST0001", "JACKET0001"], ["jacket"], category_constraints=["jackets"],
        )
        self.assertEqual(res.ranked[0], "JACKET0001")

    def test_no_category_constraint_is_a_no_op(self) -> None:
        rank_module.CATEGORY_MATCH_BONUS = 0.5
        with_none = self.rr.score_by_coverage(["VEST0001", "JACKET0001"], ["jacket"])
        with_empty = self.rr.score_by_coverage(
            ["VEST0001", "JACKET0001"], ["jacket"], category_constraints=[],
        )
        self.assertEqual(with_none.ranked, with_empty.ranked)
        self.assertEqual(with_none.ranked[0], "VEST0001")


if __name__ == "__main__":
    unittest.main()
