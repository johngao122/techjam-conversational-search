"""Reranker core: score a retrieved candidate pool -> RankResult.

Public API:
    build_reranker(catalog_path) -> Reranker
    Reranker.rank(query, constraints, top_k) -> RankResult
    Reranker.rank_bucket(opening_message, constraints, top_k, transcript) -> RankResult
    default_query(constraints) -> str        # helper to build a query from constraints

Single responsibility: the ``Reranker`` *scores and orders an already-produced
candidate pool* and assembles the internals the confidence check needs
(``max_coverage`` / ``top_tier_crowd``). Producing the pool -- and choosing
which of the retrieval methods to use -- is delegated to the retrieval
strategies (:mod:`src.retrieval.strategies`), which the reranker consumes
through the uniform :class:`~src.retrieval.base.RetrievalResult` boundary.

Two scoring cores live here:
  * ``rank``        -- coverage-based (legacy BM25 path)
  * ``rank_bucket`` -- verbatim-constraint-based (bucket path)
"""

from __future__ import annotations

import os
from collections import OrderedDict
from functools import lru_cache

from src.catalog.catalog import Catalog
from src.catalog.loader import load_catalog_rows
from src.message_parser.catalog_vocab import _normalize
from src.reranker.coverage import Product, compile_constraints
from src.retrieval.base import RetrievalRequest
from src.retrieval.buckets import BucketIndex, head_noun_token
from src.retrieval.constraint_index import ConstraintIndex, is_inert, prepare
from src.retrieval.retrieval import Retriever
from src.retrieval.strategies import (
    Bm25Strategy,
    BucketPipeline,
    BucketStrategy,
    ConstraintStrategy,
    DEFAULT_POOL,
    PreparedConstraints,
)
from src.reranker.types import RankResult


def retrieval_mode() -> str:
    """Ship default is ``bucket``; ``RETRIEVAL_MODE=legacy`` reproduces the
    original BM25 pipeline byte-identically (the A/B control and the last
    fallback rung)."""
    return os.environ.get("RETRIEVAL_MODE", "bucket").strip().lower() or "bucket"


# Additive rerank bonus for a product whose structured `categories` field
# contains a disclosed category term. 0.0 reproduces current behavior
# byte-for-byte -- see scripts/ab_eval.py + scripts/paraphrase_stress.py runs
# in runs/log.jsonl before changing this value; two similarly-plausible
# reranking boosts in this repo's history looked good and measurably
# regressed the score when actually tested (see project memory).
CATEGORY_MATCH_BONUS = 0.5


def _category_terms(category_constraints: list[str] | None) -> frozenset[str]:
    """Normalize disclosed category constraint value(s) into individual
    lowercase leaf terms -- same normalization + comma-split-per-element
    handling as ``src.message_parser.catalog_vocab.load_catalog_vocab``."""
    terms: set[str] = set()
    for raw in (category_constraints or []):
        for part in str(raw).split(","):
            cleaned = _normalize(part)
            if cleaned:
                terms.add(cleaned)
    return frozenset(terms)


def _category_bonus(product: Product, category_terms: frozenset[str]) -> float:
    """Soft nudge, never a filter: CATEGORY_MATCH_BONUS if any disclosed
    category term appears verbatim in the product's structured categories,
    else 0."""
    if not CATEGORY_MATCH_BONUS or not category_terms or not product.categories:
        return 0.0
    return CATEGORY_MATCH_BONUS if category_terms & product.categories else 0.0


_ROW_COLUMNS = (
    "parent_asin",
    "title",
    "categories",
    "features",
    "details",
    "store",
    "description",
    "price",
    "average_rating",
    "rating_number",
)


def default_query(constraints: list[str], extra: str = "") -> str:
    """Build a retrieval query string from known constraints (+ optional text)."""
    return " ".join([*constraints, extra]).strip()


def _hydrate_products(
    catalog: Catalog,
    parent_asins: list[str],
    cache: dict[str, Product] | None = None,
) -> dict[str, Product]:
    """Batch-fetch catalog rows for ``parent_asins`` and build reranker ``Product``
    shims keyed by parent_asin.

    ``cache`` (if given) is a persistent, content-addressed store of
    previously hydrated products (keyed by ``parent_asin``, never mutated by
    the catalog during a run) -- only the ids missing from it are fetched,
    and the cache is updated in place with any newly fetched rows."""
    if not parent_asins:
        return {}
    if cache is None:
        missing = parent_asins
    else:
        missing = [pid for pid in parent_asins if pid not in cache]
    if missing:
        placeholders = ", ".join("?" for _ in missing)
        sql = (
            f"SELECT {', '.join(_ROW_COLUMNS)} FROM products "
            f"WHERE parent_asin IN ({placeholders})"
        )
        rows = catalog.execute(sql, missing)
        fetched = _rows_to_products(rows, _category_terms_by_asin(str(catalog.catalog_path)))
        if cache is not None:
            cache.update(fetched)
    else:
        fetched = {}
    if cache is None:
        return fetched
    return {pid: cache[pid] for pid in parent_asins if pid in cache}


@lru_cache(maxsize=4)
def _category_terms_by_asin(catalog_path: str) -> dict[str, frozenset[str]]:
    """Normalized leaf category terms per product, keyed by parent_asin.

    Built from the raw catalog rows (``load_catalog_rows``, already cached),
    NOT from the FTS ``categories`` column: that column is pre-flattened into
    one space-joined string per product (see ``Catalog._build_index`` /
    ``_text``), which loses the boundary between separate `categories` list
    elements -- e.g. ["Clothing, Shoes & Jewelry", "Men", "Watches"] becomes
    "Clothing, Shoes & Jewelry Men Watches", and splitting that string on ","
    would merge "Men" and "Watches" into one ungrabbable blob. Re-deriving
    from the raw list (same comma-split-per-element + normalize as
    ``src.message_parser.catalog_vocab.load_catalog_vocab``) is the only way
    to recover individual leaf terms."""
    result: dict[str, frozenset[str]] = {}
    for row in load_catalog_rows(catalog_path):
        terms: set[str] = set()
        for raw in row.get("categories") or []:
            for part in str(raw).split(","):
                cleaned = _normalize(part)
                if cleaned:
                    terms.add(cleaned)
        result[str(row["parent_asin"])] = frozenset(terms)
    return result


def _rows_to_products(
    rows: list[tuple],
    category_terms_by_asin: dict[str, frozenset[str]] | None = None,
) -> dict[str, Product]:
    category_terms_by_asin = category_terms_by_asin or {}
    products: dict[str, Product] = {}
    for row in rows:
        (
            parent_asin,
            title,
            categories,
            features,
            details,
            store,
            description,
            price,
            average_rating,
            rating_number,
        ) = row
        text = " ".join(
            str(part)
            for part in (title, categories, features, details, store, description)
            if part
        ).lower()
        products[str(parent_asin)] = Product(
            parent_asin=str(parent_asin),
            text=text,
            price=float(price) if price is not None else None,
            rating_number=int(rating_number) if rating_number is not None else 0,
            average_rating=float(average_rating) if average_rating is not None else 0.0,
            categories=category_terms_by_asin.get(str(parent_asin), frozenset()),
        )
    return products


_BUCKET_CACHE_MAX = 4096


class Reranker:
    """Scores an already-retrieved candidate pool into a :class:`RankResult`.

    Retrieval (which method, in what order, with what caching) is delegated to
    the injected strategies; the reranker only scores and assembles.
    """

    def __init__(
        self,
        catalog: Catalog,
        bm25: Bm25Strategy | None = None,
        bucket_pipeline: BucketPipeline | None = None,
        constraint: ConstraintStrategy | None = None,
        constraint_index: ConstraintIndex | None = None,
        bucket_index: BucketIndex | None = None,
    ) -> None:
        self.catalog = catalog
        self._bm25 = bm25
        self._bucket_pipeline = bucket_pipeline
        self._constraint = constraint
        self.constraint_index = constraint_index
        self.bucket_index = bucket_index
        # Process-lifetime, content-addressed product cache: the catalog is
        # read-only for the duration of a run, so a cache hit is always exactly
        # the value the uncached path would have computed.
        self._product_cache: dict[str, Product] = {}
        # Per-session resolved bucket key, keyed by opening message. The
        # coarse category is disclosed once (turn 1) and holds for the whole
        # session, so resolution is cached rather than re-run every turn.
        # Bounded: the key is arbitrary user text, one entry per session.
        self._bucket_cache: OrderedDict[str, tuple[list[str], bool, bool]] = OrderedDict()
        # Shared fallback pool. Materializing a fresh 50k list per unresolved
        # session and pinning it in the cache is a lot of memory for a list
        # nobody mutates.
        self._all_asins: list[str] | None = None

    def rank_bucket(
        self,
        opening_message: str,
        constraints: list[str] | None = None,
        top_k: int = 10,
        transcript: str = "",
        preference_tags: list[str] | None = None,
        rating_style: str | None = None,
        category_constraints: list[str] | None = None,
    ) -> RankResult:
        """Bucket-mode retrieval + verbatim-constraint scoring.

        The candidate pool -- resolved bucket, whole-catalog BM25 fallback, and
        the paraphrase-insurance token scoring -- is produced by the injected
        :class:`~src.retrieval.strategies.BucketPipeline`; the pool is then
        scored by :meth:`score_by_constraints`.
        """
        if self._bucket_pipeline is None or self._constraint is None:
            return RankResult()
        constraints = constraints or []

        request = RetrievalRequest(
            opening_message=opening_message,
            constraints=constraints,
            transcript=transcript,
            top_k=top_k,
            rating_style=rating_style,
        )
        result = self._bucket_pipeline.candidates(request)
        return self.score_by_constraints(
            result.candidates,
            self._bucket_pipeline.last_prepared,
            pool_size=self._bucket_pipeline.last_pool_size,
            preference_tags=preference_tags,
            rating_style=rating_style,
            category_constraints=category_constraints,
        )

    def rank(
        self,
        query: str,
        constraints: list[str] | None = None,
        top_k: int = 10,
        pool_size: int = DEFAULT_POOL,
        preference_tags: list[str] | None = None,
        rating_style: str | None = None,
        category_constraints: list[str] | None = None,
    ) -> RankResult:
        """Legacy BM25 retrieval + coverage scoring."""
        constraints = constraints or []
        if self._bm25 is None:
            return RankResult()

        # Consume the uniform retrieval boundary: hand the strategy a request
        # (pre-composed query for the legacy path) and read back a result.
        request = RetrievalRequest(
            constraints=constraints,
            query=query,
            top_k=top_k,
            pool_size=pool_size,
        )
        candidate_ids = self._bm25.candidates(request).candidates
        return self.score_by_coverage(
            candidate_ids,
            constraints,
            top_k=top_k,
            preference_tags=preference_tags,
            rating_style=rating_style,
            category_constraints=category_constraints,
        )

    # ------------------------------------------------------------------
    # Scoring cores: pure functions of a candidate pool. They take an
    # already-produced pool (from any strategy, or a merged pool from several)
    # so future combined pipelines can retrieve from N methods and reuse these.
    # ------------------------------------------------------------------

    def score_by_coverage(
        self,
        candidate_ids: list[str],
        constraints: list[str],
        top_k: int = 10,
        preference_tags: list[str] | None = None,
        rating_style: str | None = None,
        category_constraints: list[str] | None = None,
    ) -> RankResult:
        """Coverage scoring: order by (coverage, retrieval rank, rating).

        ``candidate_ids`` is any ordered pool of ``parent_asin`` -- its order is
        used as the retrieval-rank tiebreak, so a merged multi-strategy pool
        works here unchanged.
        """
        if not candidate_ids:
            return RankResult()

        products = _hydrate_products(self.catalog, candidate_ids, cache=self._product_cache)

        # Compile each constraint once, then reuse across all candidates.
        matchers = compile_constraints(constraints)
        pref_matchers = compile_constraints(list(preference_tags or []))
        category_terms = _category_terms(category_constraints)

        # Score each candidate: coverage, retrieval rank (lower=better), rating.
        # Track max coverage and its crowd in the same scan (no second pass).
        scored = []
        max_coverage = 0
        top_tier_crowd = 0
        for retrieval_rank, pid in enumerate(candidate_ids):
            product = products.get(pid)
            if product is None:
                continue
            cov = sum(1 for m in matchers if m.matches(product))
            pref_bonus = 0.15 * sum(1 for m in pref_matchers if m.matches(product))
            cat_bonus = _category_bonus(product, category_terms)
            score = cov + pref_bonus + cat_bonus
            scored.append((score, retrieval_rank, product))
            if cov > max_coverage:
                max_coverage = cov
                top_tier_crowd = 1
            elif cov == max_coverage:
                top_tier_crowd += 1

        if not scored:
            return RankResult()

        # rating_style adjusts how much average_rating vs rating_number (volume)
        # influences tie-breaking. A "critical" rater deflates scores, making
        # average_rating noisy -- lean on volume. A "usually positive" rater's
        # high scores are meaningful -- lean on average_rating.
        style = (rating_style or "").lower()
        if "critical" in style:
            w_rating, w_volume = 0.5, 1.5
        elif "positive" in style:
            w_rating, w_volume = 1.5, 0.5
        else:  # "mixed" or unknown
            w_rating, w_volume = 1.0, 1.0

        # Rerank: coverage desc, retrieval rank asc, rating desc, id asc (stable).
        scored.sort(
            key=lambda s: (
                -s[0],
                s[1],
                -(w_volume * s[2].rating_number + w_rating * s[2].average_rating * 1000),
                s[2].parent_asin,
            )
        )

        ranked_ids = [s[2].parent_asin for s in scored[:top_k]]

        return RankResult(
            ranked=ranked_ids,
            pool_size=len(scored),
            max_coverage=max_coverage,
            top_tier_crowd=top_tier_crowd,
        )

    def score_by_constraints(
        self,
        ranked: list[str],
        prepared: PreparedConstraints,
        pool_size: int,
        preference_tags: list[str] | None = None,
        rating_style: str | None = None,
        category_constraints: list[str] | None = None,
    ) -> RankResult:
        """Verbatim-constraint coverage over an already-ranked pool.

        ``ranked`` is expected pre-ordered by the constraint index (score, then
        popularity); this only computes the advisory ``max_coverage`` the
        confidence gate reads. ``pool_size`` is the pre-truncation pool size.
        """
        if self._constraint is None or not ranked:
            return RankResult()

        # max_coverage / crowd are advisory internals for the confidence gate;
        # in bucket mode the exposure gate is turn-based, so a coarse count of
        # constraints that landed a nonzero score on the top candidate suffices.
        max_cov = 0
        if prepared:
            best = ranked[0]
            max_cov = sum(
                1 for norm, toks, w in prepared
                if self._constraint.score(best, [(norm, toks, w)]) > 0.0
            )

        # Second-pass re-sort using rating_style (always), preference_tags
        # (only when no constraints matched yet — turn 1 pool is unordered by
        # constraints so pref signals are the only differentiator), and the
        # category bonus (every turn -- unlike preference tags, a disclosed
        # category constraint keeps nudging score for the whole session, not
        # just turn 1, since the underlying problem -- wrong-category items
        # outscoring true matches on plain BM25 text overlap -- persists
        # regardless of what other constraints have accumulated).
        use_prefs = bool(preference_tags) and not prepared
        category_terms = _category_terms(category_constraints)
        use_category = bool(category_terms) and CATEGORY_MATCH_BONUS > 0
        if rating_style or use_prefs or use_category:
            products = _hydrate_products(self.catalog, ranked, cache=self._product_cache)
            pref_matchers = compile_constraints(list(preference_tags or [])) if use_prefs else []
            style = (rating_style or "").lower()
            if "critical" in style:
                w_rating, w_volume = 0.5, 1.5
            elif "positive" in style:
                w_rating, w_volume = 1.5, 0.5
            else:
                w_rating, w_volume = 1.0, 1.0

            scored = []
            for retrieval_rank, pid in enumerate(ranked):
                product = products.get(pid)
                if product is None:
                    scored.append((retrieval_rank, 0.0, pid, 0.0))
                    continue
                pref_bonus = 0.15 * sum(1 for m in pref_matchers if m.matches(product)) if use_prefs else 0.0
                cat_bonus = _category_bonus(product, category_terms) if use_category else 0.0
                total_bonus = pref_bonus + cat_bonus
                rating_score = w_volume * product.rating_number + w_rating * product.average_rating * 1000
                scored.append((retrieval_rank, total_bonus, pid, rating_score))

            scored.sort(key=lambda s: (s[0], -s[1], -s[3], s[2]))
            ranked = [s[2] for s in scored]

        return RankResult(
            ranked=ranked,
            pool_size=pool_size,
            max_coverage=max_cov,
            top_tier_crowd=1,
        )


def build_reranker(catalog_path: str) -> Reranker:
    catalog = Catalog(catalog_path)
    retriever = Retriever(catalog)
    # The bucket + verbatim-constraint indexes share the lru_cached catalog
    # rows, so building them here is one extra pass over already-parsed data.
    rows = load_catalog_rows(str(catalog_path))
    bucket_index = BucketIndex(rows)
    constraint_index = ConstraintIndex(rows)

    # Wrap each retrieval method as a strategy; the bucket-mode ladder is the
    # composite that orchestrates all three.
    bm25 = Bm25Strategy(retriever)
    constraint = ConstraintStrategy(constraint_index)
    bucket = BucketStrategy(bucket_index, constraint_index)
    bucket_pipeline = BucketPipeline(bucket, constraint, bm25)

    return Reranker(catalog, bm25=bm25, bucket_pipeline=bucket_pipeline, constraint=constraint, constraint_index=constraint_index, bucket_index=bucket_index)
