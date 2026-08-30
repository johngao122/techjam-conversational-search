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

from src.catalog.catalog import Catalog
from src.catalog.loader import load_catalog_rows
from src.reranker.coverage import Product, compile_constraints
from src.retrieval.base import RetrievalRequest
from src.retrieval.constraint_index import ConstraintIndex
from src.retrieval.buckets import BucketIndex
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
        fetched = _rows_to_products(rows)
        if cache is not None:
            cache.update(fetched)
    else:
        fetched = {}
    if cache is None:
        return fetched
    return {pid: cache[pid] for pid in parent_asins if pid in cache}


def _rows_to_products(rows: list[tuple]) -> dict[str, Product]:
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
        )
    return products


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
    ) -> None:
        self.catalog = catalog
        self._bm25 = bm25
        self._bucket_pipeline = bucket_pipeline
        self._constraint = constraint
        # Process-lifetime, content-addressed product cache: the catalog is
        # read-only for the duration of a run, so a cache hit is always exactly
        # the value the uncached path would have computed.
        self._product_cache: dict[str, Product] = {}

    # ------------------------------------------------------------------
    # Retrieval entry points: pick a strategy, then hand its pool to the
    # matching scorer. Each is intentionally thin so the strategy and the
    # scorer stay independently swappable (and composable -- see below).
    # ------------------------------------------------------------------

    def rank_bucket(
        self,
        opening_message: str,
        constraints: list[str] | None = None,
        top_k: int = 10,
        transcript: str = "",
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
        )
        result = self._bucket_pipeline.candidates(request)
        return self.score_by_constraints(
            result.candidates,
            self._bucket_pipeline.last_prepared,
            pool_size=self._bucket_pipeline.last_pool_size,
        )

    def rank(
        self,
        query: str,
        constraints: list[str] | None = None,
        top_k: int = 10,
        pool_size: int = DEFAULT_POOL,
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
        return self.score_by_coverage(candidate_ids, constraints, top_k=top_k)

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
            scored.append((cov, retrieval_rank, product))
            if cov > max_coverage:
                max_coverage = cov
                top_tier_crowd = 1
            elif cov == max_coverage:
                top_tier_crowd += 1

        if not scored:
            return RankResult()

        # Rerank: coverage desc, retrieval rank asc, rating desc, id asc (stable).
        scored.sort(
            key=lambda s: (
                -s[0],
                s[1],
                -s[2].rating_number,
                -s[2].average_rating,
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

    return Reranker(catalog, bm25=bm25, bucket_pipeline=bucket_pipeline, constraint=constraint)
