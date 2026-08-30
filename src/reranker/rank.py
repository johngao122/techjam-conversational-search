"""Reranker core: retrieve -> coverage-rerank -> RankResult.

Public API:
    build_reranker(catalog_path) -> Reranker
    Reranker.rank(query, constraints, top_k) -> RankResult
    default_query(constraints) -> str        # helper to build a query from constraints

``rank`` reranks retrieved candidates by (coverage desc, retrieval rank asc,
rating desc) and assembles the internals the confidence check needs:
``max_coverage`` and ``top_tier_crowd``.
"""

from __future__ import annotations

import os
from collections import OrderedDict

from src.catalog.catalog import Catalog
from src.catalog.loader import load_catalog_rows
from src.reranker.coverage import Product, compile_constraints
from src.retrieval.buckets import BucketIndex, head_noun_token
from src.retrieval.constraint_index import ConstraintIndex, is_inert, prepare
from src.retrieval.retrieval import Retriever
from src.reranker.types import RankResult

DEFAULT_POOL = 200


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


_BUCKET_CACHE_MAX = 4096


class Reranker:
    def __init__(
        self,
        catalog: Catalog,
        retriever: Retriever,
        bucket_index: BucketIndex | None = None,
        constraint_index: ConstraintIndex | None = None,
    ) -> None:
        self.catalog = catalog
        self.retriever = retriever
        self.bucket_index = bucket_index
        self.constraint_index = constraint_index
        # Process-lifetime, content-addressed caches: the catalog is
        # read-only for the duration of a run, so a cache hit is always
        # exactly the value the uncached path would have computed.
        self._bm25_cache: dict[tuple[str, int], list[str]] = {}
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
    ) -> RankResult:
        """Deterministic bucket -> verbatim-constraint -> popularity ranking.

        The opening message names (verbatim, on the public set) the target's
        own coarse category, which resolves to a bucket guaranteed to contain
        the target. Within that pool candidates are scored by weighted
        verbatim-constraint match, then popularity.

        Robustness ladder, each rung degrading rather than losing the turn:
          1. resolved bucket + verbatim constraints  (primary path)
          2. resolved bucket + popularity            (no constraint matched yet)
          3. whole catalog, narrowed by BM25 over the accumulated transcript
             (category unresolved -- template/paraphrase drift on the private set)
          4. whole catalog + popularity              (nothing else fired)
        """
        if self.constraint_index is None or self.bucket_index is None:
            return RankResult()
        constraints = constraints or []

        resolved = True
        resolved_exact = True
        cached = self._bucket_cache.get(opening_message)
        if cached is None:
            key, how = self.bucket_index.resolve(opening_message)
            pool = self.bucket_index.get(key) if key else []
            resolved_exact = how == "exact"
            if not pool:
                resolved = False
                if self._all_asins is None:
                    self._all_asins = list(self.constraint_index.attributes.keys())
                pool = self._all_asins
            self._bucket_cache[opening_message] = (pool, resolved, resolved_exact)
            if len(self._bucket_cache) > _BUCKET_CACHE_MAX:
                self._bucket_cache.popitem(last=False)
        else:
            pool, resolved, resolved_exact = cached

        # Rung 3: category unresolved -> narrow the whole-catalog pool with a
        # BM25 pass over the accumulated transcript (not just parsed spans, so
        # a reworded disclosure still contributes). Only when a transcript is
        # available; otherwise fall straight through to popularity.
        if not resolved and transcript.strip():
            bm25_pool = self.retriever.retrieve_bm25(
                {"keywords": [transcript]}, top_k=DEFAULT_POOL
            )
            if bm25_pool:
                pool = bm25_pool

            # Title-relevance gate: the transcript-wide BM25 pass above treats
            # the item type as just one soft OR'd term among color/budget/etc,
            # so an off-type product with strong matches elsewhere can win.
            # Hard-require the disclosed type word to actually appear (as a
            # prefix match) in the candidate's own title/categories. Degrades
            # to the unfiltered pool if that leaves nothing, or if no type
            # word could be parsed at all. Uses only the single head-noun
            # token (not every content word) -- OR-ing in color/material
            # words here would readmit exactly the off-type matches this
            # gate exists to block (e.g. a heel whose title also says
            # "satin"). Parsed from the opening message, whose tail names
            # the category by construction (see buckets.py's module doc).
            head = head_noun_token(opening_message)
            if head:
                relevant = self.retriever.title_relevant_ids({head})
                gated = [pid for pid in pool if pid in relevant]
                if gated:
                    pool = gated

        prepared = [
            (norm, toks, 1.0)
            for norm, toks in (prepare(c) for c in constraints if not is_inert(c))
            if norm
        ]

        # Paraphrase insurance: score the bucket by transcript-token overlap
        # when the verbatim path produced nothing AND the opening category
        # itself failed to resolve exactly -- i.e. the template wording drifted.
        # Gating on the inexact-resolution signal keeps the clean public set
        # (where the category always resolves exactly and an exact constraint
        # match dominates) completely untouched.
        if not prepared and not resolved_exact and transcript.strip():
            _, t_toks = prepare(transcript)
            for tok in dict.fromkeys(t_toks):
                prepared.append((tok, (tok,), 0.15))

        ranked = self.constraint_index.rank(pool, prepared, top_k)
        if not ranked:
            return RankResult()

        # max_coverage / crowd are advisory internals for the confidence gate;
        # in bucket mode the exposure gate is turn-based, so a coarse count of
        # constraints that landed a nonzero score on the top candidate suffices.
        max_cov = 0
        if prepared:
            best = ranked[0]
            max_cov = sum(
                1 for norm, toks, w in prepared
                if self.constraint_index.score(best, [(norm, toks, w)]) > 0.0
            )
        return RankResult(
            ranked=ranked,
            pool_size=len(pool),
            max_coverage=max_cov,
            top_tier_crowd=1,
        )

    def rank(
        self,
        query: str,
        constraints: list[str] | None = None,
        top_k: int = 10,
        pool_size: int = DEFAULT_POOL,
    ) -> RankResult:
        constraints = constraints or []
        cache_key = (query, pool_size)
        candidate_ids = self._bm25_cache.get(cache_key)
        if candidate_ids is None:
            candidate_ids = self.retriever.retrieve_bm25({"keywords": [query]}, top_k=pool_size)
            self._bm25_cache[cache_key] = candidate_ids

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


def build_reranker(catalog_path: str) -> Reranker:
    catalog = Catalog(catalog_path)
    retriever = Retriever(catalog)
    # The bucket + verbatim-constraint indexes share the lru_cached catalog
    # rows, so building them here is one extra pass over already-parsed data.
    rows = load_catalog_rows(str(catalog_path))
    bucket_index = BucketIndex(rows)
    constraint_index = ConstraintIndex(rows)
    return Reranker(catalog, retriever, bucket_index, constraint_index)
