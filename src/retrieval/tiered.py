"""Tiered bucket-fusion retrieval: bucket score, boosted by BM25 / vector.

Unlike :mod:`src.retrieval.hybrid` (which pools bucket + BM25 + vector into a
single RRF ranking where an off-bucket product can surface), this module keeps
the candidate set *confined to the resolved bucket pool* and only re-orders it:

    final(asin) = bucket_score(asin)
                + alpha * [asin in bm25]   * 1/(k + rank_bm25(asin))
                + beta  * [asin in vector] * 1/(k + rank_vector(asin))

So an overlapping hit (a product BM25/vector *also* retrieved) has its
bucket-derived score *boosted* -- shifting the top-k ordering -- but a product
that is only strong in BM25/vector and absent from the bucket never enters the
result set, and never automatically jumps to the top. The boost is rank-decayed
(RRF-style): a product ranked #1 by BM25/vector boosts more than a marginal one.

Degrades gracefully: when a complementary source returns nothing (empty BM25,
vector layer unavailable, endpoint down), its term contributes 0 and the
ranking reduces to the pure bucket ordering.

Activated by ``RETRIEVAL_MODE`` in :mod:`src.agent`:
    bucket_bm25          -- bucket score + BM25 boost
    bucket_vector        -- bucket score + vector boost
    bucket_bm25_vector   -- bucket score + BM25 + vector boost
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.retrieval.retrieval import Retriever
    from src.retrieval.constraint_index import ConstraintIndex
    from src.retrieval.buckets import BucketIndex

_LOG = logging.getLogger(__name__)


class TieredRetriever:
    """Bucket pool, re-ordered by additive rank-decayed BM25/vector boosts."""

    def __init__(
        self,
        retriever: "Retriever",
        constraint_index: "ConstraintIndex",
        bucket_index: "BucketIndex | None" = None,
        pool_size: int = 200,
        rrf_k: int = 60,
        bm25_weight: float = 1.0,
        vector_weight: float = 1.0,
        use_bm25: bool = True,
        use_vector: bool = True,
    ) -> None:
        self._retriever = retriever
        self._constraint_index = constraint_index
        self._bucket_index = bucket_index
        self._pool_size = pool_size
        self._rrf_k = rrf_k
        self._bm25_weight = bm25_weight
        self._vector_weight = vector_weight
        self._use_bm25 = use_bm25
        self._use_vector = use_vector
        # FilteredVectorIndex is built only when vectors are available AND this
        # tier actually consumes them.
        self._fvi = None
        if use_vector and retriever.has_vectors:
            from src.embeddings.filtered_index import FilteredVectorIndex

            self._fvi = FilteredVectorIndex(retriever.vector_index)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(
        self,
        search_key: dict,
        vector_query: str,
        constraints: list[str] | None = None,
        top_k: int = 10,
        opening_message: str = "",
    ) -> list[str]:
        """Return the bucket pool re-ordered by fused score.

        ``search_key`` drives BM25; ``vector_query`` drives vector search;
        ``constraints`` are the verbatim disclosed strings scored against the
        bucket. Returns ``[]`` only when the bucket pool itself is empty (the
        caller then falls back to popularity).
        """
        pool, bucket_score = self._bucket_pool_and_scores(constraints, opening_message)
        if not pool:
            _LOG.debug("tiered: empty bucket pool")
            return []

        pool_set = set(pool)

        # Complementary ranked lists, each restricted to the bucket pool so a
        # boost can only ever land on an in-pool candidate.
        bm25_ranked: list[str] = []
        if self._use_bm25:
            enriched_key = self._enrich_category(search_key, opening_message)
            bm25_ranked = self._bm25_within_pool(enriched_key, pool_set)

        vector_ranked: list[str] = []
        if self._use_vector:
            vector_ranked = self._vector_within_pool(vector_query, pool_set)

        # Additive rank-decayed boost onto the base bucket score.
        final = dict(bucket_score)
        k = self._rrf_k
        for rank, asin in enumerate(bm25_ranked, start=1):
            if asin in final:
                final[asin] += self._bm25_weight * (1.0 / (k + rank))
        for rank, asin in enumerate(vector_ranked, start=1):
            if asin in final:
                final[asin] += self._vector_weight * (1.0 / (k + rank))

        popularity = self._constraint_index.popularity
        ranked = sorted(
            pool,
            key=lambda a: (-final.get(a, 0.0), -popularity.get(a, 0.0), a),
        )
        _LOG.debug(
            "tiered: pool=%d bm25=%d vector=%d boosted=%d",
            len(pool),
            len(bm25_ranked),
            len(vector_ranked),
            sum(1 for a in pool if final.get(a, 0.0) != bucket_score.get(a, 0.0)),
        )
        return ranked

    # ------------------------------------------------------------------
    # Bucket pool + base scores
    # ------------------------------------------------------------------

    def _bucket_pool_and_scores(
        self,
        constraints: list[str] | None,
        opening_message: str,
    ) -> tuple[list[str], dict[str, float]]:
        """Resolve the category bucket and score it by verbatim constraints.

        Mirrors the bucket-mode contract: narrow to the coarse-category bucket
        (fuzzy resolution), then score each member by constraint match. Returns
        ``(pool, {asin: bucket_score})``. Falls back to popularity ordering when
        no constraints match, and to ``([], {})`` when the bucket won't resolve.
        """
        from src.retrieval.constraint_index import is_inert, prepare

        category_pool: list[str] | None = None
        if opening_message and self._bucket_index is not None:
            bucket_key, _how = self._bucket_index.resolve(opening_message)
            if bucket_key:
                bucket_asins = self._bucket_index.get(bucket_key)
                if bucket_asins:
                    category_pool = list(bucket_asins)

        if not category_pool:
            return [], {}

        prepared = [
            (norm, toks, 1.0)
            for norm, toks in (prepare(c) for c in (constraints or []) if not is_inert(c))
            if norm
        ]

        bucket_score = self._constraint_index.score_pool(category_pool, prepared)
        return category_pool, bucket_score

    def _enrich_category(self, search_key: dict, opening_message: str) -> dict:
        """Fold the opening-message category fragment into the BM25 search key."""
        enriched = dict(search_key)
        if opening_message:
            from src.retrieval.buckets import parse_category

            fragment = parse_category(opening_message).strip().lower()
            if fragment:
                existing = enriched.get("category") or []
                if fragment not in existing:
                    enriched["category"] = [fragment, *existing]
        return enriched

    # ------------------------------------------------------------------
    # Complementary sources (each restricted to the bucket pool)
    # ------------------------------------------------------------------

    def _bm25_within_pool(self, search_key: dict, pool_set: set[str]) -> list[str]:
        try:
            results = self._retriever.retrieve_bm25(search_key, top_k=self._pool_size)
        except Exception as exc:  # noqa: BLE001 - never break ranking on BM25 failure
            _LOG.debug("tiered bm25 failed: %s", exc)
            return []
        return [r for r in results if r in pool_set]

    def _vector_within_pool(self, vector_query: str, pool_set: set[str]) -> list[str]:
        if self._fvi is None or not (vector_query or "").strip():
            return []
        try:
            query_vec = self._retriever.embedding_client.embed_one(vector_query)
            hits = self._fvi.search_filtered(query_vec, pool_set, top_k=self._pool_size)
            return [asin for asin, _score in hits]
        except Exception as exc:  # noqa: BLE001 - degrade to no vector boost
            _LOG.debug("tiered vector failed: %s", exc)
            return []
