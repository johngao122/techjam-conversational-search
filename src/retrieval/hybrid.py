"""Hybrid retrieval: BM25 + vector in parallel on constraint-filtered pool, fused with RRF.

Flow:
    1. Pre-filter products by disclosed constraints (hard filter).
    2. Run BM25 and vector search IN PARALLEL on the filtered pool.
    3. Reciprocal Rank Fusion of both ranked lists.

Activated by ``RETRIEVAL_MODE=hybrid`` in :mod:`src.agent`.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.retrieval.retrieval import Retriever
    from src.retrieval.constraint_index import ConstraintIndex

_LOG = logging.getLogger(__name__)

# Field order for the natural-language constraint string.
# Category last = English head noun ("blue cotton dress").
_FIELD_ORDER = ["color", "material", "style", "use_case", "brand", "type", "category"]


class HybridRetriever:
    """Constraint pre-filter + parallel BM25/vector, fused with RRF."""

    def __init__(
        self,
        retriever: "Retriever",
        constraint_index: "ConstraintIndex | None" = None,
        pool_size: int = 200,
        rrf_k: int = 60,
        rrf_weights: tuple[float, float, float] = (6.0, 1.0, 1.0),
        bucket_index: "BucketIndex | None" = None,
    ) -> None:
        from src.retrieval.buckets import BucketIndex
        
        self._retriever = retriever
        self._constraint_index = constraint_index
        self._pool_size = pool_size
        self._rrf_k = rrf_k
        self._rrf_weights = rrf_weights
        self._bucket_index = bucket_index
        # Build FilteredVectorIndex only when vectors are available
        self._fvi = None
        if retriever.has_vectors:
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
        """Return a fused candidate pool for the reranker.

        Flow:
            1. Pre-filter by constraints (if ConstraintIndex available)
            2. BM25 and vector search run IN PARALLEL on the filtered pool
            3. RRF fuse results

        ``constraints`` are verbatim disclosed constraint strings used as a
        hard pre-filter before both searches. Returns ``[]`` when both searches
        find nothing, so the caller can fall back to the popularity list.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        start_total = time.time()

        # Step 1: Build the constraint-filtered pool
        start_constraint = time.time()
        filtered_pool = self._get_constraint_filtered_pool(constraints, opening_message)
        constraint_time = time.time() - start_constraint
        _LOG.debug(f"constraint_filter: {constraint_time:.3f}s, pool_size={len(filtered_pool) if filtered_pool else 0}")

        # Enrich category with full fragment from opening message for BM25
        enriched_key = dict(search_key)
        if opening_message:
            from src.retrieval.buckets import parse_category
            fragment = parse_category(opening_message).strip().lower()
            if fragment:
                existing = enriched_key.get("category") or []
                if fragment not in existing:
                    enriched_key["category"] = [fragment, *existing]

        # Step 2: Run BM25 and vector search IN PARALLEL on the filtered pool
        bm25_ranked: list[str] = []
        vector_ranked: list[str] = []

        def run_bm25() -> list[str]:
            """BM25 search, optionally restricted to filtered pool."""
            results = self._retriever.retrieve_bm25(enriched_key, top_k=self._pool_size)
            if filtered_pool is not None:
                # Keep only results in the filtered pool, preserve BM25 order
                pool_set = set(filtered_pool)
                results = [r for r in results if r in pool_set]
            return results

        def run_vector() -> list[str]:
            """Vector search on filtered pool."""
            if self._fvi is None or not vector_query.strip():
                return []
            try:
                query_vec = self._retriever.embedding_client.embed_one(vector_query)
                # Search within filtered pool (or full index if no constraints)
                if filtered_pool is not None:
                    hits = self._fvi.search_filtered(
                        query_vec, set(filtered_pool), top_k=self._pool_size
                    )
                else:
                    # No constraint filter: search full index
                    hits = self._retriever.vector_index.search(query_vec, top_k=self._pool_size)
                return [asin for asin, _ in hits]
            except Exception:
                return []

        # Execute both searches in parallel with timeout
        start_parallel = time.time()
        bm25_ranked: list[str] = []
        vector_ranked: list[str] = []

        with ThreadPoolExecutor(max_workers=2) as executor:
            future_bm25 = executor.submit(run_bm25)
            future_vector = executor.submit(run_vector)

            try:
                bm25_ranked = future_bm25.result(timeout=5.0)
                _LOG.debug(f"bm25_search: {time.time() - start_parallel:.3f}s, results={len(bm25_ranked)}")
            except Exception as e:
                _LOG.debug(f"bm25_search failed: {e}")
                pass

            try:
                vector_ranked = future_vector.result(timeout=5.0)
                _LOG.debug(f"vector_search: {time.time() - start_parallel:.3f}s, results={len(vector_ranked)}")
            except Exception as e:
                _LOG.debug(f"vector_search failed: {e}")
                pass

        parallel_time = time.time() - start_parallel

        # Step 3: RRF fuse — constraint pool (already sorted by score) is
        # included as a third ranked list to boost high-constraint matches.
        start_rrf = time.time()
        if not bm25_ranked and not vector_ranked:
            # Both failed: return filtered pool ordered by constraint score if available
            if filtered_pool:
                result = filtered_pool[:self._pool_size]
                total_time = time.time() - start_total
                _LOG.debug(f"retrieve fallback: total={total_time:.3f}s, results={len(result)}")
                return result
            result = []
            total_time = time.time() - start_total
            _LOG.debug(f"retrieve empty: total={total_time:.3f}s")
            return result

        # Build weighted ranked lists for RRF. Constraint signal is most
        # reliable (verbatim attribute match); BM25/vector aid tiebreaking.
        # Format: (ranked_list, weight)
        constraint_w, bm25_w, vector_w = self._rrf_weights
        weighted_lists: list[tuple[list[str], float]] = []
        if filtered_pool:
            weighted_lists.append((filtered_pool[:self._pool_size], constraint_w))
        weighted_lists.append((bm25_ranked, bm25_w))
        weighted_lists.append((vector_ranked, vector_w))

        result = self._rrf_fuse_weighted(weighted_lists, k=self._rrf_k)
        rrf_time = time.time() - start_rrf
        total_time = time.time() - start_total
        _LOG.debug(f"retrieve: total={total_time:.3f}s, constraint={constraint_time:.3f}s, parallel={parallel_time:.3f}s, rrf={rrf_time:.3f}s, results={len(result)}")
        return result

    def _get_constraint_filtered_pool(
        self,
        constraints: list[str] | None,
        opening_message: str = "",
    ) -> list[str] | None:
        """Pre-filter products by category bucket AND constraints.

        Narrows to the category bucket first, then scores by constraint match
        for better precision than scoring all 50k products. Returns ASINs
        ordered by constraint score then popularity, or None when unavailable.
        """
        start = time.time()
        if self._constraint_index is None:
            return None

        constraints = constraints or []

        from src.retrieval.constraint_index import is_inert, prepare
        from src.retrieval.buckets import BucketIndex, parse_category

        # Step 1: Get category bucket via fuzzy resolution (handles paraphrased openings).
        start_bucket = time.time()
        category_pool: set[str] | None = None
        if opening_message and self._bucket_index is not None:
            # Use resolve() for fuzzy matching (exact -> containment -> token overlap)
            bucket_key, match_type = self._bucket_index.resolve(opening_message)
            if bucket_key:
                bucket_asins = self._bucket_index.get(bucket_key)
                if bucket_asins:
                    category_pool = set(bucket_asins)
        bucket_time = time.time() - start_bucket

        # Step 2: Prepare constraints
        prepared = [
            (norm, toks, 1.0)
            for norm, toks in (prepare(c) for c in constraints if not is_inert(c))
            if norm
        ]

        # If no constraints but we have a category pool, return it ordered by popularity
        if not prepared:
            if category_pool:
                pop = self._constraint_index.popularity
                result = sorted(category_pool, key=lambda a: -pop.get(a, 0))
                elapsed = time.time() - start
                _LOG.debug(f"_constraint_filter (no constraints): {elapsed:.3f}s, bucket={bucket_time:.3f}s, results={len(result)}")
                return result
            return None

        # Step 3: Score products by constraint match
        # If we have a category pool, only score those products (much faster)
        start_score = time.time()
        if category_pool:
            scores: dict[str, float] = {}
            for asin in category_pool:
                score = self._constraint_index.score(asin, prepared)
                if score > 0:
                    scores[asin] = score
            # If no constraint matches in category, return category pool by popularity
            if not scores:
                pop = self._constraint_index.popularity
                result = sorted(category_pool, key=lambda a: -pop.get(a, 0))
                score_time = time.time() - start_score
                elapsed = time.time() - start
                _LOG.debug(f"_constraint_filter (no matches): {elapsed:.3f}s, bucket={bucket_time:.3f}s, score={score_time:.3f}s, results={len(result)}")
                return result
        else:
            # No category pool - fall back to fast_candidates on full catalog
            scores = self._constraint_index.fast_candidates(prepared, exact_only=False)
            if not scores:
                score_time = time.time() - start_score
                elapsed = time.time() - start
                _LOG.debug(f"_constraint_filter (candidates empty): {elapsed:.3f}s, bucket={bucket_time:.3f}s, score={score_time:.3f}s")
                return None

        score_time = time.time() - start_score

        # Sort by score descending, then by popularity (tiebreaker)
        scored = list(scores.items())
        scored.sort(key=lambda x: (-x[1], -self._constraint_index.popularity.get(x[0], 0)))
        result = [asin for asin, _ in scored]
        elapsed = time.time() - start
        _LOG.debug(f"_constraint_filter: {elapsed:.3f}s, bucket={bucket_time:.3f}s, score={score_time:.3f}s, results={len(result)}")
        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def build_constraint_query(session: dict, opening_message: str = "") -> str:
        """Build a semantic vector query from structured session state.

        Uses intent + current constraints (not raw history) to match the
        title/categories/description register the vector index embeds. Intent
        is mapped to natural-language phrases the embedding space understands.
        """
        _INTENT_PHRASES = {
            "buying": "looking to buy",
            "browsing": "looking for",
        }

        parts: list[str] = []

        # Intent phrase — only for intents that map to natural language
        intent = (session.get("intent") or "").strip()
        intent_phrase = _INTENT_PHRASES.get(intent, "")
        if intent_phrase:
            parts.append(intent_phrase)

        # Category (strongest signal — matches taxonomy path embedded in index)
        constraints = session.get("constraints") or {}
        category = constraints.get("category") or []
        if category:
            parts.append(category[0])

        # Remaining constraint values (material, color, style etc.)
        for attr, values in constraints.items():
            if attr == "category":
                continue
            if values:
                parts.append(values[0])

        if parts:
            return " ".join(parts).strip()

        # Turn 1 fallback: no constraints yet — use opening message category fragment
        if opening_message:
            from src.retrieval.buckets import parse_category
            fragment = parse_category(opening_message).strip()
            if fragment:
                return fragment

        return ""

    @staticmethod
    def _rrf_fuse_weighted(
        weighted_lists: list[tuple[list[str], float]], k: int = 60
    ) -> list[str]:
        """Weighted Reciprocal Rank Fusion.

        ``score(d) = Σ weight * 1 / (k + rank(d))`` where rank is 1-indexed.
        ASINs absent from a list contribute 0. Higher weights favour more
        reliable signals (constraint match over BM25/vector here).
        """
        scores: dict[str, float] = {}
        for ranked, weight in weighted_lists:
            for rank, asin in enumerate(ranked, start=1):
                scores[asin] = scores.get(asin, 0.0) + weight * 1.0 / (k + rank)
        return sorted(scores, key=lambda a: scores[a], reverse=True)

    @staticmethod
    def _rrf_fuse(ranked_lists: list[list[str]], k: int = 60) -> list[str]:
        """Reciprocal Rank Fusion (unweighted, for backward compat).

        ``score(d) = Σ 1 / (k + rank(d))`` where rank is 1-indexed.
        ASINs absent from a list contribute 0 from that list.
        """
        scores: dict[str, float] = {}
        for ranked in ranked_lists:
            for rank, asin in enumerate(ranked, start=1):
                scores[asin] = scores.get(asin, 0.0) + 1.0 / (k + rank)
        return sorted(scores, key=lambda a: scores[a], reverse=True)
