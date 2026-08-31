"""Hybrid retrieval: BM25 + vector in parallel on constraint-filtered pool, fused with RRF.

Flow:
    1. Pre-filter products by disclosed constraints (hard filter).
    2. Run BM25 and vector search IN PARALLEL on the filtered pool.
    3. Reciprocal Rank Fusion of both ranked lists.

Activated by ``RETRIEVAL_MODE=hybrid`` in :mod:`src.agent`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.retrieval.retrieval import Retriever
    from src.retrieval.constraint_index import ConstraintIndex

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
    ) -> None:
        self._retriever = retriever
        self._constraint_index = constraint_index
        self._pool_size = pool_size
        self._rrf_k = rrf_k
        self._rrf_weights = rrf_weights
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

        ``constraints`` are the verbatim disclosed constraint strings (e.g.
        "Material:alloy") used as a hard pre-filter before both searches.

        Returns ``[]`` when both searches find nothing, allowing the caller
        to fall back to the popularity list.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        # Step 1: Build the constraint-filtered pool
        filtered_pool = self._get_constraint_filtered_pool(constraints, opening_message)

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
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_bm25 = executor.submit(run_bm25)
            future_vector = executor.submit(run_vector)

            try:
                bm25_ranked = future_bm25.result(timeout=5.0)
            except Exception:
                pass

            try:
                vector_ranked = future_vector.result(timeout=5.0)
            except Exception:
                pass

        # Step 3: RRF fuse — includes constraint ranking as a third signal
        # The constraint-filtered pool is already sorted by constraint score,
        # so include it as a ranked list to give high-constraint-match products
        # a boost even if BM25/vector rank them poorly.
        if not bm25_ranked and not vector_ranked:
            # Both failed: return filtered pool ordered by constraint score if available
            if filtered_pool:
                return filtered_pool[:self._pool_size]
            return []

        # Build weighted ranked lists for RRF:
        # Constraint signal is most reliable (verbatim match on disclosed attributes)
        # BM25/vector are secondary signals that can help with tiebreaking
        # Format: (ranked_list, weight)
        constraint_w, bm25_w, vector_w = self._rrf_weights
        weighted_lists: list[tuple[list[str], float]] = []
        if filtered_pool:
            # Constraint gets the highest weight - it's the ground truth signal
            weighted_lists.append((filtered_pool[:self._pool_size], constraint_w))
        # BM25 and vector get lower weights
        weighted_lists.append((bm25_ranked, bm25_w))
        weighted_lists.append((vector_ranked, vector_w))

        return self._rrf_fuse_weighted(weighted_lists, k=self._rrf_k)

    def _get_constraint_filtered_pool(
        self,
        constraints: list[str] | None,
        opening_message: str = "",
    ) -> list[str] | None:
        """Pre-filter products by disclosed constraints using inverted index.

        Returns a list of ASINs that match at least one constraint, ordered by
        constraint match score. Returns None if no constraint index or no
        constraints (meaning no filtering).

        Uses the inverted index for O(matches) instead of O(n) scoring.
        """
        if self._constraint_index is None:
            return None

        constraints = constraints or []
        if not constraints:
            return None

        from src.retrieval.constraint_index import is_inert
        from src.retrieval.constraint_index import is_inert, prepare

        # Prepare constraints: (normalised, tokens, weight=1.0)
        prepared = [
            (norm, toks, 1.0)
            for norm, toks in (prepare(c) for c in constraints if not is_inert(c))
            if norm
        ]
        if not prepared:
            return None

        # Always use the full fast_candidates search to maximize recall.
        # The small performance hit is worth the improved accuracy.
        scores = self._constraint_index.fast_candidates(prepared, exact_only=False)

        if not scores:
            return None

        # Sort by score descending, then by popularity (tiebreaker)
        scored = list(scores.items())
        scored.sort(key=lambda x: (-x[1], -self._constraint_index.popularity.get(x[0], 0)))
        return [asin for asin, _ in scored]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def build_constraint_query(session: dict, opening_message: str = "") -> str:
        """Build a semantic query string from the conversation history.

        Uses the accumulated user message history as the vector query — raw
        natural language captures semantic intent better than structured
        attribute strings (the embedding space is built from title/categories/
        description, not features/details, so attribute keywords like "alloy"
        have low semantic similarity).

        Falls back to the opening message (category fragment) when history is
        empty (turn 1 before history is populated).
        """
        history = session.get("history") or []
        user_turns = [
            str(h.get("content", ""))
            for h in history
            if h.get("role") == "user" and h.get("content")
        ]
        if user_turns:
            return " ".join(user_turns).strip()

        # Turn 1: history not yet populated — use opening message category fragment
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
        ASINs absent from a list contribute 0 from that list.

        This allows giving higher weight to more reliable signals (e.g.,
        constraint matching is more reliable than BM25/vector for this dataset).
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
