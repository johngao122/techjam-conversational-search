"""Retrieval strategies: one thin wrapper per method, plus a composite.

Each of the three retrieval methods is wrapped as a :class:`RetrievalStrategy`
so the reranker consumes a single uniform :class:`RetrievalResult` shape:

  * :class:`Bm25Strategy`       -- full-text BM25 over the whole catalog
  * :class:`BucketStrategy`     -- coarse-category bucket lookup
  * :class:`ConstraintStrategy` -- verbatim-constraint re-ordering of a pool
  * :class:`BucketPipeline`     -- composite reproducing the bucket-mode ladder

Caching that used to live on the ``Reranker`` moves here, next to the method it
belongs to, so the reranker owns no retrieval state.
"""

from __future__ import annotations

from src.retrieval.base import RetrievalRequest, RetrievalResult
from src.retrieval.buckets import BucketIndex
from src.retrieval.constraint_index import ConstraintIndex, is_inert, prepare
from src.retrieval.retrieval import Retriever

DEFAULT_POOL = 200

# Prepared constraint triples: (normalised, tokens, weight).
PreparedConstraints = list[tuple[str, tuple[str, ...], float]]


def prepare_constraints(constraints: list[str]) -> PreparedConstraints:
    """Normalise disclosed constraints into scoreable triples (weight 1.0)."""
    return [
        (norm, toks, 1.0)
        for norm, toks in (prepare(c) for c in constraints if not is_inert(c))
        if norm
    ]


class Bm25Strategy:
    """Full-text BM25 retrieval over the whole catalog.

    Owns the process-lifetime BM25 cache (the catalog is read-only for a run,
    so a cache hit is exactly the uncached value).
    """

    def __init__(self, retriever: Retriever, pool_size: int = DEFAULT_POOL) -> None:
        self._retriever = retriever
        self._pool_size = pool_size
        self._cache: dict[tuple[str, int], list[str]] = {}

    def search(self, query: str, top_k: int) -> list[str]:
        key = (query, top_k)
        hit = self._cache.get(key)
        if hit is None:
            hit = self._retriever.retrieve_bm25({"keywords": [query]}, top_k=top_k)
            self._cache[key] = hit
        return hit

    def candidates(self, request: RetrievalRequest) -> RetrievalResult:
        # A pre-composed query (legacy path) is used verbatim; otherwise compose
        # one from the disclosed constraints plus the running transcript.
        if request.query is not None:
            query = request.query
        else:
            query = " ".join([*request.constraints, request.transcript]).strip()
        top_k = request.pool_size or self._pool_size
        pool = self.search(query, top_k)
        return RetrievalResult(candidates=pool, how="bm25")


class BucketStrategy:
    """Coarse-category bucket lookup.

    Owns the per-session resolved-bucket cache keyed by opening message (the
    category is disclosed once on turn 1 and holds for the whole session).
    """

    def __init__(self, bucket_index: BucketIndex, constraint_index: ConstraintIndex) -> None:
        self._bucket_index = bucket_index
        self._constraint_index = constraint_index
        # opening_message -> (pool, resolved, resolved_exact)
        self._cache: dict[str, tuple[list[str], bool, bool]] = {}

    def candidates(self, request: RetrievalRequest) -> RetrievalResult:
        opening = request.opening_message
        cached = self._cache.get(opening)
        if cached is None:
            key, how = self._bucket_index.resolve(opening)
            pool = self._bucket_index.get(key) if key else []
            resolved = True
            resolved_exact = how == "exact"
            if not pool:
                resolved = False
                pool = list(self._constraint_index.attributes.keys())
            self._cache[opening] = (pool, resolved, resolved_exact)
        else:
            pool, resolved, resolved_exact = cached
        return RetrievalResult(
            candidates=pool,
            resolved=resolved,
            resolved_exact=resolved_exact,
            how="bucket" if resolved else "whole-catalog",
        )


class ConstraintStrategy:
    """Verbatim-constraint re-ordering of an incoming candidate pool."""

    def __init__(self, constraint_index: ConstraintIndex) -> None:
        self._constraint_index = constraint_index

    def rank(self, pool: list[str], prepared: PreparedConstraints, top_k: int, rating_style: str | None = None) -> list[str]:
        return self._constraint_index.rank(pool, prepared, top_k, rating_style=rating_style)

    def score(self, asin: str, prepared: PreparedConstraints) -> float:
        return self._constraint_index.score(asin, prepared)


class BucketPipeline:
    """Composite strategy: the bucket-mode robustness ladder.

    Composes the three methods so the reranker never sees the branching:
      1. resolved bucket + verbatim constraints  (primary path)
      2. resolved bucket + popularity            (no constraint matched)
      3. whole catalog, narrowed by BM25 over the transcript (category
         unresolved -- template/paraphrase drift on the private set)
      4. whole catalog + popularity              (nothing else fired)

    Returns a :class:`RetrievalResult` whose ``candidates`` are already ranked
    (constraint score, then popularity). The prepared constraints actually used
    are exposed via :attr:`last_prepared` so the reranker can compute coverage
    without re-deriving them.
    """

    def __init__(
        self,
        bucket: BucketStrategy,
        constraint: ConstraintStrategy,
        bm25: Bm25Strategy,
    ) -> None:
        self._bucket = bucket
        self._constraint = constraint
        self._bm25 = bm25
        # Advisory outputs from the most recent candidates() call, for the
        # reranker's coverage / pool-size computation (read-only).
        self.last_prepared: PreparedConstraints = []
        self.last_pool_size: int = 0

    def candidates(self, request: RetrievalRequest) -> RetrievalResult:
        bucket_result = self._bucket.candidates(request)
        pool = bucket_result.candidates

        # Rung 3: category unresolved -> narrow the whole-catalog pool with a
        # BM25 pass over the accumulated transcript. Only when a transcript is
        # available; otherwise fall straight through to popularity.
        if not bucket_result.resolved and request.transcript.strip():
            bm25_pool = self._bm25.search(request.transcript, DEFAULT_POOL)
            if bm25_pool:
                pool = bm25_pool

        prepared = prepare_constraints(request.constraints)

        # Paraphrase insurance: score the bucket by transcript-token overlap
        # when the verbatim path produced nothing AND the opening category
        # itself failed to resolve exactly -- i.e. the template wording drifted.
        # Gating on the inexact-resolution signal keeps the clean public set
        # (where the category always resolves exactly and an exact constraint
        # match dominates) completely untouched.
        if not prepared and not bucket_result.resolved_exact and request.transcript.strip():
            _, t_toks = prepare(request.transcript)
            for tok in dict.fromkeys(t_toks):
                prepared.append((tok, (tok,), 0.15))

        self.last_prepared = prepared
        self.last_pool_size = len(pool)
        ranked = self._constraint.rank(pool, prepared, request.top_k, rating_style=request.rating_style)
        return RetrievalResult(
            candidates=ranked,
            resolved=bucket_result.resolved,
            resolved_exact=bucket_result.resolved_exact,
            how=bucket_result.how,
        )
