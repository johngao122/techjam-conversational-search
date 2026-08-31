"""Uniform retrieval boundary.

Every retrieval method presents the same shape to the reranker: a
:class:`RetrievalStrategy` turning a :class:`RetrievalRequest` into a
:class:`RetrievalResult`. The reranker consumes only that result, ignorant of
how the pool was produced. New methods implement the protocol and slot into
the pipeline with no reranker change.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class RetrievalRequest:
    """Everything a retrieval strategy may need, in one shape.

    Attributes:
        opening_message: Turn-1 message; carries the coarse category the bucket
            resolver keys off.
        constraints: Verbatim disclosed constraint strings (literal slices of
            the target's metadata).
        transcript: Accumulated user turns, used by the BM25 fallback rung when
            the category fails to resolve.
        query: Pre-composed retrieval query. When set, a text strategy uses it
            verbatim instead of composing one from ``constraints``/``transcript``
            (the legacy BM25 path hands in an already-built query string).
        top_k: Number of ranked candidates the caller ultimately wants.
        pool_size: Candidate-pool size a strategy should retrieve before the
            reranker scores/truncates. Defaults to ``top_k`` when unset (0).
    """

    opening_message: str = ""
    constraints: list[str] = field(default_factory=list)
    transcript: str = ""
    query: str | None = None
    top_k: int = 10
    pool_size: int = 0
    rating_style: str | None = None


@dataclass(frozen=True)
class RetrievalResult:
    """Ordered candidate pool plus advisory metadata.

    Attributes:
        candidates: Ordered ``parent_asin`` strings (best-to-worst / retrieval
            order). This is the sole payload the reranker requires.
        resolved: False when the strategy fell back to the whole catalog
            (category could not be resolved to a bucket).
        resolved_exact: True when the coarse category resolved by exact match
            (the clean-public-set signal; drives paraphrase insurance).
        how: Names the rung that produced the pool, for A/B logging.
    """

    candidates: list[str] = field(default_factory=list)
    resolved: bool = True
    resolved_exact: bool = True
    how: str = ""


@runtime_checkable
class RetrievalStrategy(Protocol):
    """A candidate generator. The single extension point for new methods."""

    def candidates(self, request: RetrievalRequest) -> RetrievalResult:
        ...
