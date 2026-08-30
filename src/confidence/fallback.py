"""Fallback path: never raise, always return a top-10.

When the reranker/retrieval yields an empty pool or throws, the agent must
still emit recommendations. We fall back to a popularity ordering computed once
from the catalog (rating_number desc, then average_rating desc).
"""

from __future__ import annotations

import heapq
from pathlib import Path

from src.catalog.loader import load_catalog_rows
from src.confidence.session_ledger import SessionLedger
from src.confidence.payload import ConfidencePayload
from src.confidence.policy import FIXED_ASK_ATTRIBUTE


def popularity_top10(catalog_path: str | Path) -> list[str]:
    """Compute the popularity fallback list (top 10 parent_asin)."""
    rows = (
        (
            float(p.get("rating_number") or 0),
            float(p.get("average_rating") or 0),
            str(p["parent_asin"]),
        )
        for p in load_catalog_rows(str(catalog_path))
    )
    # Same key and same descending order as the full sort it replaces, but a
    # partial selection: this only ever needs the top 10 of 50k.
    top = heapq.nlargest(10, rows, key=lambda r: (r[0], r[1]))
    return [asin for _, _, asin in top]


def safe_decide(
    rank_fn,
    ledger: SessionLedger,
    fallback_recs: list[str],
    theta: float,
    policy: str = "always_ask",
    known_attrs: set[str] | None = None,
) -> tuple[ConfidencePayload, list[str]]:
    """Run ranking + policy, guaranteeing no raise.

    ``rank_fn`` is a zero-arg callable returning a ``RankResult``. On any
    exception or empty pool we return the popularity fallback with conf=0 and
    clarify=True. ``policy`` selects the clarify decision: ``"always_ask"``
    (the ship-gate champion arm, see ``scripts/sweep_confidence.py``),
    ``"confidence"`` (the coverage-based ``decide`` heuristic, gated by
    ``theta``), or ``"attribute_cycle"`` (asks a specific, not-yet-asked
    attribute each turn instead of the fixed "other" wildcard -- see
    ``decide_specific_attribute``'s docstring for the measured tradeoff;
    ``known_attrs`` is only consulted by this policy). Returns
    ``(payload, recommendations)``.
    """
    # Local import to avoid cycles at import time.
    from src.confidence.policy import always_ask, decide, decide_specific_attribute

    try:
        rank = rank_fn()
    except Exception:
        rank = None

    if rank is None or rank.pool_size <= 0 or not rank.ranked:
        payload = ConfidencePayload(
            score=0.0,
            clarify=True,
            ask_attribute=FIXED_ASK_ATTRIBUTE,
            reason="empty pool / rank failure -> popularity fallback",
        )
        return payload, list(fallback_recs[:10])

    if policy == "always_ask":
        payload = always_ask(ledger)
    elif policy == "attribute_cycle":
        payload = decide_specific_attribute(rank, ledger, known_attrs=known_attrs)
    else:
        payload = decide(rank, ledger, theta=theta)
    return payload, list(rank.ranked[:10])
