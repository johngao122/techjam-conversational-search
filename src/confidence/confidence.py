"""Confidence function.

Confidence gates the *question*, never the products: every turn still returns a
top-10. The confidence score only informs whether the agent should attach a
clarifying ``ask_attribute``.

Signals (in measured order of value):

    s1 = max_coverage / max(1, n_constraints_known)   # constraints satisfied?
    s2 = 1 - min(1, top_tier_crowd / CROWD_SCALE)     # tie crowd = ambiguity
    s3 = min(1, n_constraints_known / TYPICAL_CARD)   # how much of a card we hold

    conf = W1*s1 + W2*s2 + W3*s3

Weights are the starting point; they are swept in the ship-gate step.
"""

from __future__ import annotations

from src.reranker.types import RankResult

# Starting weights (swept in ship-gate step). Must sum to 1.0.
W1 = 0.45
W2 = 0.35
W3 = 0.20

CROWD_SCALE = 50.0   # crowd size at which ambiguity signal saturates
TYPICAL_CARD = 4.0   # constraints on a "typical" intent card


def compute_confidence(rank: RankResult, n_constraints_known: int) -> tuple[float, str]:
    """Return ``(score, reason)`` for the current turn.

    ``score`` is clamped to [0, 1]. ``reason`` is a human-readable explanation
    for the demo/report.
    """
    if rank.pool_size <= 0:
        return 0.0, "no candidates in pool"

    s1 = rank.max_coverage / max(1, n_constraints_known)
    s1 = min(1.0, s1)

    s2 = 1.0 - min(1.0, rank.top_tier_crowd / CROWD_SCALE)

    s3 = min(1.0, n_constraints_known / TYPICAL_CARD)

    score = W1 * s1 + W2 * s2 + W3 * s3
    score = max(0.0, min(1.0, score))

    reason = (
        f"{rank.max_coverage} of {n_constraints_known} constraints satisfied, "
        f"{rank.top_tier_crowd} products tie at top coverage "
        f"(pool={rank.pool_size})"
    )
    return score, reason
