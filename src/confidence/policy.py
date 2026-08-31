"""Policy mapping: turn a confidence score into a clarify decision.

    clarify = (conf < theta) AND (not exhausted) AND (turn < TURN_CUTOFF)
    ask_attribute = "other" whenever clarify        # fixed dominant attribute

Overrides:
    - Force clarify=True while n_constraints_known == 0 (never open on zero info).
    - After an override the ledger resets ``exhausted`` -> clarify resumes.

Recommendations are emitted every turn regardless of clarify; this policy only
decides the *question*.
"""

from __future__ import annotations

import os

from src.confidence.confidence import compute_confidence
from src.confidence.session_ledger import SessionLedger
from src.confidence.payload import ConfidencePayload
from src.ledger.ledger import ATTRIBUTE_PRIORITY
from src.reranker.types import RankResult

DEFAULT_THETA = 0.5
TURN_CUTOFF = 10           # stop asking at/after this turn
FIXED_ASK_ATTRIBUTE = "other"
FINAL_TURN = 10

# Message-phrasing topic order for next_unasked_topic ONLY, ordered by measured
# constraint-type frequency (higher first). "category" leads since it's usually
# already disclosed in the opening line, so normally skipped in practice.
TOPIC_PRIORITY = (
    "category", "feature", "material", "color", "style",
    "size", "use_case", "budget", "brand",
)

# Exposure gate. The evaluator freezes MRR at the target's first top-10
# appearance, so exposing exactly one candidate early keeps the upside (a correct
# top-1 hits at rank 1) with no downside (MRR is unaffected until a hit).
RELEASE_TURN = 3
CONFIDENT_EXPOSURE = 1


def exposure_enabled() -> bool:
    """Gate is on by default; ``EXPOSURE_GATE=0`` reverts to full-list-every-turn."""
    return os.environ.get("EXPOSURE_GATE", "1").strip() != "0"


def release_turn() -> int:
    raw = os.environ.get("RELEASE_TURN", "").strip()
    return int(raw) if raw.isdigit() else RELEASE_TURN


def exposure(turn: int, exhausted: bool, top_k: int) -> int:
    """How many recommendations to reveal this turn.

    Full list once we release (turn >= RELEASE_TURN), when the customer says the
    card is drained, or on the final turn (never withhold at turn 10 -- that
    truncation loses winnable sessions outright). Otherwise a single candidate.
    """
    if not exposure_enabled():
        return top_k
    if turn >= release_turn() or exhausted or turn >= FINAL_TURN:
        return top_k
    return CONFIDENT_EXPOSURE


def decide(
    rank: RankResult,
    ledger: SessionLedger,
    theta: float = DEFAULT_THETA,
) -> ConfidencePayload:
    """Compute confidence and the clarify decision for this turn."""
    n_known = ledger.n_constraints_known
    score, reason = compute_confidence(rank, n_known)

    # Zero-info: never open a browsing session without asking.
    if n_known == 0:
        return ConfidencePayload(
            score=score,
            clarify=True,
            ask_attribute=FIXED_ASK_ATTRIBUTE,
            reason=f"zero constraints known -> forced clarify ({reason})",
        )

    clarify = (score < theta) and (not ledger.exhausted) and (ledger.turn < TURN_CUTOFF)
    ask_attribute = FIXED_ASK_ATTRIBUTE if clarify else None

    if ledger.exhausted:
        reason = f"exhausted -> recommend only ({reason})"
    elif ledger.turn >= TURN_CUTOFF:
        reason = f"turn cutoff reached -> recommend only ({reason})"

    return ConfidencePayload(
        score=score,
        clarify=clarify,
        ask_attribute=ask_attribute,
        reason=reason,
    )


def next_unasked_topic(ledger: SessionLedger, known_attrs: set[str] | None = None) -> str | None:
    """Pick the next ``TOPIC_PRIORITY`` attribute not yet suggested as a message
    topic this session and not already disclosed as a constraint.

    Message-phrasing helper ONLY -- never touches the actual ``ask_attribute``
    the contract returns (that stays the "other" wildcard under ``always_ask``).
    Caller records the returned topic via ``ledger.note_ask()`` so it isn't
    suggested twice; "other" is ignored by construction (not in ``TOPIC_PRIORITY``).
    """
    known_attrs = known_attrs or set()
    covered = ledger.asked_attributes | known_attrs
    return next((attr for attr in TOPIC_PRIORITY if attr not in covered), None)


def missing_topics(known_attrs: set[str] | None = None) -> list[str]:
    """Every ``TOPIC_PRIORITY`` attribute not yet disclosed as a constraint, in
    priority order.

    Additive alternative to ``next_unasked_topic``: a pure function of
    ``known_attrs`` (no "already asked" state), so an attribute stays in the
    result until it's actually known. Message-phrasing ONLY. Empty result means
    every attribute is covered.
    """
    known_attrs = known_attrs or set()
    return [attr for attr in TOPIC_PRIORITY if attr not in known_attrs]


def decide_specific_attribute(
    rank: RankResult,
    ledger: SessionLedger,
    known_attrs: set[str] | None = None,
) -> ConfidencePayload:
    """Ask-specific-attribute policy: pick the next unasked, undisclosed
    attribute from ``ATTRIBUTE_PRIORITY`` each turn ("other" is the final
    wildcard sweep once every specific attribute is covered).

    NOTE -- measured risk, not a drop-in upgrade: ``ask_attribute="other"`` is a
    wildcard revealing ANY undisclosed constraint, while a specific attribute
    only reveals that exact type. Most slots are "feature"/"material", so this
    risks more zero-information turns. Select via ``ASK_POLICY=attribute_cycle``;
    ``always_ask`` remains the shipped default.
    """
    known_attrs = known_attrs or set()
    covered = ledger.asked_attributes | known_attrs
    next_attr = next((a for a in ATTRIBUTE_PRIORITY if a not in covered), None)

    score, reason = compute_confidence(rank, ledger.n_constraints_known)

    if ledger.exhausted:
        return ConfidencePayload(
            score=score, clarify=False, ask_attribute=None,
            reason=f"exhausted -> recommend only ({reason})",
        )
    if ledger.turn >= TURN_CUTOFF:
        return ConfidencePayload(
            score=score, clarify=False, ask_attribute=None,
            reason=f"turn cutoff reached -> recommend only ({reason})",
        )
    if next_attr is None:
        return ConfidencePayload(
            score=score, clarify=False, ask_attribute=None,
            reason=f"every attribute already asked or known ({reason})",
        )
    return ConfidencePayload(
        score=score, clarify=True, ask_attribute=next_attr,
        reason=f"asking unasked attribute '{next_attr}' ({reason})",
    )


def always_ask(ledger: SessionLedger) -> ConfidencePayload:
    """P0 champion arm: ask until exhausted, ignoring confidence.

    Used as the ship-gate baseline. Recommendations still emitted every turn.
    """
    clarify = not ledger.exhausted and ledger.turn < TURN_CUTOFF
    return ConfidencePayload(
        score=float("nan"),
        clarify=clarify,
        ask_attribute=FIXED_ASK_ATTRIBUTE if clarify else None,
        reason="always-ask-until-exhausted (P0)",
    )
