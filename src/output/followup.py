"""Situational follow-up message generation (hardcoded, first pass).

Under the shipped ``always_ask`` policy, the actual ``ask_attribute`` the API
contract returns is always the "other" wildcard -- that's what earns the
score (see ``src/confidence/policy.py::decide_specific_attribute``'s docstring
for the measured reason it must stay that way). This module gives the
*message text* real content anyway, in two ways:

- ``build_ask_message`` + ``context.topic``: one specific attribute per
  question (from ``next_unasked_topic``, cycling, never repeating).
- ``build_all_missing_ask_message`` + ``context.missing_attrs``: every
  attribute not yet disclosed, bundled into one question (from
  ``missing_topics``), dynamically shrinking each turn as more becomes
  known. Once nothing is missing, pivots to a ranking-preference closer
  (ratings vs. popularity) instead of continuing to ask about attributes --
  cosmetic only: the reranker's rating/popularity tie-break already runs
  unconditionally regardless of customer input.

Both are message-phrasing options over the same underlying signals; neither
touches the contract's ``ask_attribute`` field.

Note: the evaluator's ``customer_reply()`` (``evaluator/local_evaluator.py``)
decides what to reveal next turn purely from the contract's ``ask_attribute``
-- it never reads ``message`` text, and neither ``topic`` nor
``missing_attrs`` touch that field. This module therefore has zero effect on
HitRate@10/MRR/MTTC/TechnicalScore; it is a demo/UX quality improvement, not
a scoring lever. Kept deliberately separate from the confidence *decision*
(``policy.py``) so a change here can never regress the champion
``always_ask`` arm.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class FollowUpContext:
    """Situational signals already computed every turn in ``src/agent.py``,
    threaded through so message selection doesn't need to recompute or
    guess at anything.

    Attributes:
        scenario: One of "buying" / "browsing" / "intent_override" /
            "boundary", from ``src.intent_router.detect_scenario``.
        n_constraints_known: ``SessionLedger.n_constraints_known`` -- how many
            distinct constraints the customer has disclosed so far.
        exhausted: ``SessionLedger.exhausted`` -- customer signaled no further
            preferences exist.
        turn: Current turn number (1-indexed).
        override_seen: ``SessionLedger.override_seen`` -- an intent override
            has occurred at some point this session (used to soften the
            "still nothing?" late-turn phrasing after a genuine pivot).
        topic: A specific attribute to phrase the question around (e.g.
            "color"), from ``next_unasked_topic`` -- used by
            ``build_ask_message``. ``None`` falls through to situational
            phrasing.
        missing_attrs: Every attribute not yet disclosed, weighted-priority
            order, from ``missing_topics`` -- used by
            ``build_all_missing_ask_message``. Empty means everything is
            covered.
    """

    scenario: str
    n_constraints_known: int
    exhausted: bool
    turn: int
    override_seen: bool = False
    topic: str | None = None
    missing_attrs: tuple[str, ...] = field(default_factory=tuple)


# Late-turn threshold: past this, phrasing shifts to acknowledge we're running
# low on turns rather than opening with a fresh discovery question.
_LATE_TURN = 6

# Specific-attribute question text, used when `topic` names a real attribute
# (build_ask_message) or when only one attribute is left missing
# (build_all_missing_ask_message).
_ATTRIBUTE_QUESTIONS = {
    "category": "What type of item are you looking for?",
    "material": "Do you have a material preference?",
    "color": "Any color in mind?",
    "size": "What size do you need?",
    "style": "What style are you going for?",
    "brand": "Any brand you prefer?",
    "budget": "What's your budget?",
    "feature": "Any specific features in mind?",
    "use_case": "What will you mainly use it for?",
}

# Short labels for bundling several attributes into one sentence -- distinct
# from the standalone questions above, since "Do you have a preference for
# the type of item, material, color, and size?" needs noun-phrase fragments,
# not full questions, to read naturally when several are joined.
_ATTRIBUTE_LABELS = {
    "category": "the type of item",
    "material": "material",
    "color": "color",
    "size": "size",
    "style": "style",
    "brand": "brand",
    "budget": "budget",
    "feature": "any specific features",
    "use_case": "what you'll use it for",
}

# -- Situational fallback (used when topic is None -- every one of the 9
# specific attributes has already been suggested this session, or the caller
# passed no context at all). Single-focus, open-ended -- never enumerates
# several attribute names in one sentence, same rule as the topic-specific
# questions above: one thing asked at a time, not several at once. -----------

_ASK_INTENT_OVERRIDE = "Got it, updating my search based on that! Is there anything else that would help?"
_ASK_BOUNDARY = "No worries, I'll use my judgment there. Anything else you'd like me to keep in mind?"
_ASK_ZERO_INFO = "I'd love to help you find the right thing! Could you tell me a bit more about what you're looking for?"
_ASK_LATE_TURN = "We're getting close! Is there anything specific that would help me lock in the best match?"
_ASK_DEFAULT = "Thanks! Is there anything else that would help me narrow this down?"

# -- Lead-ins for build_all_missing_ask_message -------------------------------

_ASK_INTENT_OVERRIDE_LEAD = "Got it, updating my search based on that!"
_ASK_BOUNDARY_LEAD = "No worries, I'll use my judgment there."
_ASK_LATE_TURN_LEAD = "We're getting close!"

# All attributes covered: pivot to a ranking-preference closer instead of
# continuing to ask about search-narrowing attributes. Cosmetic only -- see
# module docstring; the reranker's rating/popularity tie-break already runs
# unconditionally regardless of what's answered here.
_ASK_ALL_COVERED = "I think I have everything I need! Do you tend to prefer higher-rated options, or more popular picks?"

# -- Recommend-only variants (used when payload.clarify is False) ------------

_RECOMMEND_EXHAUSTED = "Based on everything you've shared, here's what I found!"
_RECOMMEND_LATE_TURN = "Here's my best selection based on our conversation so far!"
_RECOMMEND_DEFAULT = "Here are the closest matches I found."


def build_ask_message(context: FollowUpContext | None) -> str:
    """Message text for a turn where the agent is asking a follow-up
    question, phrased around one specific attribute (``context.topic``).
    Falls back to the generic default with no context.

    A ``topic`` (an unused attribute suggested for phrasing) takes priority
    and is layered with a situational lead-in for override/boundary turns;
    no topic falls through to purely situational phrasing.
    """
    if context is None:
        return _ASK_DEFAULT

    question = _ATTRIBUTE_QUESTIONS.get(context.topic or "")

    if context.scenario == "intent_override":
        if question:
            return f"Got it, updating my search based on that! {question}"
        return _ASK_INTENT_OVERRIDE
    if context.scenario == "boundary":
        if question:
            return f"No worries, I'll use my judgment there. {question}"
        return _ASK_BOUNDARY
    if question and context.n_constraints_known == 0:
        # First question, nothing known yet: the attribute question alone is
        # a natural opener -- no filler needed.
        return question
    if context.n_constraints_known == 0:
        return _ASK_ZERO_INFO
    if question and context.turn >= _LATE_TURN:
        return f"We're getting close! {question}"
    if question:
        return question
    if context.turn >= _LATE_TURN:
        return _ASK_LATE_TURN
    return _ASK_DEFAULT


def _join_labels(attrs: tuple[str, ...]) -> str:
    labels = [_ATTRIBUTE_LABELS.get(a, a) for a in attrs]
    if len(labels) == 1:
        return labels[0]
    if len(labels) == 2:
        return f"{labels[0]} and {labels[1]}"
    return ", ".join(labels[:-1]) + f", and {labels[-1]}"


def build_all_missing_ask_message(context: FollowUpContext | None) -> str:
    """Message text for a turn where the agent asks about every attribute
    still missing (``context.missing_attrs``), bundled into one question.

    Recomputed fresh from what's known every turn -- an attribute stays in
    the question until it's actually disclosed, not just asked once. Layered
    with a situational lead-in for override/boundary/late-turn, same as
    ``build_ask_message``. Once nothing is missing, pivots to a
    ranking-preference closer instead of a narrowing question. Falls back to
    the generic default with no context.
    """
    if context is None:
        return _ASK_DEFAULT

    if not context.missing_attrs:
        return _ASK_ALL_COVERED

    question = f"Could you share your preference for {_join_labels(context.missing_attrs)}?"

    if context.scenario == "intent_override":
        return f"{_ASK_INTENT_OVERRIDE_LEAD} {question}"
    if context.scenario == "boundary":
        return f"{_ASK_BOUNDARY_LEAD} {question}"
    if context.turn >= _LATE_TURN:
        return f"{_ASK_LATE_TURN_LEAD} {question}"
    return question


def build_recommend_message(context: FollowUpContext | None) -> str:
    """Message text for a turn where the agent is recommending without
    asking (clarify=False). Falls back to the generic default with no
    context."""
    if context is None:
        return _RECOMMEND_DEFAULT
    if context.exhausted:
        return _RECOMMEND_EXHAUSTED
    if context.turn >= _LATE_TURN:
        return _RECOMMEND_LATE_TURN
    return _RECOMMEND_DEFAULT
