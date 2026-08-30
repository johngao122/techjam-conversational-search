"""OutputFormatter: assemble the Agent API turn_response dict.

Reads a :class:`~src.confidence.payload.ConfidencePayload` (the decision) plus a
ranked list of ``parent_asin`` strings and emits the frozen contract dict. It
never decides *whether* to ask -- that is the confidence component's job; it
only shapes the response and phrases the clarifying question.

Message phrasing is delegated to :mod:`src.output.followup`, which picks a
hardcoded line based on the conversational situation (vague opener, boundary
brush-off, intent override, late turn, ...) rather than a single static
sentence. See that module's docstring for why this has no effect on the
evaluator's score.
"""

from __future__ import annotations

from src.confidence.payload import ConfidencePayload
from src.output.followup import (
    FollowUpContext,
    build_all_missing_ask_message,
    build_ask_message,
    build_recommend_message,
)

# Legacy fallback: static phrasing per allowed ask_attribute (contract enum),
# used only when no FollowUpContext is supplied (keeps existing callers
# working). In the live path ask_attribute is always "other" (see
# src/confidence/policy.py), so entries beyond "other" are effectively dead
# code today -- kept for the alternate `decide()` policy and any caller that
# still wants attribute-keyed phrasing.
_QUESTION_BY_ATTRIBUTE = {
    "category": "What type of item are you looking for?",
    "material": "Do you have a material preference?",
    "color": "Any color in mind?",
    "size": "What size do you need?",
    "style": "What style are you going for?",
    "brand": "Any brand you prefer?",
    "budget": "What's your budget?",
    "feature": "Are there any features that matter most to you?",
    "use_case": "What will you mainly use it for?",
    "other": "Anything else that would help me narrow this down?",
}

_RECOMMEND_MESSAGE = "Here are the closest matches I found."
_DEFAULT_ASK_MESSAGE = "Could you tell me a bit more about what you're after?"


class OutputFormatter:
    """Shape pipeline results into the Agent API ``turn_response`` contract."""

    def format(
        self,
        payload: ConfidencePayload,
        recommendations: list[str],
        usage: dict | None = None,
        context: FollowUpContext | None = None,
    ) -> dict:
        """Build the contract dict from a decision payload and recommendations.

        Recommendations are always attached (every turn returns a top-10). When
        ``payload.clarify`` is set, a clarifying ``message`` + ``ask_attribute``
        are included; otherwise ``ask_attribute`` is ``None``.

        ``context`` (optional): situational signals for message phrasing (see
        :class:`~src.output.followup.FollowUpContext`). When omitted, falls
        back to the legacy static ``_QUESTION_BY_ATTRIBUTE``/``_RECOMMEND_MESSAGE``
        phrasing -- existing callers are unaffected.
        """
        recs = [{"parent_asin": asin} for asin in recommendations[:10]]

        if payload.clarify and payload.ask_attribute:
            if context is not None:
                message = build_all_missing_ask_message(context)
            else:
                message = _QUESTION_BY_ATTRIBUTE.get(
                    payload.ask_attribute, _DEFAULT_ASK_MESSAGE
                )
            ask_attribute = payload.ask_attribute
        else:
            message = build_recommend_message(context) if context is not None else _RECOMMEND_MESSAGE
            ask_attribute = None

        return {
            "message": message,
            "ask_attribute": ask_attribute,
            "recommendations": recs,
            "usage": usage or {"prompt_tokens": 0, "completion_tokens": 0},
        }
