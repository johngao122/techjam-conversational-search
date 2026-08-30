"""Output component: shape pipeline results into the Agent API contract dict.

Every ``respond`` turn must return a dict with exactly these keys (see
``docs/agent_api_contract.json``)::

    {
        "message": str,
        "ask_attribute": str | None,
        "recommendations": [{"parent_asin": str}, ...],
        "usage": {"prompt_tokens": int, "completion_tokens": int},
    }

:class:`OutputFormatter` centralizes that shaping so the orchestrator never
hand-builds the contract inline.
"""

from src.output.followup import FollowUpContext
from src.output.formatter import OutputFormatter

__all__ = ["OutputFormatter", "FollowUpContext"]
