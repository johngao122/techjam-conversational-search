"""Interactive REPL to try the full Agent pipeline on your own input.

Unlike src/message_parser/try_it.py (parser only), this runs a real session
end to end: Intent Router -> Ledger -> Retrieval/Rerank -> Confidence ->
Output -- exactly the path evaluator/local_evaluator.py drives, just with you
typing the customer's side instead of the simulator.

Run from the repo root:
    python3 scripts/try_agent.py

Turn numbers auto-increment per session, starting at 1. Commands:
    reset   start a new session (fresh turn counter, fresh ledger state)
    json    toggle showing the raw API response dict alongside the chat view
    quit    exit (Ctrl+D also works)
Anything else is sent as the customer's message for the current turn.
"""

from __future__ import annotations

import json as json_module
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agent import Agent  # noqa: E402
from src.catalog.loader import load_catalog_rows  # noqa: E402

# A representative user_profile shape (see docs/agent_api_contract.json) --
# doesn't need to be realistic, the pipeline only reads it for the
# rating_style tie-break in the reranker.
_DEFAULT_PROFILE = {
    "purchase_frequency": "3-4 prior purchases",
    "average_prior_rating": 4.2,
    "rating_style": "usually positive",
    "preference_tags": ["comfort", "fit"],
    "summary": "Prior purchases emphasize comfort and fit.",
}

_TITLE_MAX = 70


def _load_titles(catalog_path: str = "data/catalog.jsonl") -> dict[str, str]:
    return {row["parent_asin"]: row["title"] for row in load_catalog_rows(catalog_path)}


def _new_session(agent: Agent) -> tuple[str, int]:
    session_id = f"try_{uuid.uuid4().hex[:8]}"
    agent.reset(session_id, dict(_DEFAULT_PROFILE))
    print(f"\n--- new session: {session_id} ---\n")
    return session_id, 1


def _print_turn(response: dict, titles: dict[str, str]) -> None:
    for rec in response["recommendations"]:
        asin = rec["parent_asin"]
        title = titles.get(asin, "(title unavailable)")
        if len(title) > _TITLE_MAX:
            title = title[: _TITLE_MAX - 1] + "…"
        print(f"  • {title}  [{asin}]")
    if not response["recommendations"]:
        print("  (no recommendations this turn)")
    print(f"Agent: {response['message']}")
    print()


def main() -> None:
    print("Building the agent (FTS5 index + catalog load)...")
    agent = Agent()
    titles = _load_titles()
    print("Ready.\n")
    print("Type a customer message each turn ('reset' / 'json' / 'quit'):\n")

    session_id, turn = _new_session(agent)
    show_json = False

    while True:
        try:
            text = input(f"You [turn {turn}]: ").strip()
        except EOFError:
            break

        if not text:
            continue
        if text.lower() in {"quit", "exit"}:
            break
        if text.lower() == "reset":
            session_id, turn = _new_session(agent)
            continue
        if text.lower() == "json":
            show_json = not show_json
            print(f"(raw JSON display {'on' if show_json else 'off'})\n")
            continue

        response = agent.respond(session_id, text, turn, top_k=10)
        _print_turn(response, titles)
        if show_json:
            print(json_module.dumps(response, indent=2))
            print()

        turn += 1
        if turn > 10:
            print("Session hit the 10-turn cap -- type 'reset' to start a new one.\n")


if __name__ == "__main__":
    sys.exit(main())
