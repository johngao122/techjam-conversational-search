"""Interactive REPL to try the full Agent pipeline on your own input.

Unlike src/message_parser/try_it.py (parser only), this runs a real session
end to end: Intent Router -> Ledger -> Retrieval/Rerank -> Confidence ->
Output -- exactly the path evaluator/local_evaluator.py drives, just with you
typing the customer's side instead of the simulator.

Run from the repo root:
    python3 scripts/try_agent.py                       # shipped defaults
    python3 scripts/try_agent.py --retrieval legacy    # any mechanism swapped
    python3 scripts/try_agent.py --parser llm          # needs DOCKER_MODEL_* vars
    python3 scripts/try_agent.py --no-exposure --theta 0.3

Every alternative mechanism is a flag (see --help); unset flags fall back to
the env var, then the shipped default. The active configuration is printed at
startup and on demand via the `config` command.

Turn numbers auto-increment per session, starting at 1. Commands:
    reset   start a new session (fresh turn counter, fresh ledger state)
    config  reprint the active mode configuration
    json    toggle showing the raw API response dict alongside the chat view
    quit    exit (Ctrl+D also works)
Anything else is sent as the customer's message for the current turn.
"""

from __future__ import annotations

import argparse
import json as json_module
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agent import Agent  # noqa: E402
from src.catalog.loader import load_catalog_rows  # noqa: E402
from src.config import (  # noqa: E402
    ASK_POLICIES,
    OVERRIDE_POLICIES,
    PARSERS,
    RETRIEVAL_MODES,
    AgentConfig,
)

# A representative user_profile shape (see docs/agent_api_contract.json) --
# doesn't need to be realistic, the pipeline only reads it for the
# rating_style tie-break and preference_tags nudge in the reranker.
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


def _new_session(agent: Agent, profile: dict, config: AgentConfig) -> tuple[str, int]:
    session_id = f"try_{uuid.uuid4().hex[:8]}"
    agent.reset(session_id, dict(profile))
    print(f"\n--- new session: {session_id} ---")
    print(f"    {config.banner()}\n")
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


def _build_config(args: argparse.Namespace) -> AgentConfig:
    """Start from the environment, then apply any explicitly-passed flags."""
    overrides: dict[str, object] = {}
    if args.parser is not None:
        overrides["parser"] = args.parser
    if args.retrieval is not None:
        overrides["retrieval_mode"] = args.retrieval
    if args.ask_policy is not None:
        overrides["ask_policy"] = args.ask_policy
    if args.exposure is not None:
        overrides["exposure_gate"] = args.exposure
    if args.release_turn is not None:
        overrides["release_turn"] = args.release_turn
    if args.override_policy is not None:
        overrides["override_policy"] = args.override_policy
    if args.idf:
        overrides["idf_weight"] = True
    if args.theta is not None:
        overrides["theta"] = args.theta
    return AgentConfig.from_env().replace(**overrides)


def _load_profile(path: str | None) -> dict:
    if not path:
        return dict(_DEFAULT_PROFILE)
    data = json_module.loads(Path(path).read_text(encoding="utf-8"))
    # Accept either a bare user_profile or a full public_set sample.
    return data.get("user_profile", data)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--parser", choices=PARSERS, default=None,
                   help="message parser: rule (default) or llm (needs DOCKER_MODEL_* vars)")
    p.add_argument("--retrieval", choices=RETRIEVAL_MODES, default=None,
                   help="retrieval/rerank core: bucket (default) or legacy BM25+coverage")
    p.add_argument("--ask-policy", choices=ASK_POLICIES, default=None,
                   help="clarification-ask policy")
    exposure = p.add_mutually_exclusive_group()
    exposure.add_argument("--exposure", dest="exposure", action="store_true", default=None,
                          help="force the turn-1/2 single-candidate exposure gate on")
    exposure.add_argument("--no-exposure", dest="exposure", action="store_false",
                          help="reveal the full list every turn")
    p.add_argument("--release-turn", type=int, default=None,
                   help="turn from which the full list is always revealed (default 3)")
    p.add_argument("--override-policy", choices=OVERRIDE_POLICIES, default=None,
                   help="constraint supersession policy on intent override")
    p.add_argument("--idf", action="store_true",
                   help="IDF-weight verbatim constraint matches (default off)")
    p.add_argument("--theta", type=float, default=None,
                   help="confidence threshold for ask-vs-recommend (default 0.5)")
    p.add_argument("--profile", default=None,
                   help="path to a user_profile JSON (or a public_set sample); default is a built-in profile")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    config = _build_config(args)
    profile = _load_profile(args.profile)

    print("Building the agent (FTS5 index + catalog load)...")
    try:
        agent = Agent(config=config)
    except RuntimeError as exc:
        # Most likely: --parser llm without the DOCKER_MODEL_* env vars set.
        print(f"\nCould not build the agent:\n  {exc}\n")
        if config.parser == "llm":
            print("Set DOCKER_MODEL_BASE_URL / _API_KEY / _NAME (see "
                  "src/message_parser/README.md), or run with --parser rule.")
        return 1
    titles = _load_titles()
    print("Ready.\n")
    print(f"Config: {config.banner()}")
    print("Type a customer message each turn ('reset' / 'config' / 'json' / 'quit'):\n")

    session_id, turn = _new_session(agent, profile, config)
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
            session_id, turn = _new_session(agent, profile, config)
            continue
        if text.lower() == "config":
            print(f"  {config.banner()}\n")
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

    return 0


if __name__ == "__main__":
    sys.exit(main())
