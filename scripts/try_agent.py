"""Interactive terminal chat to try the full Agent pipeline on your own input.

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

_TITLE_MAX = 66

# Plain ANSI codes -- no extra dependency. Disabled automatically when
# stdout isn't a real terminal (e.g. piped into a file or CI log).
_COLOR = sys.stdout.isatty()


def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _COLOR else text


def _bold(text: str) -> str:
    return _c("1", text)


def _dim(text: str) -> str:
    return _c("2", text)


def _cyan(text: str) -> str:
    return _c("36", text)


def _green(text: str) -> str:
    return _c("32", text)


def _yellow(text: str) -> str:
    return _c("33", text)


def _magenta(text: str) -> str:
    return _c("35", text)


def _load_products(catalog_path: str = "data/catalog.jsonl") -> dict[str, dict]:
    return {row["parent_asin"]: row for row in load_catalog_rows(catalog_path)}


def _new_session(agent: Agent) -> tuple[str, int]:
    session_id = f"try_{uuid.uuid4().hex[:8]}"
    agent.reset(session_id, dict(_DEFAULT_PROFILE))
    print(_dim(f"\n--- new session: {session_id} ---\n"))
    return session_id, 1


def _format_price(price: object) -> str | None:
    if price in (None, ""):
        return None
    try:
        return f"${float(price):.2f}"
    except (TypeError, ValueError):
        return str(price)


def _format_rating(product: dict) -> str | None:
    rating = product.get("average_rating")
    count = product.get("rating_number")
    if rating in (None, ""):
        return None
    stars = "★" * round(float(rating)) + "☆" * (5 - round(float(rating)))
    count_str = f" ({count:,})" if isinstance(count, (int, float)) else ""
    return f"{stars} {rating}{count_str}"


def _print_card(rank: int, asin: str, product: dict | None) -> None:
    title = (product or {}).get("title") or "(title unavailable)"
    if len(title) > _TITLE_MAX:
        title = title[: _TITLE_MAX - 1] + "…"
    meta_bits = []
    if product:
        price = _format_price(product.get("price"))
        if price:
            meta_bits.append(price)
        rating = _format_rating(product)
        if rating:
            meta_bits.append(rating)
        store = product.get("store")
        if store:
            meta_bits.append(str(store))
    meta = _dim("  ·  ".join(meta_bits)) if meta_bits else ""
    print(f"  {_bold(f'{rank}.')} {title}")
    print(f"     {_dim(asin)}" + (f"  {meta}" if meta else ""))


def _print_turn(response: dict, products: dict[str, dict]) -> None:
    recs = response.get("recommendations") or []
    if recs:
        print(_dim(f"  {len(recs)} recommendation(s):"))
        for i, rec in enumerate(recs, start=1):
            asin = rec.get("parent_asin", "?") if isinstance(rec, dict) else rec
            _print_card(i, asin, products.get(asin))
    else:
        print(_dim("  (no recommendations this turn)"))
    print()
    print(f"{_green(_bold('Agent'))}: {response.get('message', '')}")
    if response.get("ask_attribute"):
        print(_yellow(f"  ↳ asking about: {response['ask_attribute']}"))
    print()


def main() -> None:
    print(_dim("Building the agent (FTS5 index + catalog load)..."))
    agent = Agent()
    products = _load_products()
    print(_dim("Ready.\n"))
    print(_bold("Shopping Copilot -- terminal chat"))
    print(_dim("Type a message each turn. Commands: 'reset', 'json', 'quit'.\n"))

    session_id, turn = _new_session(agent)
    show_json = False

    while True:
        try:
            text = input(f"{_cyan(_bold(f'You [turn {turn}]'))}: ").strip()
        except EOFError:
            print()
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
            print(_dim(f"(raw JSON display {'on' if show_json else 'off'})\n"))
            continue

        response = agent.respond(session_id, text, turn, top_k=10)
        _print_turn(response, products)
        if show_json:
            print(_magenta(json_module.dumps(response, indent=2)))
            print()

        turn += 1
        if turn > 10:
            print(_dim("Session hit the 10-turn cap -- type 'reset' to start a new one.\n"))


if __name__ == "__main__":
    sys.exit(main())
