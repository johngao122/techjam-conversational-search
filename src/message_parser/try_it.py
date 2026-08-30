"""Interactive REPL to try the parser on your own text.

Run from the repo root:
    python3 -m src.message_parser.try_it

Type a message, see the extraction, repeat. Ctrl+D or "quit" to exit.
"""

from __future__ import annotations

import json
import os
import sys

# Load .env file
from pathlib import Path
env_file = Path(".env")
if env_file.exists():
    for line in env_file.read_text().strip().split("\n"):
        if line and "=" in line:
            key, value = line.split("=", 1)
            os.environ[key.strip()] = value.strip()

from .catalog_vocab import load_catalog_vocab
from .parser import MessageParser


def main() -> None:
    try:
        categories, brands = load_catalog_vocab("data/catalog.jsonl")
        print(f"Loaded catalog vocab: {len(categories)} categories, {len(brands)} brands.")
    except FileNotFoundError:
        categories, brands = set(), set()
        print("data/catalog.jsonl not found — running without category/brand matching.")

    parser = MessageParser(known_categories=categories, known_brands=brands)
    print("Type a customer message (or 'quit'):\n")

    # Test with conversation summarization
    print("=" * 70)
    print("TESTING WITH CONVERSATION SUMMARIZATION")
    print("=" * 70)
    print()

    from src.agent import Agent

    agent = Agent()
    summarizer = agent._summarizer
    print(f"✓ {type(summarizer).__name__} initialized (model: {summarizer._model})\n")

    # Conversation history tracking
    session_id = "test_session"
    user_id = "test_user"
    agent.reset(session_id, {"user_id": user_id})
    
    turn = 1
    while True:
        try:
            text = input("> ").strip()
        except EOFError:
            break
        if not text or text.lower() in {"quit", "exit"}:
            break
        
        # Parse the message
        parsed = parser.parse(text)
        print("\n--- MESSAGE PARSING ---")
        print(json.dumps(parsed.to_dict(), indent=2))
        
        # Add to conversation and summarize
        print("\n--- CONVERSATION SUMMARIZATION ---")
        with agent._ledger.session(session_id) as s:
            s.setdefault("history", []).append({
                "turn": turn,
                "role": "user",
                "content": text
            })
            s["intent"] = parsed.intent
            # Handle intent overrides: clear previous constraints and set new ones
            if parsed.is_override:
                s["constraints"] = {}
            # Merge new attributes with existing constraints (only add/update what was extracted)
            for attr, value in parsed.attributes.items():
                if value is not None:  # Only add non-None attributes
                    s.setdefault("constraints", {})[attr] = [value]

        # Call summarizer
        ledger = agent._ledger.read(session_id)
        summary = summarizer.summarize(
            last_user_message=text,
            previous_summary=ledger.get("conversation_summary", ""),
        )
        agent._ledger.set_conversation_summary(session_id, summary)

        # Update search key from summary
        search_key = agent._build_summary_search_key(summary)
        agent._ledger.set_search_key(session_id, search_key)

        print(f"Summary: {summary or '(not yet)'}")
        print(f"Search Key: '{search_key.get('_string', '')}'")
        print()

        turn += 1


if __name__ == "__main__":
    sys.exit(main())
