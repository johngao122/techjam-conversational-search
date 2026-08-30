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
        print("(gzip -dk data/catalog.jsonl.gz && mv data/catalog.jsonl data/catalog.jsonl to enable it)")

    # Use regular MessageParser (no Docker model needed)
    parser = MessageParser(known_categories=categories, known_brands=brands)
    print("Type a customer message (or 'quit'):\n")

    # NEW: Test with conversation summarization
    print("=" * 70)
    print("TESTING WITH CONVERSATION SUMMARIZATION")
    print("=" * 70)
    print()
    
    from src.agent import Agent
    from src.ledger.llm_summarizer import LLMSummarizer
    
    agent = Agent()
    summarizer = agent._summarizer
    
    api_key = os.environ.get("API_KEY", "").replace("sk-", "sk-...")[:20]
    if not summarizer._client:
        print("⚠️  DeepSeek client not initialized.")
        print("  Check API_KEY env var (currently: {})".format(api_key))
        print("  Summaries will be empty, but message parsing will work.")
        print()
    else:
        print("✓ DeepSeek summarizer initialized")
        print(f"  API Key: {api_key}...")
        print(f"  Model: {summarizer._model}")
        print(f"  Base URL: {summarizer._client.base_url}")
        print()

    # Conversation history tracking
    session_id = "test_session"
    user_id = "test_user"
    agent.reset(session_id, {"user_id": user_id})
    
    turn = 1
    print("-" * 70)
    print()
    
    while True:
        try:
            text = input("> ").strip()
        except EOFError:
            break
        if not text or text.lower() in {"quit", "exit"}:
            break
        
        # Parse the message
        parsed = parser.parse(text)
        print("\n📝 MESSAGE PARSING:")
        print("-" * 70)
        print(json.dumps(parsed.to_dict(), indent=2))
        
        # Add to conversation and summarize
        print("\n📋 CONVERSATION STATE:")
        print("-" * 70)
        with agent._ledger.session(session_id) as s:
            s.setdefault("history", []).append({
                "turn": turn,
                "role": "user",
                "content": text
            })
            s["intent"] = parsed.intent
            for attr, value in parsed.attributes.items():
                s.setdefault("constraints", {})[attr] = [value]
        
        # Call summarizer
        ledger = agent._ledger.read(session_id)
        summary = summarizer.summarize(
            history=ledger.get("history", []),
            constraints=ledger.get("constraints", {}),
            intent=ledger.get("intent"),
        )
        agent._ledger.set_conversation_summary(session_id, summary)
        
        print(f"Summary: {summary.get('summary', '(not yet)') or '(not yet)'}")
        prefs = summary.get('remembered_preferences', {})
        if prefs:
            print(f"Preferences: {json.dumps(prefs)}")
        print(f"Topics: {summary.get('topics_covered', [])}")
        print()
        
        turn += 1


if __name__ == "__main__":
    sys.exit(main())
