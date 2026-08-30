"""Interactive REPL to try the parser + Ollama summarizer.

Run from the repo root:
    python3 -m src.message_parser.try_it

Type a message, see the extraction, repeat. Ctrl+D or "quit" to exit.
"""

from __future__ import annotations

import json
import sys

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

    from src.ledger.ollama_summarizer import OllamaSummarizer
    summarizer = OllamaSummarizer(model="phi3:mini")
    print(f"Summarizer: {type(summarizer).__name__} (model: {summarizer._model})")
    print("\nType a customer message (or 'quit'):\n")

    previous_summary = ""
    while True:
        try:
            text = input("> ").strip()
        except EOFError:
            break
        if not text or text.lower() in {"quit", "exit"}:
            break

        parsed = parser.parse(text)
        print("\n--- PARSING ---")
        print(json.dumps(parsed.to_dict(), indent=2))

        print("\n--- SUMMARIZATION ---")
        previous_summary = summarizer.summarize(
            last_user_message=text,
            previous_summary=previous_summary,
        )
        print(f"Summary: {previous_summary or '(none)'}")
        print()


if __name__ == "__main__":
    sys.exit(main())
