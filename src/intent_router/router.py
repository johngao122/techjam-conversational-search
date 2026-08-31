from __future__ import annotations

import os

from src.message_parser import MessageParser, ParsedMessage, load_catalog_vocab

# Fallback only. ``warm_parser`` is the real entry point: ``Agent.__init__``
# passes the constructor's catalog path so the router indexes the same file the
# evaluator was pointed at, rather than whatever this env var happened to hold.
_CATALOG_PATH = os.environ.get("CATALOG_PATH", "data/catalog.jsonl")
_parser: MessageParser | None = None


def warm_parser(catalog_path: str | None = None) -> MessageParser:
    """Build the vocab-backed parser eagerly.

    The vocab scan costs seconds over a 50k-row catalog; ``Agent.__init__`` calls
    this so the cost is paid at construction rather than on the first ``respond()``.
    """
    global _parser
    if _parser is None:
        categories, brands = load_catalog_vocab(catalog_path or _CATALOG_PATH)
        _parser = MessageParser(known_categories=categories, known_brands=brands)
    return _parser


def _get_parser() -> MessageParser:
    return warm_parser()


def parse_message(message: str) -> ParsedMessage:
    """Parse once and return the full ParsedMessage (intent, category, product, attributes)."""
    return _get_parser().parse(message)


def detect_scenario(message: str, history: list[dict] | None = None) -> str:
    """
    Returns one of: 'buying', 'browsing', 'intent_override', 'boundary'
    """
    return parse_message(message).intent


def extract_attributes(message: str) -> dict:
    """
    Extract structured attributes from a user message.
    Returns a dict with keys matching KIV fields where detected.

    ``feature`` is filtered here for the legacy BM25 path. The bucket pipeline
    ranks against the verbatim ConstraintMemory instead, where the dropped
    ``feature`` strings are retained (see src/intent_router/constraint_memory.py).
    """
    return attributes_of(parse_message(message))


def attributes_of(parsed: ParsedMessage) -> dict:
    """``extract_attributes`` for a message that has already been parsed.

    ``Agent.respond`` needs both the intent and the attributes of the same
    string; going through ``detect_scenario`` + ``extract_attributes`` parsed
    it twice per turn.
    """
    return {k: v for k, v in parsed.attributes.items() if k != "feature"}


def build_search_key(session: dict) -> dict:
    search_key: dict = {}
    for attr, values in session["constraints"].items():
        search_key[attr] = values
    price_c = session.get("price_constraint")
    if price_c:
        if price_c["operator"] in ("<", "<=", "~"):
            search_key["price"] = [{"lte": price_c["amount"]}]
        else:
            search_key["price"] = [{"gte": price_c["amount"]}]
    return search_key
