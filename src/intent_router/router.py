from __future__ import annotations

import os

from src.message_parser import (
    LLMMessageParser,
    MessageParser,
    ParsedMessage,
    load_catalog_vocab,
)

# Fallback only. ``warm_parser`` is the real entry point: ``Agent.__init__``
# passes the constructor's catalog path so the router indexes the same file the
# evaluator was pointed at, rather than whatever this env var happened to hold.
_CATALOG_PATH = os.environ.get("CATALOG_PATH", "data/catalog.jsonl")

# The vocab scan is the multi-second cost; the parser instance on top of it is
# cheap. Cache them separately so switching ``kind`` (rule <-> llm) in one
# process -- the A/B REPL, tests -- does not re-scan the catalog.
_vocab: tuple[set[str], set[str]] | None = None
_vocab_path: str | None = None
_parser: MessageParser | LLMMessageParser | None = None
_parser_kind: str | None = None

# Parser interface shared by both implementations: ``.parse(text) -> ParsedMessage``,
# constructed with ``known_categories`` / ``known_brands``.
_PARSER_CLASSES: dict[str, type] = {"rule": MessageParser, "llm": LLMMessageParser}


def warm_parser(catalog_path: str | None = None, kind: str | None = None) -> MessageParser | LLMMessageParser:
    """Build the vocab-backed parser eagerly.

    ``kind`` is ``"rule"`` (regex/vocab, no dependencies) or ``"llm"``
    (:class:`~src.message_parser.LLMMessageParser`, needs the ``DOCKER_MODEL_*``
    env vars + a local OpenAI-compatible endpoint -- it raises ``RuntimeError``
    on construction if those are unset). ``None`` keeps the kind already built
    (``"rule"`` on the first ever call), so the no-arg ``parse_message`` path
    never silently reverts a caller's ``"llm"`` choice.

    The vocab scan costs seconds over a 50k-row catalog. Left lazy it landed
    inside the first ``respond()`` call, turning turn 1 of session 1 into a
    multi-second outlier; ``Agent.__init__`` calls this so the cost is paid at
    construction alongside the FTS5 index build instead. A later call with a
    different ``kind`` rebuilds only the parser, reusing the cached vocab.
    """
    global _vocab, _vocab_path, _parser, _parser_kind

    if kind is None:
        kind = _parser_kind or "rule"
    if kind not in _PARSER_CLASSES:
        raise ValueError(f"parser kind must be one of {tuple(_PARSER_CLASSES)}, got {kind!r}")

    path = catalog_path or _CATALOG_PATH
    if _vocab is None or _vocab_path != path:
        _vocab = load_catalog_vocab(path)
        _vocab_path = path
        _parser = None  # vocab changed underneath it

    if _parser is None or _parser_kind != kind:
        categories, brands = _vocab
        _parser = _PARSER_CLASSES[kind](known_categories=categories, known_brands=brands)
        _parser_kind = kind

    return _parser


def _get_parser() -> MessageParser | LLMMessageParser:
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

    ``feature`` is still filtered here for the legacy BM25 path (keeping it
    byte-identical to the recorded 0.680 control). The bucket pipeline does not
    consume these taxonomy attributes at all -- it ranks against the verbatim
    ConstraintMemory, which is exactly where the dropped ``feature`` strings are
    now retained (see src/intent_router/constraint_memory.py).
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
