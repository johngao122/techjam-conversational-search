"""Curated embedding-document builder.

Turns a raw catalog product into a compact string capturing its core
semantics for vector search, complementary to BM25: BM25 owns the
keyword/attribute-dense fields, so the embed doc uses only the semantic
fields that survive paraphrase and vocabulary mismatch:

    title        -- the product name (concept + type)
    categories   -- taxonomy path (e.g. "Shirts > T-Shirts")
    description  -- a short leading slice of the natural-language description

``store``/brand, ``features`` and ``details`` are excluded (BM25 signal). We
also do NOT distil the description into parsed attributes: that reproduces
BM25's signal and the parser is noisy on long marketing prose.

The output is a plain string; embedding is handled elsewhere.
"""

from __future__ import annotations

import re

# Leading slice of the description field: leading sentences carry the
# semantic gist, and bounding it keeps docs inside a 512-token window.
_DESCRIPTION_CHAR_BUDGET = 250

# Overall character cap on the assembled document, aligned with a typical
# 512-token embedding window (~120-170 tokens of English prose).
_DOC_CHAR_BUDGET = 512

_WS_RE = re.compile(r"\s+")


def _fragments(value: object) -> list[str]:
    """Flatten a catalog value into a list of string fragments."""
    if value is None:
        return []
    if isinstance(value, dict):
        return [f"{key}: {item}" for key, item in value.items() if item not in (None, "", [])]
    if isinstance(value, list):
        return [str(item) for item in value if item not in (None, "")]
    text = str(value).strip()
    return [text] if text else []


def _clean(text: str) -> str:
    return _WS_RE.sub(" ", text).strip()


def product_embed_text(product: dict) -> str:
    """Build the curated *semantic* embedding document for a catalog product.

    Composition (order matters -- models weight earlier tokens more):
        1. title
        2. categories (taxonomy path)
        3. a leading slice of the description

    Products without a description (~48% of the catalog) fall back to
    ``title + categories``; no attribute/feature back-filling is done -- that
    signal lives in BM25.
    """
    parts: list[str] = []

    title = _clean(str(product.get("title") or ""))
    if title:
        parts.append(title)

    categories = _fragments(product.get("categories"))
    if categories:
        parts.append(" ".join(_clean(c) for c in categories if _clean(c)))

    description_fragments = _fragments(product.get("description"))
    if description_fragments:
        description = _clean(" ".join(description_fragments))
        if description:
            parts.append(description[:_DESCRIPTION_CHAR_BUDGET])

    doc = _clean(" ".join(parts))
    return doc[:_DOC_CHAR_BUDGET]
