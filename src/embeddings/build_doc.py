"""Curated embedding-document builder.

Turns a raw catalog product into a compact string capturing its **core
semantics** for vector search. This layer is deliberately *complementary to
BM25*: FTS5 BM25 already matches the keyword- and attribute-dense fields that
product owners stuff (``title``, ``features``, ``details`` -- material, size,
fit, brand, closure type, etc.). Duplicating that lexical signal in the
embedding adds nothing. Instead the vector layer should capture the conceptual
"what is this / what is it for" that survives paraphrase and vocabulary
mismatch, which BM25 cannot.

So the embed doc uses only the semantic fields:

    title        -- the product name (concept + type)
    categories   -- taxonomy path (e.g. "Shirts > T-Shirts"), high-semantic
    description  -- a short leading slice of the natural-language description

``store``/brand, ``features`` and ``details`` are intentionally **excluded** --
they are keyword/attribute signal owned by BM25.

Note: we deliberately do NOT distil the description into parsed attributes
(via ``MessageParser``). That was considered and rejected: attribute extraction
reproduces exactly the material/category/brand signal BM25 already handles
(recreating the redundancy we are removing), and the parser -- tuned for short
customer messages -- is noisy on long marketing prose (e.g. it reads
"color: stone", "size: S", "use_case: party" from incidental words). The raw
description slice preserves the paraphrasable semantics that are this layer's
entire value-add.

The output is a plain string; embedding is handled elsewhere.
"""

from __future__ import annotations

import re

# Leading characters taken from the (often long, unbounded) description field.
# Kept tight: the description is natural-language prose whose leading sentences
# carry the product's semantic gist; later sentences are marketing filler. This
# is also the field that drives embedding-model token overflow, so bounding it
# here keeps docs comfortably inside a 512-token window.
_DESCRIPTION_CHAR_BUDGET = 250

# Overall character cap on the assembled document, aligned with a typical
# 512-token embedding window. Description prose is natural English (low token
# density), so 512 chars is ~120-170 tokens -- safely inside the window. The
# EmbeddingClient enforces its own per-input cap + overflow handling as a
# backstop for pathological records.
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
