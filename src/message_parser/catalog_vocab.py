"""Builds category/brand vocab from the real catalog for higher-precision
attribute matching than static word lists alone."""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

from src.catalog.loader import load_catalog_rows

from .vocab import EXCLUDED_CATEGORY_TERMS

# Catalog terms must be normalized the same way as query-side tokenization
# (split on non-alphanumeric, rejoin with a space) or a literal hyphen
# ("t-shirts") never matches space-joined query tokens ("t shirts").
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")

# Tokenizing with a positive match is cheaper than substituting separators
# and splitting (which copies every product's searchable text to discard it).
_TOKEN_RE = re.compile(r"[a-z0-9]+")

_SEARCH_FIELDS = ("title", "features", "description", "details", "categories", "store")

# A single-word store name is only trusted as a brand signal if it's mostly
# used AS that store, not scattered across products as ordinary text. E.g.
# "skechers" (ratio ~0.97) is kept; "machine" (ratio ~0.0001, mostly from
# "Machine Wash") is dropped.
BRAND_DISTINCTIVENESS_THRESHOLD = 0.3


def _normalize(text: str) -> str:
    return _NON_ALNUM_RE.sub(" ", text.lower()).strip()


def _searchable_words(product: dict) -> set[str]:
    return set(_TOKEN_RE.findall(_searchable_text(product)))


def _searchable_text(product: dict) -> str:
    """Lowercased concatenation of every searchable field of one product."""
    parts: list[str] = []
    for field in _SEARCH_FIELDS:
        value = product.get(field)
        if isinstance(value, dict):
            parts.extend(f"{key} {item}" for key, item in value.items())
        elif isinstance(value, list):
            parts.extend(str(item) for item in value)
        elif value is not None:
            parts.append(str(value))
    return " ".join(parts).lower()


def load_catalog_vocab(
    catalog_path: str | Path,
    brand_distinctiveness_threshold: float = BRAND_DISTINCTIVENESS_THRESHOLD,
) -> tuple[set[str], set[str]]:
    """Returns (categories, brands), both lowercase and space-normalized (see
    `_normalize`). Categories are individual leaf terms from the catalog's
    `categories` lists.

    Brands start from `store` values, then single-word store names are
    filtered by distinctiveness (see BRAND_DISTINCTIVENESS_THRESHOLD).
    Multi-word store names are kept as-is; a few residual collisions with
    ordinary phrases are a known limitation."""
    rows = load_catalog_rows(str(catalog_path))

    categories: set[str] = set()
    store_counts: Counter[str] = Counter()

    for product in rows:
        for cat in product.get("categories") or []:
            for part in str(cat).split(","):
                cleaned = _normalize(part)
                if cleaned and cleaned not in EXCLUDED_CATEGORY_TERMS:
                    categories.add(cleaned)

        store = product.get("store")
        if store:
            normalized_store = _normalize(str(store))
            if normalized_store:
                store_counts[normalized_store] += 1

    # Document frequency is consulted below for single-word store names only,
    # so only those words are worth counting. Tallying every word in the
    # catalog builds a 50k-key Counter whose vast majority is never read.
    candidates = {name for name in store_counts if " " not in name}
    word_doc_freq: Counter[str] = Counter()
    if candidates:
        findall = _TOKEN_RE.findall
        for product in rows:
            word_doc_freq.update(candidates.intersection(findall(_searchable_text(product))))

    brands: set[str] = set()
    for name, count in store_counts.items():
        if " " in name:
            brands.add(name)
            continue
        doc_freq = word_doc_freq.get(name, count)
        ratio = count / doc_freq if doc_freq else 1.0
        if ratio >= brand_distinctiveness_threshold:
            brands.add(name)

    return categories, brands
