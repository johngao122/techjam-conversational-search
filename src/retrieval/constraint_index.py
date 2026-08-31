"""Verbatim constraint matching against the catalog.

The customer's disclosed constraints are not paraphrases -- they are literal
slices of the hidden target's own ``features`` and ``details``, normalised the
same way for every product. So the highest-precision signal available is a
straight set-membership test: does this exact string appear in this product's
own attribute set? Within a category bucket that test is close to an identity
check, and it degrades gracefully through substring and token containment when
the wording has drifted.

Weights are flat rather than IDF-scaled. IDF was measured worse here (0.962 vs
0.967) and two other teams reached the same conclusion independently, so
``IDF_WEIGHT=1`` keeps the variant runnable for the record without shipping it.
"""

from __future__ import annotations

import heapq
import math
import os
import re

# Match tiers, best to worst. A verbatim hit is worth 3x a loose token match
# because within a bucket it is very nearly an identity check.
EXACT_WEIGHT = 3.0
SUBSTRING_WEIGHT = 1.0
TOKEN_WEIGHT = 0.5

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_WS_RE = re.compile(r"\s+")
# True exactly when ``_WS_RE.sub(" ", text)`` would change ``text``: either a
# run of two or more whitespace characters, or a single whitespace character
# that is not a plain space (a tab, a newline, or Unicode space such as \xa0).
_WS_COLLAPSE_NEEDED_RE = re.compile(r"\s\s|[^ \S]")

# Structural punctuation is normalised to whitespace on BOTH sides of every
# comparison. `details` flatten to "key: value" in the customer's constraint
# but index as "key value" in the product text; without this every
# details-derived match silently fails.
_PUNCT_RE = re.compile(r"[:%]")

# One pass instead of two. Substituting the punctuation and then collapsing
# whitespace materializes an intermediate copy of every product's full text at
# index-build time; folding both classes into a single character class gives a
# byte-identical result (runs of punctuation and space collapse to one space
# either way) for half the work.
_FOLD_RE = re.compile(r"[:%\s]+")

_LABEL_PREFIX_RE = re.compile(r"^\s*color\s*:\s*", re.IGNORECASE)

# Budget constraints are inert: price is missing for 79% of the catalog and the
# budget candidate is appended past the intent card's 4-constraint slice, so no
# session ever discloses one. Recognised only so it can be dropped.
_BUDGET_RE = re.compile(r"^\s*budget\s+around\s*\$", re.IGNORECASE)


def normalize(text: str) -> str:
    """Fold punctuation and whitespace so both sides compare identically."""
    return _FOLD_RE.sub(" ", str(text).lower()).strip()


def clean_constraint(value: str, limit: int = 180) -> str:
    """Mirror the trimming the simulator applies when it builds a constraint."""
    text = str(value)
    # The collapse is a no-op for 99.9% of catalog attribute strings, and this
    # runs ~470k times at index build. Testing whether the sub would change
    # anything is far cheaper than always allocating a new string.
    if text.isascii():
        needed = "  " in text or any(c in text for c in "\t\n\r\v\f")
    else:
        needed = _WS_COLLAPSE_NEEDED_RE.search(text) is not None
    if needed:
        text = _WS_RE.sub(" ", text)
    return text.strip(" -;,.\t\n")[:limit].rstrip()


def flatten_attributes(product: dict) -> set[str]:
    """The product's own attribute strings, normalised for exact comparison."""
    out: set[str] = set()
    for field in ("features", "details"):
        value = product.get(field)
        if isinstance(value, dict):
            items = [f"{key}: {item}" for key, item in value.items()
                     if item not in (None, "", [])]
        elif isinstance(value, list):
            items = [str(item) for item in value if item not in (None, "")]
        elif value not in (None, ""):
            items = [str(value)]
        else:
            items = []
        for item in items:
            cleaned = normalize(clean_constraint(item))
            if cleaned:
                out.add(cleaned)
    return out


def searchable_text(product: dict) -> str:
    parts: list[str] = []
    for field in ("title", "features", "details", "description", "categories", "store"):
        value = product.get(field)
        if isinstance(value, dict):
            parts.extend(f"{key} {item}" for key, item in value.items())
        elif isinstance(value, list):
            parts.extend(str(item) for item in value)
        elif value is not None:
            parts.append(str(value))
    return normalize(" ".join(parts))


def is_inert(constraint: str) -> bool:
    """True for constraints that carry no usable catalog signal."""
    return bool(_BUDGET_RE.match(constraint))


def prepare(constraint: str) -> tuple[str, tuple[str, ...]]:
    """Normalise a disclosed constraint into (comparable form, content tokens)."""
    stripped = _LABEL_PREFIX_RE.sub("", constraint)
    normalised = normalize(clean_constraint(stripped))
    tokens = tuple(t for t in _TOKEN_RE.findall(normalised) if len(t) > 2)
    return normalised, tokens


class ConstraintIndex:
    """Per-product attribute sets and text, plus the popularity prior."""

    def __init__(self, rows, use_idf: bool | None = None) -> None:
        self.attributes: dict[str, set[str]] = {}
        self.text: dict[str, str] = {}
        self.popularity: dict[str, float] = {}
        self._document_frequency: dict[str, int] = {}
        # Only ``_exact_weight`` reads this, and only under IDF weighting.
        # Building it unconditionally is a 50k-product tally nobody looks at.
        # ``use_idf`` comes from AgentConfig; ``None`` falls back to the env var
        # so bare ``IDF_WEIGHT=1 python …`` still works.
        if use_idf is None:
            use_idf = os.environ.get("IDF_WEIGHT", "") == "1"
        for product in rows:
            asin = str(product["parent_asin"])
            attributes = flatten_attributes(product)
            self.attributes[asin] = attributes
            self.text[asin] = searchable_text(product)
            self.popularity[asin] = math.log1p(float(product.get("rating_number") or 0))
            if use_idf:
                for attribute in attributes:
                    self._document_frequency[attribute] = self._document_frequency.get(attribute, 0) + 1
        self._total = max(1, len(self.attributes))
        self._use_idf = use_idf

    def _exact_weight(self, constraint: str) -> float:
        if not self._use_idf:
            return EXACT_WEIGHT
        df = self._document_frequency.get(constraint, 0)
        return math.log(self._total / max(1, df))

    def score(self, asin: str, constraints: list[tuple[str, tuple[str, ...], float]]) -> float:
        """Weighted match score for one product.

        ``constraints`` are ``(normalised, tokens, weight)`` triples; the weight
        is the supersession factor from the conversation state (0.0 once a
        constraint has been contradicted).
        """
        attributes = self.attributes.get(asin)
        if attributes is None:
            return 0.0
        text = self.text[asin]
        total = 0.0
        for normalised, tokens, weight in constraints:
            if weight <= 0.0 or not normalised:
                continue
            if normalised in attributes:
                total += weight * self._exact_weight(normalised)
            elif normalised in text:
                total += weight * SUBSTRING_WEIGHT
            elif tokens and all(token in text for token in tokens):
                total += weight * TOKEN_WEIGHT
        return total

    def rank(self, pool, constraints, limit: int) -> list[str]:
        """Order a candidate pool by constraint score, then popularity.

        Popularity is a first-class key rather than a tiebreak that never
        fires: public-set targets sit at the 99.4th percentile of review count,
        so within a bucket it carries real signal on its own.
        """
        popularity = self.popularity
        scores = {asin: self.score(asin, constraints) for asin in pool}
        key = lambda a: (-scores[a], -popularity.get(a, 0.0), a)
        if limit:
            # Same key, so ties break identically -- but a partial selection
            # instead of a full sort. Matters most on the unresolved-bucket
            # fallback, where the pool is the whole 50k catalog.
            return heapq.nsmallest(limit, pool, key=key)
        return sorted(pool, key=key)
