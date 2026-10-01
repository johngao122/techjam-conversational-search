"""Verbatim constraint matching against the catalog.

Disclosed constraints are literal slices of the target's own ``features`` and
``details``, so the highest-precision signal is a set-membership test: does this
exact string appear in the product's attribute set? It degrades through substring
and token containment when wording drifts.

Weights are flat rather than IDF-scaled (IDF measured worse: 0.962 vs 0.967).
``IDF_WEIGHT=1`` keeps the variant runnable for the record without shipping it.
"""

from __future__ import annotations

import hashlib
import heapq
import math
import os
import pickle
import re
from pathlib import Path

# Match tiers, best to worst. A verbatim hit is worth 3x a loose token match
# because within a bucket it is very nearly an identity check.
EXACT_WEIGHT = 3.0
SUBSTRING_WEIGHT = 1.0
TOKEN_WEIGHT = 0.5

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Stemmer for normalizing word forms (dress/dresses/dressed -> dress)
try:
    from nltk.stem import PorterStemmer
    _STEMMER = PorterStemmer()
except ImportError:
    _STEMMER = None


def stem_token(token: str) -> str:
    """Stem a token to its root form (e.g., 'dresses' -> 'dress')."""
    if _STEMMER is None:
        return token
    return _STEMMER.stem(token)
_WS_RE = re.compile(r"\s+")
# True when ``_WS_RE.sub(" ", text)`` would change ``text``: a run of 2+
# whitespace chars, or a single whitespace char that is not a plain space.
_WS_COLLAPSE_NEEDED_RE = re.compile(r"\s\s|[^ \S]")

# Structural punctuation normalised to whitespace on both sides of the compare.
# `details` flatten to "key: value" in constraints but "key value" in the text.
_PUNCT_RE = re.compile(r"[:%]")

# Fold punctuation + whitespace in one pass (byte-identical to two passes).
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
    # Runs ~470k times at index build; the collapse is a no-op for almost all
    # strings, so test-before-substitute is cheaper than always reallocating.
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


_CACHE_DIR = Path(".cache")
_CONSTRAINT_INDEX_CACHE = _CACHE_DIR / "constraint_index.pkl"


def _catalog_hash(rows: tuple) -> str:
    """Compute a hash of the catalog to detect changes."""
    # Hash the first and last few ASINs + total count as a quick fingerprint
    asins = [str(p.get("parent_asin", "")) for p in rows[:10]] + [str(p.get("parent_asin", "")) for p in rows[-10:]]
    fingerprint = f"{len(rows)}:{':'.join(asins)}"
    return hashlib.md5(fingerprint.encode()).hexdigest()[:12]


class ConstraintIndex:
    """Per-product attribute sets and text, plus the popularity prior."""

    def __init__(self, rows, cache: bool = True) -> None:
        self.attributes: dict[str, set[str]] = {}
        self.text: dict[str, str] = {}
        self.popularity: dict[str, float] = {}
        self.average_rating: dict[str, float] = {}
        self._document_frequency: dict[str, int] = {}
        # Inverted index: attribute -> set of ASINs that have this attribute
        # Enables O(1) lookup instead of O(n) scan for constraint matching
        self._inverted: dict[str, set[str]] = {}
        # Token inverted index: token -> set of ASINs whose text contains this token
        self._token_inverted: dict[str, set[str]] = {}
        # Stemmed token inverted index: stem -> set of ASINs
        # Enables matching \"dress\" to products with \"dresses\", \"dressed\", etc.
        self._stem_inverted: dict[str, set[str]] = {}
        self._use_idf = os.environ.get("IDF_WEIGHT", "") == "1"
        self._total = 0

        # Try loading from cache
        if cache and self._load_cache(rows):
            return

        # Build from scratch
        self._build_index(rows)

        # Save to cache
        if cache:
            self._save_cache(rows)

    def _load_cache(self, rows) -> bool:
        """Try to load index from cache. Returns True if successful."""
        if not _CONSTRAINT_INDEX_CACHE.exists():
            return False
        try:
            with open(_CONSTRAINT_INDEX_CACHE, "rb") as f:
                cached = pickle.load(f)
            # Verify catalog hash matches
            if cached.get("catalog_hash") != _catalog_hash(rows):
                return False
            if cached.get("use_idf") != self._use_idf:
                return False
            self.attributes = cached["attributes"]
            self.text = cached["text"]
            self.popularity = cached["popularity"]
            self.average_rating = cached["average_rating"]
            self._document_frequency = cached["_document_frequency"]
            self._inverted = cached["_inverted"]
            self._token_inverted = cached["_token_inverted"]
            self._stem_inverted = cached["_stem_inverted"]
            self._total = cached["_total"]
            return True
        except Exception:
            return False

    def _save_cache(self, rows) -> None:
        """Save index to cache."""
        try:
            _CACHE_DIR.mkdir(exist_ok=True)
            cached = {
                "catalog_hash": _catalog_hash(rows),
                "use_idf": self._use_idf,
                "attributes": self.attributes,
                "text": self.text,
                "popularity": self.popularity,
                "average_rating": self.average_rating,
                "_document_frequency": self._document_frequency,
                "_inverted": self._inverted,
                "_token_inverted": self._token_inverted,
                "_stem_inverted": self._stem_inverted,
                "_total": self._total,
            }
            with open(_CONSTRAINT_INDEX_CACHE, "wb") as f:
                pickle.dump(cached, f)
        except Exception:
            pass  # Caching is best-effort

    def _build_index(self, rows) -> None:
        """Build the index from scratch."""
        use_idf = self._use_idf
        for product in rows:
            asin = str(product["parent_asin"])
            attributes = flatten_attributes(product)
            self.attributes[asin] = attributes
            text = searchable_text(product)
            self.text[asin] = text
            self.popularity[asin] = math.log1p(float(product.get("rating_number") or 0))
            self.average_rating[asin] = float(product.get("average_rating") or 0)
            # Build inverted index for exact attribute matches
            for attr in attributes:
                if attr not in self._inverted:
                    self._inverted[attr] = set()
                self._inverted[attr].add(asin)
            # Build token inverted index for token containment
            for token in _TOKEN_RE.findall(text):
                if len(token) > 2:
                    if token not in self._token_inverted:
                        self._token_inverted[token] = set()
                    self._token_inverted[token].add(asin)
                    # Also index by stem for fuzzy matching
                    stemmed = stem_token(token)
                    if stemmed not in self._stem_inverted:
                        self._stem_inverted[stemmed] = set()
                    self._stem_inverted[stemmed].add(asin)
            if use_idf:
                for attribute in attributes:
                    self._document_frequency[attribute] = self._document_frequency.get(attribute, 0) + 1
        self._total = max(1, len(self.attributes))

    def _exact_weight(self, constraint: str) -> float:
        if not self._use_idf:
            return EXACT_WEIGHT
        df = self._document_frequency.get(constraint, 0)
        return math.log(self._total / max(1, df))

    def fast_candidates(
        self, constraints: list[tuple[str, tuple[str, ...], float]],
        exact_only: bool = False,
    ) -> dict[str, float]:
        """Constraint candidate retrieval using inverted index where possible.

        Returns {asin: score} for all products matching at least one constraint.
        Uses inverted index for exact matches (O(1)), then uses stemmed token
        inverted index for fuzzy matching (dress -> dresses, dressed, etc.).

        Matches the same tiered logic as score():
        - Tier 1: Exact attribute match (weight 3.0) - uses inverted index
        - Tier 2: Substring match (weight 1.0) - uses stem index, then substring check
        - Tier 3: Token containment (weight 0.5) - uses stem index intersection

        If ``exact_only=True``, skips Tier 2 & 3 and returns only exact matches.
        """
        scores: dict[str, float] = {}

        for normalised, tokens, weight in constraints:
            if weight <= 0.0 or not normalised:
                continue

            exact_matched: set[str] = set()

            # Tier 1: Exact attribute match via inverted index (O(1) lookup)
            exact_matches = self._inverted.get(normalised, set())
            for asin in exact_matches:
                scores[asin] = scores.get(asin, 0.0) + weight * self._exact_weight(normalised)
                exact_matched.add(asin)

            if exact_only:
                continue

            # Tier 2 & 3: Use STEMMED token inverted index
            # This handles word variations: dress -> dresses, dressed, dressing
            # O(1) lookup per stem instead of O(vocab) substring scan
            if not tokens:
                continue

            # Get candidates via stem index intersection
            # Stem each query token and intersect their posting lists
            stemmed_tokens = [stem_token(t) for t in tokens]
            candidate_asins: set[str] | None = None
            
            for stem in stemmed_tokens:
                posting = self._stem_inverted.get(stem, set())
                if candidate_asins is None:
                    candidate_asins = posting.copy()
                else:
                    candidate_asins &= posting
                # Early exit if intersection is empty
                if not candidate_asins:
                    break

            if not candidate_asins:
                continue

            # Check candidates using substring matching (preserves original scoring behavior)
            for asin in candidate_asins:
                if asin in exact_matched:
                    continue
                if asin in scores:
                    # Already scored by a previous constraint
                    continue
                text = self.text.get(asin, "")
                # Tier 2: Full normalized string as substring
                if normalised in text:
                    scores[asin] = scores.get(asin, 0.0) + weight * SUBSTRING_WEIGHT
                # Tier 3: All tokens as substrings (e.g., "dress" in "dressed")
                elif all(token in text for token in tokens):
                    scores[asin] = scores.get(asin, 0.0) + weight * TOKEN_WEIGHT

        return scores

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

    def score_pool(
        self,
        pool,
        constraints: list[tuple[str, tuple[str, ...], float]],
    ) -> dict[str, float]:
        """Raw verbatim-constraint score for every asin in ``pool``.

        Exposes the same per-product score :meth:`rank` uses internally, but as
        a ``{asin: score}`` map so a fusion layer can *boost* these base scores
        with complementary retrieval signals (BM25 / vector) rather than
        re-deriving them. Zero-score products are kept (score ``0.0``) so the
        caller sees the full bucket pool, not just the matched subset.
        """
        return {asin: self.score(asin, constraints) for asin in pool}

    def rank(self, pool, constraints, limit: int, rating_style: str | None = None) -> list[str]:
        """Order a candidate pool by constraint score, then rating-style-weighted popularity.

        Popularity is a first-class key rather than a tiebreak that never
        fires: public-set targets sit at the 99.4th percentile of review count,
        so within a bucket it carries real signal on its own.
        """
        popularity = self.popularity
        avg_rating = self.average_rating
        style = (rating_style or "").lower()
        if "critical" in style:
            w_rating, w_volume = 0.5, 1.5
        elif "positive" in style:
            w_rating, w_volume = 1.5, 0.5
        else:
            w_rating, w_volume = 1.0, 1.0
        scores = {asin: self.score(asin, constraints) for asin in pool}
        key = lambda a: (-scores[a], -(w_volume * popularity.get(a, 0.0) + w_rating * avg_rating.get(a, 0.0)), a)
        if limit:
            return heapq.nsmallest(limit, pool, key=key)
        return sorted(pool, key=key)
