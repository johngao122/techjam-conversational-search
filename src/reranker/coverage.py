"""Constraint coverage: does a product satisfy a disclosed constraint?

Constraints arrive as strings shaped by the simulator, e.g.:
    "cotton", "color: blue", "size 10", "budget around $19.99",
    "department: womens", "long sleeve", "100% Polyester",
    "High quality mesh for maximum breathability to keep you cool".

Coverage uses *content-token* matching: a constraint is covered when its
meaningful content tokens appear (word-boundary) in the product's lowercased
searchable text. Pure-numeric / percentage noise tokens (e.g. "100", "%") are
ignored so "100% Polyester" matches on "polyester". Budget constraints are
matched numerically against the product price.

For efficiency each constraint is parsed ONCE into a :class:`ConstraintMatcher`
(compiled patterns / budget value), then reused across all candidates in a
``rank()`` call -- avoiding per-candidate regex recompilation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_TOKEN_RE = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class Product:
    """Reranker-local view of a catalog row (mirrors the fields ``rank.py``
    hydrates from :class:`src.catalog.catalog.Catalog`)."""

    parent_asin: str
    text: str          # lowercased searchable text
    price: float | None
    rating_number: int = 0
    average_rating: float = 0.0
    categories: frozenset[str] = frozenset()  # normalized leaf category terms

# Budget: match a number after a budget cue ($, budget, under, less than, <=),
# or a number immediately followed by a currency word ("25 dollars").
_BUDGET_CUE_RE = re.compile(
    r"(?:budget|under|less than|<=|\$)\s*\$?\s*(\d+(?:\.\d+)?)", re.IGNORECASE
)
_BUDGET_TRAILING_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(?:dollars|usd|bucks)\b", re.IGNORECASE
)
_BUDGET_KEYWORDS = ("budget", "under", "less than", "<=", "$", "dollar", "usd", "bucks")

_PRICE_TOLERANCE = 0.25  # +/-25% counts as satisfying a budget constraint

# Leading labels the simulator prepends that carry no matchable value themselves.
_LABEL_PREFIXES = ("color:", "budget", "department:")

# Structural / filler tokens that are not product-content signals on their own.
_NOISE_TOKENS = {
    # generic filler
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "in", "is", "it", "of", "on", "or", "the", "this", "to", "with", "around",
    "prefer", "preference", "please", "quality", "high", "maximum", "keep",
    "you", "your", "cool", "made", "usa",
    # structural attribute labels (handled by dedicated logic, not text match)
    "size", "color", "department", "material", "budget",
}


def _content_tokens(text: str) -> list[str]:
    """Meaningful content tokens: drop noise, pure-numeric, and 1-char tokens."""
    tokens: list[str] = []
    for tok in _TOKEN_RE.findall(text.lower()):
        if len(tok) <= 1:
            continue
        if tok.isdigit():          # pure-numeric noise ("100" in "100% Polyester")
            continue
        if tok in _NOISE_TOKENS:
            continue
        tokens.append(tok)
    return tokens


def _strip_label(constraint: str) -> str:
    lowered = constraint.lower()
    for prefix in _LABEL_PREFIXES:
        if lowered.startswith(prefix):
            return lowered[len(prefix):]
    return lowered


def _budget_target(constraint: str) -> float | None:
    lowered = constraint.lower()
    if not any(kw in lowered for kw in _BUDGET_KEYWORDS):
        return None
    match = _BUDGET_CUE_RE.search(lowered) or _BUDGET_TRAILING_RE.search(lowered)
    return float(match.group(1)) if match else None


@dataclass(frozen=True)
class ConstraintMatcher:
    """A constraint parsed once for reuse across all candidates.

    Exactly one of ``budget`` / ``patterns`` is active:
      - budget is not None  -> numeric price match
      - patterns non-empty  -> all compiled content-token patterns must match
      - both empty          -> constraint carries no signal (never covered)
    """

    budget: float | None
    patterns: tuple[re.Pattern, ...]

    def matches(self, product: Product) -> bool:
        if self.budget is not None:
            if product.price is None:
                return False
            return product.price <= self.budget * (1.0 + _PRICE_TOLERANCE)
        if not self.patterns:
            return False
        return all(p.search(product.text) for p in self.patterns)


def compile_constraint(constraint: str) -> ConstraintMatcher:
    """Parse a constraint string once into a reusable matcher."""
    budget = _budget_target(constraint)
    if budget is not None:
        return ConstraintMatcher(budget=budget, patterns=())
    tokens = _content_tokens(_strip_label(constraint))
    patterns = tuple(re.compile(rf"\b{re.escape(t)}\b") for t in tokens)
    return ConstraintMatcher(budget=None, patterns=patterns)


def compile_constraints(constraints: list[str]) -> list[ConstraintMatcher]:
    return [compile_constraint(c) for c in constraints]


def covers(product: Product, constraint: str) -> bool:
    """Return True if ``product`` satisfies ``constraint`` (convenience API)."""
    return compile_constraint(constraint).matches(product)


def coverage_count(product: Product, constraints: list[str]) -> int:
    """Number of distinct constraints the product satisfies (convenience API)."""
    matchers = compile_constraints(constraints)
    return sum(1 for m in matchers if m.matches(product))
