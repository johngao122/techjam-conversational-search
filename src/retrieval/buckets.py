"""Category bucketing over the catalog.

The customer's opening line names the *coarse category* of the hidden target
(last two segments of its ``categories`` path), so the first message names a
bucket guaranteed to contain the target -- turning a 50k ranking problem into
a ~180 one. ``resolve`` never fails: it degrades through exact match,
containment, and token overlap before returning ``None`` ("search the whole
catalog"); the inexact rungs are paraphrase insurance for reworded openings.
"""

from __future__ import annotations

import re
from collections import defaultdict

# Top-level category strings that carry no discriminative signal -- every
# product in this catalog is under Clothing, Shoes & Jewelry.
_EXCLUDED = {"clothing", "clothing shoes & jewelry", "clothing, shoes & jewelry"}

_FALLBACK_CATEGORY = "clothing item"

# Opening line whose tail is the coarse category. Kept loose on the verb so
# reworded wrappers still match; parse_category falls back to the whole message
# when no verb matches, so the fuzzy rung still has input.
_OPENING_RE = re.compile(
    r"(?:looking|searching|hunting|shopping)\s+for\s+|"
    r"(?:i\s+(?:want|need)|show\s+me|interested\s+in|after)\s+",
    re.IGNORECASE,
)

_OPENING_TAIL_RE = re.compile(
    r"(?:,\s*but\s+i'?m\s+still\s+exploring|\.\s|\.$)",
    re.IGNORECASE,
)

# Filler that appears around a paraphrased category but never inside a real
# catalog category key.
_FILLER = {"a", "an", "the", "some", "any", "new", "today", "please", "me", "i",
           "for", "of", "in", "and", "to", "my", "looking", "want", "need",
           "hi", "hello", "hey", "there", "recommendations", "recommendation",
           "suggestions", "on", "about", "something", "anything", "im", "am"}

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Minimum token overlap for the fuzzy rung to fire; below this it's noise.
# Scored as an overlap coefficient (overlap / min sizes), not Jaccard, so a
# short structural key contained in a longer descriptive fragment still matches.
_MIN_OVERLAP = 0.34


def _singularize(token: str) -> str:
    """Cheap plural stripping so e.g. "dress" overlaps catalog "dresses".

    Not a real lemmatizer -- just enough to bridge singular customer terms and
    plural catalog taxonomy for the token-overlap rung.
    """
    if token.endswith("sses"):
        return token[:-2]
    if token.endswith("s") and not token.endswith("ss") and len(token) > 3:
        return token[:-1]
    return token


def coarse_category(values: list[str] | None) -> str:
    """The last two meaningful segments of a product's category path."""
    cleaned: list[str] = []
    for value in values or []:
        for part in str(value).split(","):
            part = part.strip()
            if part and part.lower() not in _EXCLUDED:
                cleaned.append(part)
    return " ".join(cleaned[-2:]) if cleaned else _FALLBACK_CATEGORY


def _tokens(text: str, drop_filler: bool = False, singularize: bool = False) -> set[str]:
    found = set(_TOKEN_RE.findall(text.lower()))
    if drop_filler:
        found -= _FILLER
    if singularize:
        found = {_singularize(tok) for tok in found}
    return found


def _ordered_tokens(text: str, drop_filler: bool = False, singularize: bool = False) -> list[str]:
    """Like ``_tokens`` but preserves word order (a set can't)."""
    tokens = _TOKEN_RE.findall(text.lower())
    if drop_filler:
        tokens = [t for t in tokens if t not in _FILLER]
    if singularize:
        tokens = [_singularize(t) for t in tokens]
    return tokens


def fragment_type_tokens(message: str) -> set[str]:
    """The content words an opening message implies about item type.

    Used by the bucket resolver's overlap rung, where matching against a
    whole set of catalog-key tokens makes any of these words useful signal.
    """
    fragment = parse_category(message)
    if not fragment:
        return set()
    return _tokens(fragment.lower(), drop_filler=True, singularize=True)


def head_noun_token(message: str) -> str | None:
    """The last content word of the category fragment -- the head noun that
    names the item's *type* ("blue satin dress" -> "dress").

    Used by the title-relevance gate; OR-ing in colors/materials would readmit
    the off-type matches the gate exists to block.
    """
    fragment = parse_category(message)
    if not fragment:
        return None
    tokens = _ordered_tokens(fragment.lower(), drop_filler=True, singularize=True)
    return tokens[-1] if tokens else None


def parse_category(message: str) -> str:
    """Pull the category fragment out of an opening message.

    Falls back to the whole message when no opening verb is recognised, so a
    fully reworded greeting still reaches the token-overlap rung rather than
    silently degrading to a whole-catalog scan.
    """
    text = message or ""
    match = _OPENING_RE.search(text)
    body = text[match.end():] if match else text
    tail = _OPENING_TAIL_RE.search(body)
    if tail:
        body = body[: tail.start()]
    return re.sub(r"\s+", " ", body).strip(" .,;:")


class BucketIndex:
    """Maps a coarse-category string to the parent_asins that live in it."""

    def __init__(self, rows) -> None:
        self._buckets: dict[str, list[str]] = defaultdict(list)
        for product in rows:
            key = coarse_category(product.get("categories"))
            self._buckets[key.lower()].append(str(product["parent_asin"]))
        self._buckets = dict(self._buckets)
        self._token_sets = {
            key: _tokens(key, singularize=True) for key in self._buckets
        }

    def __len__(self) -> int:
        return len(self._buckets)

    def get(self, key: str) -> list[str]:
        return self._buckets.get(key.lower(), [])

    def resolve(self, message: str) -> tuple[str | None, str]:
        """Resolve an opening message to a bucket key.

        Returns ``(key, how)`` where ``how`` names the rung that fired, for the
        A/B log. ``key`` is ``None`` when nothing matched well enough, which
        callers must read as "fall back to the whole catalog".
        """
        fragment = parse_category(message)
        if not fragment:
            return None, "no-category-parsed"

        lowered = fragment.lower()
        if lowered in self._buckets:
            return lowered, "exact"

        # Containment: reworded wrapper or trimmed/extended path. Longest wins
        # -- a longer key shares more of the path, so it's the more specific bucket.
        contained = [
            key for key in self._buckets
            if key and (key in lowered or lowered in key)
        ]
        if contained:
            return max(contained, key=len), "containment"

        # Token overlap: word order changed or a connector was dropped.
        # Singularized on both sides -- catalog category taxonomy is plural
        # ("Dresses") while a customer names the item singular ("a dress").
        fragment_tokens = _tokens(lowered, drop_filler=True, singularize=True)
        if fragment_tokens:
            best_key, best_score = None, 0.0
            for key, key_tokens in self._token_sets.items():
                if not key_tokens:
                    continue
                overlap = len(fragment_tokens & key_tokens)
                if not overlap:
                    continue
                score = overlap / min(len(fragment_tokens), len(key_tokens))
                if score > best_score:
                    best_key, best_score = key, score
            if best_key is not None and best_score >= _MIN_OVERLAP:
                return best_key, f"overlap:{best_score:.2f}"

        return None, "unresolved"
