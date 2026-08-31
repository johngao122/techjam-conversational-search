"""Category bucketing over the catalog.

The customer's opening line is generated as ``I'm looking for {category}...``
where ``{category}`` is the *coarse category* of the hidden target -- the last
two comma-separated segments of that product's own ``categories`` path. So the
first message names, verbatim, a bucket that is guaranteed to contain the
target. The catalog splits into 1,115 such buckets with a median size of 8
(median ~182 for the buckets targets actually fall in), which turns a 50,000
product ranking problem into a ~180 product one.

``coarse_category`` is reimplemented here rather than imported from
``evaluator/`` -- the shipped agent must not depend on the evaluator package.

``resolve`` never fails: it degrades through exact match, containment, and
token overlap before giving up and returning ``None`` (meaning "search the
whole catalog"). The inexact rungs are paraphrase insurance for the private
set, where the opening template may be reworded.
"""

from __future__ import annotations

import re
from collections import defaultdict

# Top-level category strings that carry no discriminative signal -- every
# product in this catalog is under Clothing, Shoes & Jewelry.
_EXCLUDED = {"clothing", "clothing shoes & jewelry", "clothing, shoes & jewelry"}

_FALLBACK_CATEGORY = "clothing item"

# The opening line, whose tail is the coarse category. The browsing/boundary
# variant ends ", but I'm still exploring."; buying continues ". A key
# requirement is: ..."; intent_override continues ". <preference>".
# Kept deliberately loose on the verb: the private set may reword the wrapper,
# and the category tail is what matters. If no verb matches at all, `parse_category`
# falls back to the whole message so the fuzzy rung still has something to chew on.
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

# Minimum token overlap for the fuzzy rung to fire. Below this the match is
# noise and searching the whole catalog is the safer degradation.
#
# Scored as an overlap coefficient (overlap / min(|fragment|, |key|)), not a
# Jaccard index (overlap / union): a bucket key is a short, structural
# category name (e.g. "clothing dress"), while the message fragment is a
# longer, descriptive phrase that legitimately carries extra adjectives
# ("blue satin dress") the key was never going to contain. Jaccard's union
# term punishes exactly those legitimate extra words, so a real match like
# {blue, satin, dress} vs {clothing, dress} (overlap=1, union=4 -> 0.25)
# would fail a Jaccard threshold that an overlap coefficient (1 / min(3, 2)
# = 0.5) correctly recognises as "the key is contained in the fragment."
_MIN_OVERLAP = 0.34


def _singularize(token: str) -> str:
    """Cheap plural stripping so e.g. "dress" overlaps catalog "dresses".

    Not a real lemmatizer -- just enough to close the gap between how a
    customer names an item ("a dress") and how the catalog's category taxonomy
    names it ("Dresses"), which the exact/containment rungs never see because
    they compare unstemmed strings.
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
    """The single word most likely to name the item's *type*, not its
    attributes -- the last content word of the category fragment (English
    noun phrases put the head noun last: "blue satin dress" -> "dress").

    Used by the title-relevance gate, where OR-ing in every content word
    (colors, materials) would readmit exactly the off-type matches the gate
    exists to block -- e.g. a heel whose title also happens to say "satin".
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

        # Containment has two distinct directions with opposite tie-breaks --
        # collapsing them into one `key in lowered or lowered in key` list and
        # always taking the longest match was a real bug: a short customer
        # fragment like "jackets" is a raw substring of many unrelated, long
        # combined-category bucket keys (e.g. "jackets & vests quilted
        # lightweight jackets", a vest-heavy bucket), and `max(..., key=len)`
        # would pick that coincidentally-long key over the short, exact,
        # category-pure "men jackets" bucket -- recommending vests for a
        # jacket request.
        #
        #   key_in_fragment: the fragment survived a reworded wrapper, or the
        #     organizer trimmed/extended the path (fragment is the LONGER
        #     side, e.g. "a blue men jackets today"). Longest key wins here --
        #     it shares more of the path and so is the more specific bucket.
        #   fragment_in_key: the customer said less than the full templated
        #     category (fragment is the SHORTER side, e.g. just "jackets").
        #     Shortest key wins here -- the closest, least-diluted specific
        #     category, not whichever combined-category name happens to be
        #     longest and merely contains the word somewhere.
        key_in_fragment = [key for key in self._buckets if key and key in lowered]
        if key_in_fragment:
            return max(key_in_fragment, key=len), "containment"

        fragment_in_key = [key for key in self._buckets if key and lowered in key]
        if fragment_in_key:
            return min(fragment_in_key, key=len), "containment-reverse"

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
