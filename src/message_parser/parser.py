"""Core extraction: raw message -> keywords + structured attributes + signals."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache

from .vocab import (
    BUDGET_RE,
    BUDGET_WORDS_RE,
    COLOR_RE,
    COMPOUND_ALIASES,
    GENERIC_SINGLE_WORD_BLOCKLIST,
    MATERIAL_RE,
    MIN_SINGLE_WORD_BRAND_LEN,
    MIN_SINGLE_WORD_VOCAB_LEN,
    NEGATION_CUES,
    NEGATION_WINDOW,
    NO_PREFERENCE_PATTERNS,
    OVERRIDE_PATTERNS,
    SIZE_BARE_LETTER_RE,
    SIZE_LETTER_RE,
    SIZE_NUMERIC_RE,
    SIZE_UNIT_MARKER_RE,
    SIZE_WIDTH_RE,
    STOPWORDS,
    STYLE_KEYWORDS,
    TOKEN_RE,
    USE_CASE_KEYWORDS,
    VAGUE_PATTERNS,
)


@dataclass
class ParsedMessage:
    raw_text: str
    keywords: list[str] = field(default_factory=list)
    attributes: dict[str, str] = field(default_factory=dict)
    is_override: bool = False
    is_no_preference: bool = False
    is_vague: bool = False
    intent: str | None = None
    category: str | None = None
    product: str | None = None

    def to_dict(self) -> dict:
        return {
            "raw_text": self.raw_text,
            "keywords": self.keywords,
            "attributes": self.attributes,
            "intent": self.intent,
            "category": self.category,
            "product": self.product,
            "signals": {
                "is_override": self.is_override,
                "is_no_preference": self.is_no_preference,
                "is_vague": self.is_vague,
            },
        }


def _clean_terms(text: str) -> list[str]:
    seen: set[str] = set()
    terms: list[str] = []
    for token in TOKEN_RE.findall(text):
        lowered = token.lower()
        if len(lowered) <= 1 or lowered in STOPWORDS or lowered in seen:
            continue
        seen.add(lowered)
        terms.append(lowered)
    return terms


def _matches_any(lowered_text: str, patterns: tuple[str, ...]) -> bool:
    return any(pattern in lowered_text for pattern in patterns)


_CLAUSE_BOUNDARY_RE = re.compile(r"[,.;:!]")


def _is_negated(lowered_text: str, match_start: int) -> bool:
    """True if a negation cue ("don't want", "no", "without", ...) appears
    in a small window immediately before `match_start`, not crossing a
    clause boundary. Position-aware, clause-bounded (not whole-message) so
    "I don't want polyester, I love cotton" negates only "polyester" -- a
    whole-message (or unbounded-window) check would let "don't want" reach
    across the comma and wrongly suppress the unrelated, genuinely positive
    "cotton" in the next clause too."""
    window = lowered_text[max(0, match_start - NEGATION_WINDOW):match_start]
    boundaries = list(_CLAUSE_BOUNDARY_RE.finditer(window))
    if boundaries:
        window = window[boundaries[-1].end():]
    return _NEGATION_RE.search(window) is not None


_NEGATION_RE = re.compile(
    r"\b(" + "|".join(re.escape(cue) for cue in NEGATION_CUES) + r")\b"
)


@lru_cache(maxsize=8)
def _compiled_vocab(vocab: tuple[str, ...]) -> tuple[re.Pattern, tuple[tuple[str, re.Pattern], ...]]:
    """(combined-alternation, per-term patterns) for a keyword vocabulary.

    Compiled once per vocabulary rather than re-escaped and re-hashed into
    ``re``'s internal cache on every call.
    """
    return (
        re.compile(r"\b(" + "|".join(re.escape(t) for t in vocab) + r")\b"),
        tuple((t, re.compile(rf"\b{re.escape(t)}\b")) for t in vocab),
    )


def _all_keyword_hits(lowered_text: str, vocab: tuple[str, ...]) -> tuple[list[str], list[str]]:
    """Returns (all_hits, positive_hits): all matching vocab terms, and the
    subset not preceded by a negation cue. The attribute value is the first
    *positive* hit (e.g. skip a negated "leather" and use the next real
    match), but every hit -- negated or not -- must still be claimed, or an
    unclaimed second match (e.g. "outdoor" claimed as use_case but "work"
    not, from ".. Outdoor & Work Snow & Cold Weather ..") stays free for a
    later, wrong classifier (e.g. brand) to grab."""
    any_re, patterns = _compiled_vocab(vocab)
    # Cheap reject: most messages hit none of the ~100 style / ~40 use-case
    # terms, and one alternation answers that in a single scan. The per-term
    # loop below still runs in vocab order when it does hit, because callers
    # depend on the first *positive* hit being the first in vocab order.
    if not any_re.search(lowered_text):
        return [], []
    all_hits: list[str] = []
    positive_hits: list[str] = []
    for term, pattern in patterns:
        match = pattern.search(lowered_text)
        if not match:
            continue
        all_hits.append(term)
        if not _is_negated(lowered_text, match.start()):
            positive_hits.append(term)
    return all_hits, positive_hits


def _match_compound_alias(
    tokens: list[str],
    vocab: set[str],
    claimed: set[str] | None = None,
) -> tuple[str, list[str]] | None:
    """Checks each raw token's compound-alias phrase (e.g. "crossbody" ->
    "cross body") against `vocab` directly. Deliberately does NOT inject the
    split fragments ("cross", "body") into the general token stream for
    n-gram matching — treating them as independent words would let an
    unrelated single-word match (e.g. a store literally named "Cross")
    fire on a fragment that only exists because of the alias split."""
    claimed = claimed or set()
    for token in tokens:
        if token in claimed:
            continue
        alias = COMPOUND_ALIASES.get(token)
        if not alias:
            continue
        if alias in vocab:
            return alias, [token]
        plural = alias if alias.endswith("s") else alias + "s"
        if plural in vocab:
            return plural, [token]
    return None


def _match_vocab_ngrams(
    tokens: list[str],
    vocab: set[str],
    claimed: set[str] | None = None,
    max_n: int = 4,
    min_single_word_len: int = MIN_SINGLE_WORD_VOCAB_LEN,
) -> tuple[str, list[str]] | None:
    """Longest vocab phrase present in `tokens`. Returns (matched_vocab_term,
    original_tokens_matched) — callers must add the *original* tokens (not
    the matched/pluralized term) to `claimed`, or a singular customer word
    ("jacket") and its plural catalog match ("jackets") won't be recognized
    as the same token and a later matcher (e.g. category, after brand) could
    double-assign it.

    `claimed` tokens (already consumed by a higher-priority attribute) are
    skipped so an ambiguous term like "cotton" (a real material AND a real
    catalog category) isn't double-assigned. Single-word matches under
    `min_single_word_len` or in the generic blocklist are skipped (some real
    store names are plain English words, e.g. "Key", "Not"). Windows whose
    first/last token is a stopword are skipped too (real vocab terms don't
    start/end mid-phrase like "the wave" from marketing copy, e.g. "... blue
    and orange colors of the wave strand bracelet ...")."""
    claimed = claimed or set()
    n = len(tokens)
    for size in range(min(max_n, n), 0, -1):
        for i in range(n - size + 1):
            window = tokens[i : i + size]
            if any(t in claimed for t in window):
                continue
            if window[0] in STOPWORDS or window[-1] in STOPWORDS:
                continue
            phrase = " ".join(window)
            if size == 1 and (len(phrase) < min_single_word_len or phrase in GENERIC_SINGLE_WORD_BLOCKLIST):
                continue
            if phrase in vocab:
                return phrase, window
            # Plural-insensitive fallback: catalog terms are frequently
            # plural ("Shirts", "Jackets") while a customer says singular.
            plural_variant = phrase if phrase.endswith("s") else phrase + "s"
            singular_variant = phrase[:-1] if phrase.endswith("s") and len(phrase) > 3 else None
            if plural_variant in vocab:
                return plural_variant, window
            if singular_variant and singular_variant in vocab:
                return singular_variant, window
    return None


class MessageParser:
    """Stateless — one instance can be reused across all sessions."""

    def __init__(
        self,
        known_categories: set[str] | None = None,
        known_brands: set[str] | None = None,
    ) -> None:
        self.known_categories = known_categories or set()
        self.known_brands = known_brands or set()

    def parse(self, text: str) -> ParsedMessage:
        text = text or ""
        lowered = text.lower()
        result = ParsedMessage(raw_text=text)

        result.is_override = _matches_any(lowered, OVERRIDE_PATTERNS)
        result.is_no_preference = _matches_any(lowered, NO_PREFERENCE_PATTERNS)
        pattern_vague = _matches_any(lowered, VAGUE_PATTERNS)

        if not result.is_no_preference:
            self._extract_attributes(lowered, text, result)

        result.keywords = _clean_terms(text)

        should_fallback = (
            not result.attributes
            and result.keywords
            and not result.is_no_preference
            and not result.is_override
            and not pattern_vague
        )
        if should_fallback:
            result.attributes["feature"] = " ".join(result.keywords[:8])

        # "Vague" = an explicit uncertainty phrase, OR no structured slot was
        # found (only the loose `feature` catch-all, or nothing at all).
        # Deliberately NOT based on message length/word count: a short reply
        # like "blue" or "Size 10" carries a real, actionable structured
        # attribute and is not vague, even though "I want something
        # comfortable" (longer, but produces only a `feature` catch-all) is.
        # A message with an explicit uncertainty phrase AND a real attribute
        # (e.g. "still exploring, but I like blue") stays vague overall while
        # still keeping the extracted attribute.
        structured_keys = set(result.attributes) - {"feature"}
        result.is_vague = pattern_vague or (
            not structured_keys and not result.is_no_preference and not result.is_override
        )

        if result.is_override:
            result.intent = "intent_override"
        elif result.is_no_preference:
            result.intent = "boundary"
        elif result.is_vague:
            result.intent = "browsing"
        else:
            result.intent = "buying" if structured_keys else "browsing"

        result.category = result.attributes.get("category")
        result.product = result.attributes.get("brand")

        return result

    def _extract_attributes(self, lowered: str, original: str, result: ParsedMessage) -> None:
        claimed: set[str] = set()

        # finditer (not search): a negated first mention ("I don't want
        # polyester, I love cotton") must not block a genuinely positive
        # later mention. Every match found is still claimed regardless of
        # polarity, or the negated word stays free for a later matcher.
        material_value = None
        for match in MATERIAL_RE.finditer(lowered):
            value = match.group(1).lower()
            claimed.update(value.split())
            if material_value is None and not _is_negated(lowered, match.start()):
                material_value = value
        if material_value:
            result.attributes["material"] = material_value

        color_value = None
        for match in COLOR_RE.finditer(lowered):
            value = match.group(1).lower()
            claimed.update(value.split())
            if color_value is None and not _is_negated(lowered, match.start()):
                color_value = value
        if color_value:
            result.attributes["color"] = color_value

        # finditer + skip: raw catalog text labels a physical dimension the
        # same way it labels a real size ("Size: 2.5'' in length" vs
        # "Size 10") -- a unit marker right after the number means it's a
        # dimension, not a garment/shoe size. See SIZE_UNIT_MARKER_RE.
        numeric_size = None
        for match in SIZE_NUMERIC_RE.finditer(lowered):
            tail = lowered[match.end():match.end() + 12]
            if SIZE_UNIT_MARKER_RE.search(tail):
                continue
            numeric_size = match.group(1)
            break

        letter_match = None if numeric_size else SIZE_LETTER_RE.search(lowered)
        if numeric_size or letter_match:
            value = numeric_size if numeric_size else letter_match.group(1)
            result.attributes["size"] = value.upper() if value.isalpha() else value
            claimed.add(value.lower())
        else:
            bare_size = SIZE_BARE_LETTER_RE.search(lowered)
            width = SIZE_WIDTH_RE.search(lowered)
            if bare_size:
                result.attributes["size"] = bare_size.group(1).upper()
                claimed.add(bare_size.group(1).lower())
            elif width:
                result.attributes["size"] = width.group(0)

        budget = BUDGET_RE.search(original) or BUDGET_WORDS_RE.search(original)
        if budget:
            result.attributes["budget"] = budget.group(1)

        style_all, style_positive = _all_keyword_hits(lowered, STYLE_KEYWORDS)
        if style_positive:
            result.attributes["style"] = style_positive[0]
        for hit in style_all:
            claimed.update(hit.split())

        use_case_all, use_case_positive = _all_keyword_hits(lowered, USE_CASE_KEYWORDS)
        if use_case_positive:
            result.attributes["use_case"] = use_case_positive[0]
        for hit in use_case_all:
            claimed.update(hit.split())

        tokens = [t.lower() for t in TOKEN_RE.findall(lowered)]

        # Category before brand: a word coincidentally matching both a real
        # store name and a real category (e.g. "jacket"/"Jackets") almost
        # always reflects category intent, not a store mention. Compound
        # alias ("crossbody" -> "cross body") is tried first since it's a
        # more specific signal than generic n-gram matching.
        if self.known_categories:
            category_match = _match_compound_alias(tokens, self.known_categories, claimed=claimed) \
                or _match_vocab_ngrams(tokens, self.known_categories, claimed=claimed)
            if category_match:
                value, window = category_match
                result.attributes["category"] = value
                claimed.update(window)

        if self.known_brands:
            brand_match = _match_compound_alias(tokens, self.known_brands, claimed=claimed) \
                or _match_vocab_ngrams(
                    tokens, self.known_brands, claimed=claimed,
                    min_single_word_len=MIN_SINGLE_WORD_BRAND_LEN,
                )
            if brand_match:
                value, window = brand_match
                result.attributes["brand"] = value
                claimed.update(window)
