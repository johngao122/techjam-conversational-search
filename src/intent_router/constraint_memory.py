"""Cumulative verbatim constraint memory with value-conflict supersession.

The evaluator's disclosed constraints are literal slices of the hidden target's
own metadata, wrapped in a small set of fixed reply templates
(``local_evaluator.py``)::

    turn 1 buying:    "I'm looking for {cat}. A key requirement is: {c}."
    turn 1 override:  "I'm looking for {cat}. {old_value}"
    any turn:         "For that, what matters is: {c1}; {c2}."
    override turn:    "Actually, ignore my earlier preference. What I need is: {c}."

The highest-value signal is the *verbatim* constraint string, not its taxonomy
classification. We extract those by splitting on the template markers
(punctuation/whitespace-tolerant) and accumulate them per session.

Supersession is "evict on value conflict": a prior constraint is dropped only
when a new one names the SAME attribute with a DIFFERENT value
("cotton" -> "leather"). Different attributes, or open-ended feature text with
no extractable value, are always kept ("Buckle closure" and "Rubber sole"
coexist).
"""

from __future__ import annotations

import os
import re

# Closed vocabularies the evaluator's own MATERIAL_RE / COLOR_RE use. So two
# open-ended feature strings never conflict.
_MATERIALS = ("cotton", "polyester", "nylon", "leather", "wool", "spandex",
              "silk", "rayon", "fabric")
_COLORS = ("black", "white", "blue", "red", "pink", "green", "brown", "gray",
           "grey", "purple", "yellow", "orange")

_MATERIAL_RE = re.compile(r"\b(" + "|".join(_MATERIALS) + r")\b", re.I)
_COLOR_RE = re.compile(r"\b(" + "|".join(_COLORS) + r")\b", re.I)

# Template markers, matched punctuation- and whitespace-tolerantly. The payload
# is whatever follows the marker; ``;`` splits multiple constraints in one reply.
_KEY_REQ_RE = re.compile(r"a\s+key\s+requirement\s+is\s*:?\s*", re.I)
_MATTERS_RE = re.compile(r"(?:for\s+that\s*,?\s*)?what\s+matters\s+is\s*:?\s*", re.I)
_NEED_RE = re.compile(r"what\s+i\s+need\s+is\s*:?\s*", re.I)
# Aggressive paraphrase: "well X and also Y would be great"
_WELL_RE = re.compile(r"^(?:well|sure|okay|yes|hmm|right)\s+", re.I)
_WOULD_BE_GREAT_RE = re.compile(r"\s+(?:would\s+be\s+great|is\s+important|matters)\s*$", re.I)
# Opening line: "I'm looking for {cat}. {trailing}" -- the trailing clause after
# the category sentence is a free constraint (the override old_value), which the
# evaluator never adds to `disclosed`, so it must be captured here or it is lost.
_OPENING_RE = re.compile(
    r"(?:i'?m\s+|i\s+am\s+)?looking\s+for\s+.*?\.\s*", re.I
)

_WS_RE = re.compile(r"\s+")
_TOKEN_RE = re.compile(r"[a-z0-9]+", re.I)


def _clean(value: str) -> str:
    """Mirror the evaluator's _clean_constraint trimming."""
    return _WS_RE.sub(" ", str(value)).strip(" -;,.\t\n")


def _split_after(text: str, marker: re.Pattern) -> list[str]:
    """Return the ``;``-separated constraints following ``marker``, if present."""
    match = marker.search(text)
    if not match:
        return []
    tail = text[match.end():]
    # Stop at a sentence boundary so we don't swallow a following template.
    tail = re.split(r"(?<=[a-z0-9])\.\s+[A-Z]", tail)[0]
    return [_clean(part) for part in tail.split(";") if _clean(part)]


def extract_constraints(message: str, turn: int) -> list[str]:
    """Pull verbatim constraint strings out of one customer message.

    Handles every evaluator template plus the override turn-1 trailing clause.
    Returns constraints in disclosure order; empty when the message carries
    none (a vague opener, a brush-off, or a generic reprompt).
    """
    text = message or ""
    found: list[str] = []

    found.extend(_split_after(text, _KEY_REQ_RE))
    found.extend(_split_after(text, _MATTERS_RE))
    found.extend(_split_after(text, _NEED_RE))

    # Aggressive paraphrase: "well X and also Y would be great"
    if not found and _WELL_RE.match(text):
        body = _WELL_RE.sub("", text)
        body = _WOULD_BE_GREAT_RE.sub("", body)
        for part in re.split(r"\s+and\s+also\s+", body, flags=re.I):
            cleaned = _clean(part)
            if cleaned:
                found.append(cleaned)

    # Turn-1 opening trailing clause: "I'm looking for {cat}. {old_value}".
    # Only the override scenario puts a real constraint here; buying uses the
    # "A key requirement is:" marker (already captured) and browsing/boundary
    # end in "still exploring" (dropped below).
    if turn == 1 and not _KEY_REQ_RE.search(text):
        match = _OPENING_RE.search(text)
        if match:
            trailing = _clean(text[match.end():])
            low = trailing.lower()
            if trailing and "exploring" not in low and "still" not in low:
                found.append(trailing)

    # De-dupe within the message, preserving order.
    return list(dict.fromkeys(c for c in found if c))


# A disclosed constraint sourced from `details` arrives as "key: value"
# (e.g. "Sole Material: Rubber", "Size: 9", "Closure Type: Lace-up"). The key
# names the attribute and the tail names the value -- no closed vocabulary
# required. This is how conflict detection generalises to characteristics the
# hardcoded material/color lists have never seen.
_KEYED_RE = re.compile(r"^\s*([a-z][a-z0-9 /&-]{1,40}?)\s*:\s*(.+?)\s*$", re.I)

# Words that carry no discriminating value as a key (a bare "color: ..." label
# is handled by the closed-vocab pass; these are noise if treated as attributes).
_KEY_STOP = {"color", "colour"}


def _values(constraint: str) -> frozenset[tuple[str, str]]:
    """(attribute, value) pairs a constraint names, for conflict detection.

    Unions two signals: (1) a vocabulary-free ``key: value`` parse, and
    (2) closed material/color vocab for bare values. Open-ended feature text
    with neither returns empty, so two such strings never conflict.
    """
    pairs: set[tuple[str, str]] = set()

    # (1) generic key: value
    m = _KEYED_RE.match(constraint)
    if m:
        attr = _WS_RE.sub(" ", m.group(1).lower()).strip()
        value = _WS_RE.sub(" ", m.group(2).lower()).strip()
        if attr and value and attr not in _KEY_STOP:
            pairs.add((attr, value))

    # (2) closed-vocab bare values
    for mm in _MATERIAL_RE.finditer(constraint):
        pairs.add(("material", mm.group(1).lower()))
    for mm in _COLOR_RE.finditer(constraint):
        val = mm.group(1).lower()
        pairs.add(("color", "gray" if val == "grey" else val))
    return frozenset(pairs)


def _value_tokens(values: set[str]) -> set[str]:
    """Content tokens of an attribute's value strings (len>2), for overlap.

    Token comparison means "cotton" and "cotton blend" overlap (no conflict),
    while "cotton" and "leather" don't -- without needing either in a list.
    """
    tokens: set[str] = set()
    for value in values:
        # Keep content words (len>2) and any numeric token (sizes/measurements
        # like "9", "11", "10000mm" are the whole value and must compare).
        tokens.update(
            t for t in _TOKEN_RE.findall(value)
            if len(t) > 2 or t.isdigit()
        )
    return tokens


def _conflicts(prior: str, new: str) -> bool:
    """True iff ``new`` supersedes ``prior`` by naming the same attribute with a
    disjoint value (a genuine value-level contradiction).

    Same attribute + overlapping tokens ("cotton" / "cotton blend") is a
    refinement (kept); same attribute + disjoint tokens ("cotton" / "leather")
    is a mind-change (prior evicted). Different attributes never conflict.
    """
    pv, nv = _values(prior), _values(new)
    if not pv or not nv:
        return False
    shared = {a for a, _ in pv} & {a for a, _ in nv}
    if not shared:
        return False
    for attr in shared:
        prior_toks = _value_tokens({v for a, v in pv if a == attr})
        new_toks = _value_tokens({v for a, v in nv if a == attr})
        if prior_toks and new_toks and prior_toks.isdisjoint(new_toks):
            return True
    return False


def _policy() -> str:
    """Supersession policy, selectable via ``OVERRIDE_POLICY``. Ships
    ``evict_on_conflict``."""
    return os.environ.get("OVERRIDE_POLICY", "evict_on_conflict").strip() or "evict_on_conflict"


class ConstraintMemory:
    """Per-session ordered constraint list under evict-on-value-conflict."""

    def __init__(self) -> None:
        self._constraints: list[str] = []
        self._seen: set[str] = set()
        self._policy = _policy()

    def add_message(self, message: str, turn: int) -> list[str]:
        """Extract and integrate a message's constraints. Returns the new ones."""
        added: list[str] = []
        for constraint in extract_constraints(message, turn):
            if self._integrate(constraint):
                added.append(constraint)
        return added

    def _integrate(self, constraint: str) -> bool:
        if constraint in self._seen:
            return False
        if self._policy == "keep":
            survivors = list(self._constraints)
        elif self._policy == "evict_all":
            survivors = []
        else:  # evict_on_conflict (default)
            survivors = [c for c in self._constraints if not _conflicts(c, constraint)]
        if len(survivors) != len(self._constraints):
            self._constraints = survivors
            self._seen = set(self._constraints)
        self._constraints.append(constraint)
        self._seen.add(constraint)
        return True

    @property
    def constraints(self) -> list[str]:
        return list(self._constraints)

    def __len__(self) -> int:
        return len(self._constraints)
