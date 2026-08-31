"""LLM-based message parser: raw customer message -> keywords + structured
attributes + signals, using an OpenAI-compatible API (e.g. a Docker-hosted
local model).

Configuration is entirely via environment variables — no credentials in code:

    DOCKER_MODEL_BASE_URL   Base URL of the OpenAI-compatible endpoint,
                            e.g. http://localhost:8080/v1
    DOCKER_MODEL_API_KEY    API key; use "none" for unauthenticated local models.
    DOCKER_MODEL_NAME       Model identifier, e.g. "llama3" or "mistral".

Usage::

    from src.message_parser.llm_parser import LLMMessageParser
    from src.message_parser.catalog_vocab import load_catalog_vocab

    categories, brands = load_catalog_vocab("data/catalog.jsonl")
    parser = LLMMessageParser(known_categories=categories, known_brands=brands)
    parsed = parser.parse("black leather boots, size 9, under $80")
    parsed.attributes   # {'material': 'leather', 'color': 'black', 'size': '9', 'budget': '80'}
    parsed.keywords     # BM25 query terms
    parsed.is_vague     # False
"""

from __future__ import annotations

import json
import logging
import os
import random
from typing import Any

from .parser import ParsedMessage, _clean_terms
from .vocab import ALLOWED_ATTRIBUTES

logger = logging.getLogger(__name__)

# Maximum number of catalog hint samples injected into the system prompt.
# Keeping this small avoids ballooning the prompt with thousands of terms.
_MAX_CATEGORY_HINTS = 80
_MAX_BRAND_HINTS = 60

_ATTRIBUTE_DESCRIPTIONS = {
    "category": "product type or taxonomy node, e.g. 'running shoes', 'hoodies', 'earrings'",
    "material": "fabric or material composition, e.g. 'leather', 'cotton', 'stainless steel'",
    "color": "explicit color preference, e.g. 'black', 'navy blue', 'rose gold'",
    "size": "garment or shoe size, e.g. '10', 'M', 'XL', 'wide width'",
    "style": "fit, silhouette, or pattern, e.g. 'slim fit', 'floral', 'vintage', 'oversized'",
    "brand": "brand or store name, e.g. 'Nike', 'Levi\\'s'",
    "budget": "price ceiling as a plain number, e.g. '50' (from 'under $50')",
    "feature": "free-text catch-all for product features not covered by other attributes",
    "use_case": "intended activity or occasion, e.g. 'running', 'office', 'beach', 'hiking'",
}

_SYSTEM_PROMPT_TEMPLATE = """\
You are a product-search keyword extractor for a clothing, shoes, and jewelry catalog.

Your task: given a short customer message, extract structured product-search signals.

## Output format
Respond with **only** a valid JSON object — no explanation, no markdown fences, no extra text.
The JSON must have exactly these three keys:

{{
  "keywords": [...],
  "attributes": {{...}},
  "signals": {{
    "is_override": false,
    "is_no_preference": false,
    "is_vague": false
  }}
}}

## Field definitions

### keywords
A list of lowercase product-relevant terms suitable for BM25 retrieval. Include
the raw useful words from the message (product types, materials, colors, styles,
brands, use-cases). Exclude stopwords (a, the, I, want, looking, please, etc.).

### attributes
Extract **only** the attributes the customer actually specified. Omit any key
where the customer expressed no preference. All values must be lowercase strings.
Allowed keys and their meanings:

{attribute_lines}

### signals
- is_override (bool): true if the customer is changing their mind / overriding a
  previous request (keywords: "actually", "instead", "scratch that", "change of
  mind", "never mind", "on second thought", etc.).
- is_no_preference (bool): true if the customer explicitly has no preference
  ("doesn't matter", "any is fine", "up to you", "no preference", etc.).
- is_vague (bool): true if the message is exploratory or non-specific ("just
  browsing", "still exploring", "not sure yet", "open to anything", etc.), OR if
  no structured attribute could be extracted (only a feature catch-all or nothing).

## Important rules
- Do NOT invent attributes the customer did not mention.
- budget must be a plain number string without currency symbols, e.g. "50".
- size must be the raw value only, e.g. "9", "M", "XL" — no extra words.
- For material/color/style/use_case, use the first positive (non-negated) value
  the customer expressed.
- If the customer says "no leather" or "not blue", do NOT include those values.
{catalog_hint_section}\
"""

_CATALOG_HINT_SECTION_TEMPLATE = """
## Catalog hints
The following are real values from the product catalog — use them to improve
accuracy when the customer's words map to these terms.

Known categories (sample): {categories}

Known brands (sample): {brands}
"""


def _build_system_prompt(
    known_categories: set[str],
    known_brands: set[str],
) -> str:
    attribute_lines = "\n".join(
        f'- "{key}": {desc}' for key, desc in _ATTRIBUTE_DESCRIPTIONS.items()
    )

    catalog_hint_section = ""
    if known_categories or known_brands:
        cat_sample = sorted(random.sample(
            list(known_categories), min(_MAX_CATEGORY_HINTS, len(known_categories))
        )) if known_categories else []
        brand_sample = sorted(random.sample(
            list(known_brands), min(_MAX_BRAND_HINTS, len(known_brands))
        )) if known_brands else []
        catalog_hint_section = _CATALOG_HINT_SECTION_TEMPLATE.format(
            categories=", ".join(cat_sample) if cat_sample else "(none loaded)",
            brands=", ".join(brand_sample) if brand_sample else "(none loaded)",
        )

    return _SYSTEM_PROMPT_TEMPLATE.format(
        attribute_lines=attribute_lines,
        catalog_hint_section=catalog_hint_section,
    )


def _parse_llm_response(raw: str, original_text: str) -> ParsedMessage:
    """Parse the LLM's JSON response into a ParsedMessage.

    Falls back to a vague ParsedMessage with basic keyword extraction if the
    response is malformed or missing required fields.
    """
    result = ParsedMessage(raw_text=original_text)

    try:
        # Strip any accidental markdown fences the model may have emitted.
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            lines = cleaned.splitlines()
            # Drop opening fence (```json or ```) and closing fence
            inner = []
            in_block = False
            for line in lines:
                if line.startswith("```") and not in_block:
                    in_block = True
                    continue
                if line.startswith("```") and in_block:
                    break
                if in_block:
                    inner.append(line)
            cleaned = "\n".join(inner)

        data: dict[str, Any] = json.loads(cleaned)
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning("LLM returned non-JSON response; falling back. Error: %s", exc)
        logger.debug("Raw LLM response: %r", raw)
        result.keywords = _clean_terms(original_text)
        result.is_vague = True
        return result

    # --- keywords ---
    keywords = data.get("keywords")
    if isinstance(keywords, list):
        result.keywords = [str(k).lower() for k in keywords if isinstance(k, str)]
    else:
        result.keywords = _clean_terms(original_text)

    # --- attributes ---
    raw_attrs = data.get("attributes")
    if isinstance(raw_attrs, dict):
        for key, value in raw_attrs.items():
            if key in ALLOWED_ATTRIBUTES and isinstance(value, str) and value.strip():
                result.attributes[key] = value.strip().lower()

    # --- signals ---
    signals = data.get("signals")
    if isinstance(signals, dict):
        result.is_override = bool(signals.get("is_override", False))
        result.is_no_preference = bool(signals.get("is_no_preference", False))
        result.is_vague = bool(signals.get("is_vague", False))

    # Mirror the rule from MessageParser: if no_preference is set, attributes
    # should be empty (the LLM should already do this, but enforce it).
    if result.is_no_preference:
        result.attributes.clear()

    # If the LLM didn't set is_vague but produced no real structured attributes
    # (only a feature catch-all or nothing), mark as vague to match legacy
    # parser semantics.
    if not result.is_vague and not result.is_no_preference and not result.is_override:
        structured_keys = set(result.attributes) - {"feature"}
        if not structured_keys:
            result.is_vague = True

    structured_keys = set(result.attributes) - {"feature"}
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


class LLMMessageParser:
    """Drop-in replacement for MessageParser that uses an LLM for extraction.

    Reads connection details from environment variables:

        DOCKER_MODEL_BASE_URL   e.g. http://localhost:8080/v1
        DOCKER_MODEL_API_KEY    API key; "none" for unauthenticated models
        DOCKER_MODEL_NAME       Model name, e.g. "llama3", "mistral"

    Parameters
    ----------
    known_categories:
        Optional set of category strings from the catalog (see
        ``load_catalog_vocab``). A random sample is injected into the system
        prompt as grounding hints.
    known_brands:
        Optional set of brand strings from the catalog. Same usage as
        ``known_categories``.
    temperature:
        Sampling temperature for the LLM. Defaults to 0.0 for deterministic,
        structured output.
    max_tokens:
        Maximum completion tokens. 512 is enough for the JSON schema.
    """

    def __init__(
        self,
        known_categories: set[str] | None = None,
        known_brands: set[str] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> None:
        base_url = os.environ.get("DOCKER_MODEL_BASE_URL")
        api_key = os.environ.get("DOCKER_MODEL_API_KEY")
        model = os.environ.get("DOCKER_MODEL_NAME")

        missing = [
            name for name, val in [
                ("DOCKER_MODEL_BASE_URL", base_url),
                ("DOCKER_MODEL_API_KEY", api_key),
                ("DOCKER_MODEL_NAME", model),
            ] if not val
        ]
        if missing:
            raise RuntimeError(
                "LLMMessageParser requires the following environment variables to be set: "
                + ", ".join(missing)
                + "\nSee src/message_parser/README.md for setup instructions."
            )

        try:
            from openai import OpenAI  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "The 'openai' package is required for LLMMessageParser. "
                "Install it with: pip install openai"
            ) from exc

        self._client = OpenAI(base_url=base_url, api_key=api_key)
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens

        self.known_categories: set[str] = known_categories or set()
        self.known_brands: set[str] = known_brands or set()

        # Build the system prompt once; the catalog hint sample is randomised
        # here so parses are consistent within the same parser instance.
        self._system_prompt = _build_system_prompt(self.known_categories, self.known_brands)

    def parse(self, text: str) -> ParsedMessage:
        """Extract keywords, attributes, and signals from a customer message.

        Parameters
        ----------
        text:
            Raw customer message string.

        Returns
        -------
        ParsedMessage
            Populated with the same fields as the regex-based ``MessageParser``.
        """
        text = text or ""
        if not text.strip():
            result = ParsedMessage(raw_text=text)
            result.is_vague = True
            return result

        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": self._system_prompt},
                    {"role": "user", "content": text},
                ],
                temperature=self._temperature,
                max_tokens=self._max_tokens,
            )
            raw_content = response.choices[0].message.content or ""
        except Exception as exc:  # noqa: BLE001
            logger.error("LLM API call failed: %s", exc, exc_info=True)
            result = ParsedMessage(raw_text=text)
            result.keywords = _clean_terms(text)
            result.is_vague = True
            return result

        return _parse_llm_response(raw_content, text)
