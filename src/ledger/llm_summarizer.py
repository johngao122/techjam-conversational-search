"""LLM-based conversation summarizer using DeepSeek API.

Reads configuration from environment variables:
    API_KEY              DeepSeek API key (required)
    DEEPSEEK_BASE_URL    API endpoint, defaults to https://api.deepseek.com
    DEEPSEEK_MODEL       Model name, defaults to deepseek-chat
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


class LLMSummarizer:
    """Summarizes conversation history using DeepSeek API."""

    def __init__(self) -> None:
        api_key = os.environ.get("API_KEY")
        if not api_key:
            logger.warning("API_KEY not set; LLMSummarizer will not be functional")
            self._client = None
            return

        base_url = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
        model = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")

        try:
            from openai import OpenAI  # noqa: PLC0415
        except ImportError as exc:
            logger.error("openai package required for LLMSummarizer: %s", exc)
            self._client = None
            return

        self._client = OpenAI(base_url=base_url, api_key=api_key)
        self._model = model
        logger.debug(
            "LLMSummarizer initialized: model=%s, base_url=%s", model, base_url
        )

    def summarize(
        self,
        history: list[dict[str, str]],
        constraints: dict[str, list[str]],
        intent: str | None,
    ) -> dict[str, Any]:
        """Summarize conversation history into a structured summary.

        Parameters
        ----------
        history
            List of dicts with keys: turn, role, content
            Expected format: [{"turn": 1, "role": "user", "content": "..."}, ...]
        constraints
            Dict of attribute -> list of values, e.g. {"color": ["black"], "size": ["9"]}
        intent
            Current intent: "buying", "browsing", "intent_override", "boundary", or None

        Returns
        -------
        dict
            Keys: summary (str), remembered_preferences (dict), topics_covered (list),
            last_updated_turn (int)
        """
        if not self._client or not history:
            return {
                "summary": "",
                "remembered_preferences": {},
                "topics_covered": [],
                "last_updated_turn": 0,
            }

        history_text = self._format_history(history)
        constraints_text = json.dumps(constraints, indent=2)

        user_prompt = f"""\
Summarize the customer's conversation history, preferences, and product requirements.

## Conversation History
{history_text}

## Extracted Constraints
{constraints_text}

## Current Intent
{intent or "unknown"}

Respond with ONLY a valid JSON object (no markdown, no extra text):
{{
  "summary": "one sentence summarizing their search intent and key requirements",
  "remembered_preferences": {{"attribute": "value", ...}},
  "topics_covered": ["topic1", "topic2", ...]
}}
"""

        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": user_prompt}],
                temperature=0.1,
                max_tokens=256,
            )
            raw_response = response.choices[0].message.content or ""
        except Exception as exc:  # noqa: BLE001
            logger.error("DeepSeek API call failed: %s", exc, exc_info=True)
            return {
                "summary": "",
                "remembered_preferences": {},
                "topics_covered": [],
                "last_updated_turn": len(history),
            }

        return self._parse_response(raw_response, len(history))

    def _format_history(self, history: list[dict[str, str]]) -> str:
        """Format conversation history for the prompt."""
        lines = []
        for entry in history:
            turn = entry.get("turn", "?")
            content = entry.get("content", "")
            lines.append(f"Turn {turn}: {content}")
        return "\n".join(lines) if lines else "(empty)"

    def _parse_response(self, raw: str, last_turn: int) -> dict[str, Any]:
        """Parse LLM response into structured summary."""
        try:
            cleaned = raw.strip()
            if cleaned.startswith("```"):
                lines = cleaned.splitlines()
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

            data = json.loads(cleaned)
            return {
                "summary": str(data.get("summary", "")).strip(),
                "remembered_preferences": data.get("remembered_preferences", {}),
                "topics_covered": data.get("topics_covered", []),
                "last_updated_turn": last_turn,
            }
        except (json.JSONDecodeError, ValueError) as exc:
            logger.warning("Failed to parse LLM response: %s", exc)
            return {
                "summary": "",
                "remembered_preferences": {},
                "topics_covered": [],
                "last_updated_turn": last_turn,
            }
