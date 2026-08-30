"""LLM-based conversation summarizer using Google Gemini API."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv
    load_dotenv(Path(".env"))
except ImportError:
    pass

logger = logging.getLogger(__name__)


class LLMSummarizer:
    """Summarizes conversation history using Google Gemini API."""

    def __init__(self) -> None:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            logger.warning("GEMINI_API_KEY not set; LLMSummarizer will not be functional")
            self._client = None
            self._model = None
            return

        model = os.environ.get("GEMINI_MODEL", "gemini-3.6-flash")

        try:
            import google.generativeai as genai  # noqa: PLC0415
        except ImportError as exc:
            logger.error("google-generativeai package required: %s", exc)
            self._client = None
            self._model = None
            return

        try:
            genai.configure(api_key=api_key)
            self._client = genai.GenerativeModel(model)
            self._model = model
            logger.debug("LLMSummarizer initialized: %s", model)
        except Exception as exc:
            logger.error("Failed to initialize Gemini: %s", exc)
            self._client = None
            self._model = None

    def summarize(
        self,
        history: list[dict[str, str]],
        constraints: dict[str, list[str]],
        intent: str | None,
    ) -> dict[str, Any]:
        """Summarize conversation history into a structured summary."""
        if not self._client or not history:
            return {
                "summary": "",
                "remembered_preferences": {},
                "topics_covered": [],
                "last_updated_turn": 0,
            }

        constraints_str = ", ".join([v for vals in constraints.values() for v in vals])
        last_msg = history[-1].get("content", "") if history else ""

        user_prompt = f"""Summarize this shopping request in 2 fields:
SUMMARY: {last_msg}
CONSTRAINTS: {constraints_str}

Output ONLY this format (no other text):
SUMMARY LINE: [one sentence]
PREFS: [comma separated key:value]
TOPICS: [comma separated topics]"""

        try:
            response = self._client.generate_content(
                user_prompt,
                generation_config={"temperature": 0, "max_output_tokens": 150}
            )
            raw_response = response.text or ""
        except Exception as exc:
            logger.error("API failed: %s", exc, exc_info=True)
            return {
                "summary": "",
                "remembered_preferences": {},
                "topics_covered": [],
                "last_updated_turn": len(history),
            }

        return self._parse_response(raw_response, len(history))

    def _parse_response(self, raw: str, last_turn: int) -> dict[str, Any]:
        """Parse text response into structured format."""
        try:
            lines = raw.strip().split("\n")
            summary = ""
            prefs = {}
            topics = []
            
            for line in lines:
                if line.startswith("SUMMARY"):
                    summary = line.split(":", 1)[1].strip() if ":" in line else ""
                elif line.startswith("PREFS"):
                    pref_str = line.split(":", 1)[1].strip() if ":" in line else ""
                    for item in pref_str.split(","):
                        if ":" in item:
                            k, v = item.split(":", 1)
                            prefs[k.strip()] = v.strip()
                elif line.startswith("TOPICS"):
                    topic_str = line.split(":", 1)[1].strip() if ":" in line else ""
                    topics = [t.strip() for t in topic_str.split(",") if t.strip()]
            
            return {
                "summary": summary,
                "remembered_preferences": prefs,
                "topics_covered": topics,
                "last_updated_turn": last_turn,
            }
        except Exception as exc:
            logger.warning("Parse failed: %s", exc)
            return {
                "summary": "",
                "remembered_preferences": {},
                "topics_covered": [],
                "last_updated_turn": last_turn,
            }
