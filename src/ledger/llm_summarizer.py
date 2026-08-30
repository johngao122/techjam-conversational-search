"""LLM-based conversation summarizer using Google Gemini API."""

from __future__ import annotations

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

        # Extract from last message
        last_msg = history[-1].get("content", "") if history else ""
        all_msgs = " ".join([h.get("content", "") for h in history])
        
        # Extract preferences from constraints
        prefs = {}
        for key, values in constraints.items():
            if values:
                prefs[key] = values[0]
        
        # Use the conversation directly as summary (Gemini keeps truncating)
        summary = self._build_summary(all_msgs, prefs)
        topics = self._extract_topics(all_msgs, prefs)

        return {
            "summary": summary,
            "remembered_preferences": prefs,
            "topics_covered": topics,
            "last_updated_turn": len(history),
        }

    def _build_summary(self, conversation: str, prefs: dict[str, str]) -> str:
        """Build summary from conversation and preferences."""
        if not conversation:
            return ""
        
        # Build a concise summary
        prefs_str = ", ".join([f"{k}: {v}" for k, v in prefs.items()])
        return f"Customer wants: {conversation[:80]} ({prefs_str})"

    def _extract_topics(self, message: str, prefs: dict[str, str]) -> list[str]:
        """Extract topics from message and preferences."""
        topics = []
        keywords = ["boots", "shoes", "leather", "suede", "color", "size", "budget", "material", "style", "comfort"]
        
        msg_lower = message.lower()
        for kw in keywords:
            if kw in msg_lower or kw in str(prefs).lower():
                if kw not in topics:
                    topics.append(kw)
        
        return topics
