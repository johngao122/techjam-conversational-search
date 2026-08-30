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

        # Use the LLM to intelligently extract relevant preferences
        last_msg = history[-1].get("content", "") if history else ""
        all_msgs = " ".join([h.get("content", "") for h in history])
        
        prefs = self._extract_relevant_prefs_with_llm(last_msg, all_msgs, constraints)
        
        # Build summary from last message
        summary = self._build_summary(last_msg, prefs)
        topics = self._extract_topics(last_msg, prefs)

        return {
            "summary": summary,
            "remembered_preferences": prefs,
            "topics_covered": topics,
            "last_updated_turn": len(history),
        }

    def _extract_relevant_prefs_with_llm(
        self, 
        last_msg: str, 
        all_msgs: str, 
        constraints: dict[str, list[str]]
    ) -> dict[str, str]:
        """Use LLM to intelligently determine which preferences are still relevant."""
        if not constraints:
            return {}
        
        # Format constraints for the prompt
        constraint_str = ", ".join([f"{k}: {v[0]}" for k, v in constraints.items() if v])
        
        prompt = f"""Given conversation history and current message, which preferences are still relevant?

History: {all_msgs}
Current: {last_msg}
Preferences: {constraint_str}

Keep only relevant preferences. Drop if category changed. Format: key: value (one per line). Say "NONE" if none apply."""

        try:
            response = self._client.generate_content(
                prompt,
                generation_config={"temperature": 0.2, "max_output_tokens": 80}
            )
            response_text = (response.text or "").strip()
            
            if response_text.upper() == "NONE":
                return {"category": constraints.get("category", [""])[0]} if "category" in constraints else {}
            
            # Parse the LLM response
            prefs = {}
            for line in response_text.split("\n"):
                line = line.strip()
                if not line:
                    continue
                if ":" in line:
                    parts = line.split(":", 1)
                    key = parts[0].strip()
                    value = parts[1].strip()
                    if key and value:
                        prefs[key] = value
            
            return prefs if prefs else self._fallback_prefs(constraints)
            
        except Exception as exc:
            logger.error("LLM preference extraction failed: %s", exc)
            return self._fallback_prefs(constraints)

    def _fallback_prefs(self, constraints: dict[str, list[str]]) -> dict[str, str]:
        """Fallback: always keep category, drop others."""
        prefs = {}
        if "category" in constraints and constraints["category"]:
            prefs["category"] = constraints["category"][0]
        return prefs

    def _build_summary(self, last_message: str, prefs: dict[str, str]) -> str:
        """Build summary from the latest message and current preferences."""
        if not last_message:
            return ""
        
        # Build a concise summary from ONLY the last message
        prefs_str = ", ".join([f"{k}: {v}" for k, v in prefs.items() if v])
        if prefs_str:
            return f"Customer wants: {last_message} ({prefs_str})"
        else:
            return f"Customer wants: {last_message}"

    def _extract_topics(self, message: str, prefs: dict[str, str]) -> list[str]:
        """Extract topics from message and preferences."""
        topics = []
        keywords = ["boots", "shoes", "leather", "suede", "color", "size", "budget", "material", "style", "comfort", "umbrellas", "furry"]
        
        msg_lower = message.lower()
        for kw in keywords:
            if kw in msg_lower or kw in str(prefs).lower():
                if kw not in topics:
                    topics.append(kw)
        
        return topics
