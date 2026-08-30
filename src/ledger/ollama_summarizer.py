"""Local LLM summarizer using Ollama (runs on your laptop)."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class OllamaSummarizer:
    """Summarizes conversation using local Ollama models."""

    def __init__(self, model: str = "phi3:mini") -> None:
        """Initialize Ollama summarizer.

        Parameters
        ----------
        model
            Ollama model name (default: phi3:mini for lightweight)
            Other options: orca-mini:3b, mistral:7b, neural-chat:7b
        """
        self._model = model
        self._client = None

        try:
            import ollama
            self._client = ollama
            logger.debug("OllamaSummarizer initialized with model: %s", model)
        except ImportError as e:
            logger.error("ollama package required; install with: pip install ollama")
            raise

    def summarize(
        self,
        history: list[dict[str, str]],
        constraints: dict[str, list[str]],
        intent: str | None,
        session_summary: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Summarize conversation history into a structured summary."""
        if not history:
            return {
                "summary": "",
                "remembered_preferences": {},
                "topics_covered": [],
                "last_updated_turn": 0,
            }

        # Extract current preferences
        prefs = {}
        for key, values in constraints.items():
            if values:
                prefs[key] = values[0]

        # Generate new summary
        summary = self._generate_summary(history, prefs)

        return {
            "summary": summary,
            "remembered_preferences": prefs,
            "topics_covered": [],
            "last_updated_turn": len(history),
        }

    def _generate_summary(self, history: list[dict[str, str]], prefs: dict[str, str]) -> str:
        """Generate a summary using local Ollama model."""
        if not history or not self._client:
            return self._polish_message(history[-1].get("content", "") if history else "")

        # Build conversation transcript
        transcript = "\n".join([
            f"{msg.get('role', 'user').upper()}: {msg.get('content', '')}"
            for msg in history
        ])

        # Build preference context
        prefs_text = ", ".join([f"{k}: {v}" for k, v in prefs.items() if v])

        prompt = f"""You are a shopping assistant. Write a summary of what the customer currently wants.

CONVERSATION HISTORY:
{transcript}

CURRENT PREFERENCES: {prefs_text if prefs_text else '(none)'}

Write ONE short sentence showing the CURRENT/FINAL state only:
- What product they want
- Key attributes (color, flavor, size, etc) if mentioned

Do NOT include history or evolution—just the final state.

Examples:
- "Customer wants BBQ-flavored chicken drumlets"
- "Customer wants blue boots"
- "Customer wants an iPhone from Apple"

Write ONLY the summary sentence based on what was actually said:"""

        try:
            response = self._client.generate(
                model=self._model,
                prompt=prompt,
                stream=False,
                options={
                    "temperature": 0.1,
                    "num_predict": 100,
                    "top_k": 20,
                    "top_p": 0.9,
                }
            )
            summary = (response.get("response", "") or "").strip()

            if summary and len(summary.split()) >= 3 and summary[0].isalpha():
                # Ensure period at end
                if not summary.endswith("."):
                    summary = summary + "."

                print(f"  ✓ Ollama ({self._model}) generated summary")
                return summary

            # Fallback: polish the last message
            print(f"  ⚠ Ollama response invalid, using fallback")
            return self._polish_message(history[-1].get("content", ""))

        except Exception as exc:
            error_msg = str(exc)
            if "connection" in error_msg.lower():
                print(f"  ✗ Ollama not running (start with: ollama serve)")
            else:
                print(f"  ✗ Ollama error: {type(exc).__name__}")
            logger.debug("Ollama summary generation failed: %s", exc)
            # Graceful fallback: polish the last message
            return self._polish_message(history[-1].get("content", ""))

    def _polish_message(self, message: str) -> str:
        """Polish message: fix typos, capitalize, clean whitespace."""
        if not message:
            return ""

        # Remove extra whitespace
        message = " ".join(message.split())

        # Capitalize first letter
        if message:
            message = message[0].upper() + message[1:]

        return message