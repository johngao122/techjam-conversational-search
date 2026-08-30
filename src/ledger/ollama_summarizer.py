"""Local LLM summarizer using Ollama (runs on your laptop)."""

from __future__ import annotations

import logging

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
        last_user_message: str,
        previous_summary: str = "",
    ) -> str:
        """Summarize what the customer wants based on their latest message and previous context."""
        if not last_user_message or not self._client:
            return self._polish_message(last_user_message)

        return self._generate_summary(last_user_message, previous_summary)

    def _generate_summary(self, last_msg: str, previous_summary: str = "") -> str:
        """Generate a summary using local Ollama model."""
        if not last_msg or not self._client:
            return self._polish_message(last_msg)

        prompt = f"""CUSTOMER'S LATEST MESSAGE: "{last_msg}"
PREVIOUS REQUEST: {previous_summary if previous_summary else "none"}

Write ONE sentence describing what the customer wants RIGHT NOW.

RULES (apply in order):
1. If the latest message contains "nevermind", "nevermine", "instead", "actually", "forget", "change", or similar — the previous request is CANCELLED. Summarize ONLY what is stated in the latest message. Carry over the product type ONLY if no new product is mentioned.
2. If the latest message uses a vague pronoun like "ones", "them", "it", or gives only a color/size/brand with no product — carry over ONLY the product type from the previous request, nothing else.
3. Otherwise — summarize the latest message on its own.

STRICT: Only include attributes explicitly stated in the latest message. Do NOT invent or carry over any detail that was not mentioned in the latest message.

Examples:
- Previous="Customer wants pink shoes.", Latest="nevermine give red" → "Customer wants red shoes."
- Previous="Customer wants pink shoes.", Latest="give me brown ones instead" → "Customer wants brown shoes."
- Previous="Customer wants pink shoes.", Latest="size 10 please" → "Customer wants pink shoes in size 10."

Format: "Customer wants [product] [attributes]."
Output ONLY that one sentence, nothing else:"""

        try:
            response = self._client.generate(
                model=self._model,
                prompt=prompt,
                stream=False,
                options={"temperature": 0.1, "top_k": 20}
            )
            summary = (response.get("response", "") or "").strip()

            if summary and len(summary.split()) >= 3 and summary[0].isalpha():
                if not summary.endswith("."):
                    summary = summary + "."

                print(f"  ✓ Ollama ({self._model}) generated summary")
                return summary

            print(f"  ⚠ Ollama response invalid, using fallback")
            return self._polish_message(last_msg)

        except Exception as exc:
            error_msg = str(exc)
            if "connection" in error_msg.lower():
                print(f"  ✗ Ollama not running (start with: ollama serve)")
            else:
                print(f"  ✗ Ollama error: {type(exc).__name__}")
            logger.debug("Ollama summary generation failed: %s", exc)
            return self._polish_message(last_msg)

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

