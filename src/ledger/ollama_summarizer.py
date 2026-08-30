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

        prompt = f"""Write what the customer currently wants in ONE sentence.

PREVIOUS REQUEST: {previous_summary if previous_summary else "none"}
NEW MESSAGE: "{last_msg}"

Extract ONLY what they want NOW (the latest stated request):
- If message has "nevermind": extract the NEW request, ignore previous
- If message is vague ("show me", "pink ones"): apply to previous product
- Format: "Customer wants [product] [attributes]"

Examples:
- New="shopping carts" → "Customer wants to see shopping carts"
- Previous="carts", New="show me pink ones" → "Customer wants pink shopping carts"

Output ONLY one sentence:"""

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

