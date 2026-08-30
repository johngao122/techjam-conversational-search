"""Embedding client over an OpenAI-compatible endpoint.

Reuses the same env-var configuration pattern as
``src/message_parser/llm_parser.py`` (no credentials in code), but points at a
dedicated *embedding* model -- the chat model (e.g. ``ai/llama3.1``) does not
serve OAI-compatible embeddings.

    DOCKER_MODEL_BASE_URL     Base URL, e.g. http://localhost:12434/engines/v1
    DOCKER_MODEL_API_KEY      API key; "none" for unauthenticated local models.
    DOCKER_EMBED_MODEL_NAME   Embedding model id, e.g. "ai/mxbai-embed-large".

Some embedding models require *task-specific prompt prefixes* and are trained
asymmetrically (documents and queries use different prefixes). For example
EmbeddingGemma expects::

    document:  "title: none | text: {content}"
    query:     "task: search result | query: {content}"

These prefixes are configurable (constructor args or env vars) so a new model
can be supported without code changes. They default to empty, preserving the
prefix-free behaviour that models like ``mxbai-embed-large`` are happy with::

    DOCKER_EMBED_DOCUMENT_PREFIX   Prepended to each catalog document.
    DOCKER_EMBED_QUERY_PREFIX      Prepended to each search query.

The client is intentionally lenient: if configuration is missing or the
``openai`` package is unavailable, ``available`` is ``False`` and callers can
fall back to lexical retrieval instead of crashing.
"""

from __future__ import annotations

import logging
import os
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

logger = logging.getLogger(__name__)

_DEFAULT_BATCH_SIZE = 64
_MAX_RETRIES = 3
_RETRY_BACKOFF_SECONDS = 1.5

# Per-input character cap, a cheap pre-filter (NOT the guarantee -- see the
# overflow handling in ``_embed_batch``). Local embedding models have a fixed
# max sequence length (e.g. 512 tokens); a single over-long input makes the
# server reject the whole request. Curated semantic docs are already bounded to
# ~512 chars of natural-language prose (~120-170 tokens), so this cap rarely
# bites; it just backstops a pathological input before it reaches the server.
# Override via ``max_input_chars`` or ``DOCKER_EMBED_MAX_INPUT_CHARS``.
_DEFAULT_MAX_INPUT_CHARS = 512

# When the server rejects a single input for exceeding the token window, shrink
# it by this factor and retry. Halving converges quickly to any target size
# (e.g. 4000 -> 2000 -> 1000 -> ... reaches <=64 chars within ~7 steps), so the
# recovery is robust even for pathologically long inputs.
_OVERFLOW_SHRINK_FACTOR = 0.5
_OVERFLOW_MAX_SHRINKS = 12

# The two embedding "sides". Document and query prefixes may differ (asymmetric
# models like EmbeddingGemma), so callers declare which side they are embedding.
KIND_DOCUMENT = "document"
KIND_QUERY = "query"


class EmbeddingClient:
    """Batched text -> vector embedding over an OpenAI-compatible endpoint.

    Parameters
    ----------
    base_url, api_key, model:
        Override the env-var configuration (mainly for tests). When omitted the
        values are read from ``DOCKER_MODEL_BASE_URL`` / ``DOCKER_MODEL_API_KEY``
        / ``DOCKER_EMBED_MODEL_NAME``.
    batch_size:
        Number of texts sent per API request.
    document_prefix, query_prefix:
        Task prefixes prepended before embedding. When omitted they are read
        from ``DOCKER_EMBED_DOCUMENT_PREFIX`` / ``DOCKER_EMBED_QUERY_PREFIX``
        (default empty). Both the build script (documents) and query retrieval
        must use the same client configuration so the two embedding spaces
        align.
    max_input_chars:
        Hard per-input character cap applied *after* prefixing. Protects against
        a single over-long document exceeding the model's max sequence length
        (which otherwise fails the whole batch request). When omitted, read from
        ``DOCKER_EMBED_MAX_INPUT_CHARS`` (default ``1000``). Set to ``0`` to
        disable truncation.
    """

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        batch_size: int = _DEFAULT_BATCH_SIZE,
        document_prefix: str | None = None,
        query_prefix: str | None = None,
        max_input_chars: int | None = None,
    ) -> None:
        self.base_url = base_url or os.environ.get("DOCKER_MODEL_BASE_URL")
        self.api_key = api_key or os.environ.get("DOCKER_MODEL_API_KEY")
        self.model = model or os.environ.get("DOCKER_EMBED_MODEL_NAME")
        self.batch_size = max(1, batch_size)
        if max_input_chars is None:
            max_input_chars = int(
                os.environ.get("DOCKER_EMBED_MAX_INPUT_CHARS", _DEFAULT_MAX_INPUT_CHARS)
            )
        self.max_input_chars = max(0, max_input_chars)
        self.document_prefix = (
            document_prefix
            if document_prefix is not None
            else os.environ.get("DOCKER_EMBED_DOCUMENT_PREFIX", "")
        )
        self.query_prefix = (
            query_prefix
            if query_prefix is not None
            else os.environ.get("DOCKER_EMBED_QUERY_PREFIX", "")
        )

        self._client = None
        self._available = False
        self._init_error: str | None = None
        # Embedding dimension, learned on the first successful call; used to
        # size a zero-vector placeholder for any un-embeddable input.
        self._last_dim: int | None = None

        self._initialise()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    def _initialise(self) -> None:
        missing = [
            name
            for name, val in (
                ("DOCKER_MODEL_BASE_URL", self.base_url),
                ("DOCKER_MODEL_API_KEY", self.api_key),
                ("DOCKER_EMBED_MODEL_NAME", self.model),
            )
            if not val
        ]
        if missing:
            self._init_error = "missing env vars: " + ", ".join(missing)
            logger.info("EmbeddingClient disabled (%s)", self._init_error)
            return

        try:
            from openai import OpenAI  # noqa: PLC0415
        except ImportError as exc:
            self._init_error = f"openai package not installed: {exc}"
            logger.info("EmbeddingClient disabled (%s)", self._init_error)
            return

        # A per-request timeout so a wedged/overloaded endpoint fails fast with
        # a clear error instead of blocking the build indefinitely.
        self._client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
        )
        self._available = True

    @property
    def available(self) -> bool:
        """True when the client is configured and ready to embed."""
        return self._available

    @property
    def init_error(self) -> str | None:
        """Human-readable reason the client is unavailable, or ``None``."""
        return self._init_error

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------
    def _prefix_for(self, kind: str) -> str:
        if kind == KIND_DOCUMENT:
            return self.document_prefix
        if kind == KIND_QUERY:
            return self.query_prefix
        raise ValueError(f"unknown embedding kind {kind!r}; expected 'document' or 'query'")

    def embed(self, texts: list[str], kind: str = KIND_DOCUMENT) -> "np.ndarray":
        """Embed ``texts`` into a float32 ``(len(texts), dim)`` array.

        ``kind`` selects the task prefix applied to each text: ``"document"``
        (default) uses ``document_prefix``, ``"query"`` uses ``query_prefix``.
        For symmetric / prefix-free models both are empty and ``kind`` is a
        no-op.

        Raises ``RuntimeError`` if the client is unavailable -- callers that
        want graceful degradation should check ``available`` first.
        """
        import numpy as np  # local import: numpy is an optional dependency

        if not self._available:
            raise RuntimeError(
                f"EmbeddingClient is not available ({self._init_error}). "
                "Check DOCKER_MODEL_* env vars and the openai package."
            )
        if not texts:
            return np.empty((0, 0), dtype=np.float32)

        prefix = self._prefix_for(kind)
        prepared = [f"{prefix}{text}" for text in texts] if prefix else list(texts)

        # Cheap pre-filter: cap each input's chars (after prefixing, so the task
        # prefix is preserved). This is a first line of defence only -- the real
        # guarantee against token-window overflow is the server-driven
        # split/truncate recovery in ``_embed_batch``.
        if self.max_input_chars:
            truncated = 0
            for i, text in enumerate(prepared):
                if len(text) > self.max_input_chars:
                    prepared[i] = text[: self.max_input_chars]
                    truncated += 1
            if truncated:
                logger.debug(
                    "Truncated %d/%d inputs to %d chars for embedding.",
                    truncated,
                    len(prepared),
                    self.max_input_chars,
                )

        vectors: list[list[float]] = []
        for start in range(0, len(prepared), self.batch_size):
            batch = prepared[start : start + self.batch_size]
            vectors.extend(self._embed_batch(batch))

        return np.asarray(vectors, dtype=np.float32)

    def embed_one(self, text: str, kind: str = KIND_QUERY) -> "np.ndarray":
        """Embed a single string into a 1-D float32 vector.

        Defaults to ``kind="query"`` since single-string embedding is the query
        path; pass ``kind="document"`` to embed one catalog document.
        """
        matrix = self.embed([text], kind=kind)
        return matrix[0]

    @staticmethod
    def _is_overflow_error(exc: Exception) -> bool:
        """True if ``exc`` is the model rejecting an input for exceeding its
        token window (as opposed to a transient/network error).

        The server reports this as an HTTP 500 whose message mentions the token
        limit, e.g. "input (516 tokens) is too large to process". We match on
        the message text so genuine 500s still surface as real failures.
        """
        message = str(getattr(exc, "message", "") or exc).lower()
        return "too large" in message and "token" in message

    def _embed_raw(self, batch: list[str]) -> list[list[float]]:
        """Single API call, transient-retry only. Order-preserved.

        Raises the underlying exception (without extra retries) when it is a
        token-overflow error, so the caller can recover by splitting/truncating.
        """
        last_error: Exception | None = None
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                response = self._client.embeddings.create(
                    model=self.model,
                    input=batch,
                )
                # Preserve request order (API returns objects with .index).
                ordered = sorted(response.data, key=lambda item: item.index)
                vectors = [list(item.embedding) for item in ordered]
                if vectors:
                    self._last_dim = len(vectors[0])
                return vectors
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                # Overflow is deterministic -- retrying the identical input is
                # pointless. Surface immediately so the caller can shrink it.
                if self._is_overflow_error(exc):
                    raise
                logger.warning(
                    "Embedding request failed (attempt %d/%d): %s",
                    attempt,
                    _MAX_RETRIES,
                    exc,
                )
                if attempt < _MAX_RETRIES:
                    time.sleep(_RETRY_BACKOFF_SECONDS * attempt)
        raise RuntimeError(f"Embedding request failed after {_MAX_RETRIES} attempts") from last_error

    def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        """Embed a batch, recovering from token-overflow rejections.

        On overflow, a multi-item batch is split so the offending item is
        isolated; a single over-long item is progressively truncated until it
        fits. If an item cannot be made to fit, it is embedded as a zero vector
        (logged) rather than aborting the whole build -- a single pathological
        record should not cost a full 50k-product embedding run.
        """
        try:
            return self._embed_raw(batch)
        except Exception as exc:  # noqa: BLE001
            if not self._is_overflow_error(exc):
                raise

        # --- overflow recovery ---
        if len(batch) > 1:
            # Split the batch so the oversized item is isolated; recurse.
            mid = len(batch) // 2
            return self._embed_batch(batch[:mid]) + self._embed_batch(batch[mid:])

        # Single item that overflows -> progressively truncate its text.
        text = batch[0]
        for _ in range(_OVERFLOW_MAX_SHRINKS):
            new_len = int(len(text) * _OVERFLOW_SHRINK_FACTOR)
            if new_len <= 0:
                break
            text = text[:new_len]
            try:
                vectors = self._embed_raw([text])
                logger.warning(
                    "Embedding input exceeded the token window; truncated to %d chars.",
                    len(text),
                )
                return vectors
            except Exception as exc:  # noqa: BLE001
                if not self._is_overflow_error(exc):
                    raise
                continue

        # Could not fit even after maximal truncation: skip with a zero vector.
        logger.error(
            "Dropping an input that could not be embedded within the token "
            "window even after truncation (original %d chars).",
            len(batch[0]),
        )
        return [self._zero_vector()]

    def _zero_vector(self) -> list[float]:
        """A placeholder vector for an un-embeddable input.

        Dimensionality is only known after the first successful call. If nothing
        has embedded yet (the very first item overflows unrecoverably -- almost
        impossible for a <=512-char doc), we cannot size it correctly and raise
        rather than emit a mis-shaped row that would corrupt the matrix.
        """
        if not self._last_dim:
            raise RuntimeError(
                "First embedding input could not be embedded within the token "
                "window and no embedding dimension is known yet."
            )
        return [0.0] * self._last_dim
