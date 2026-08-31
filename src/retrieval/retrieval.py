"""BM25 retrieval over the product catalog.

Backed by sqlite3 FTS5 (same engine used by the weak baseline in
``src/agent.py``), so it adds no new dependencies and scales to the full
50k-row catalog. A single ``Retriever`` instance builds one in-memory index
and is safe to reuse across sessions (reads only).

It does **not** build its own index -- it borrows the shared in-memory FTS5
DB owned by :class:`src.catalog.Catalog`, which is built once at startup.

The public entrypoint is :meth:`Retriever.retrieve_bm25`, which consumes the
same ``dict[str, list]`` "search key" shape the ledger stores, e.g.::

    {"type": ["jacket"], "price": [{"lte": 30.0}]}

Field values are auto-classified by shape:

* list of **strings**  -> soft, weighted BM25 *text* terms (e.g. ``type``,
  ``color``, ``material``, ``brand``, ``style``, ``keywords``).
* list of **dicts** with ``gte``/``lte``/``gt``/``lt``/``eq`` keys -> a
  *numeric range filter* (e.g. ``price``, ``average_rating``).
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING

from ..catalog import Catalog
from ..catalog.catalog import TABLE_NAME, TEXT_COLUMNS as _TEXT_COLUMNS

if TYPE_CHECKING:
    from ..embeddings import EmbeddingClient, VectorIndex

logger = logging.getLogger(__name__)


TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "i", "in", "is", "it", "me", "my", "of", "on", "or", "please", "some",
    "that", "the", "this", "to", "want", "with", "would", "you", "looking",
}

# Default per-column BM25 weights, mirroring src/agent.py's bm25() call
# (parent_asin column is weight 0.0 -- never contributes to the score).
_DEFAULT_WEIGHTS: dict[str, float] = {
    "title": 6.0,
    "categories": 4.0,
    "features": 2.5,
    "details": 2.5,
    "store": 1.5,
    "description": 1.0,
}

# Maps preference tag substrings to per-column weight deltas.
# Tags are free-form, so we match by substring (case-insensitive).
_PREF_WEIGHT_RULES: list[tuple[str, dict[str, float]]] = [
    ("brand",    {"store": 1.5, "title": 0.5}),
    ("store",    {"store": 1.5, "title": 0.5}),
    ("feature",  {"features": 1.5, "details": 1.0}),
    ("detail",   {"features": 1.0, "details": 1.5}),
    ("quality",  {"features": 1.0, "details": 1.0}),
    ("comfort",  {"features": 1.0, "description": 0.5}),
    ("fit",      {"features": 1.0, "description": 0.5}),
    ("style",    {"title": 0.5, "description": 0.5}),
    ("categor",  {"categories": 2.0}),
    ("browse",   {"categories": 1.5, "description": 0.5}),
    ("value",    {"description": 1.0}),
    ("price",    {"description": 0.5}),
]


def _weights_for_preference_tags(
    base: dict[str, float], preference_tags: list[str]
) -> dict[str, float]:
    """Return a copy of ``base`` with column weights nudged by ``preference_tags``."""
    if not preference_tags:
        return base
    deltas: dict[str, float] = {}
    for tag in preference_tags:
        tag_lower = tag.lower()
        for keyword, weight_delta in _PREF_WEIGHT_RULES:
            if keyword in tag_lower:
                for col, delta in weight_delta.items():
                    deltas[col] = deltas.get(col, 0.0) + delta
    if not deltas:
        return base
    return {col: base.get(col, 0.0) + deltas.get(col, 0.0) for col in base}

# Maps a search-key *text field* to the FTS columns it should search. A field
# not listed here falls back to every text column.
_DEFAULT_FIELD_MAP: dict[str, tuple[str, ...]] = {
    "type": ("title", "categories"),
    "category": ("title", "categories"),
    "color": ("title", "features", "description"),
    "material": ("features", "details"),
    "brand": ("store", "title"),
    "store": ("store", "title"),
    "style": ("title", "features", "description"),
    "use_case": ("title", "features", "description"),
    "feature": _TEXT_COLUMNS,
    "keywords": _TEXT_COLUMNS,
}

# Search-key operator -> SQL comparison operator.
_OP_TO_SQL: dict[str, str] = {
    "gte": ">=",
    "lte": "<=",
    "gt": ">",
    "lt": "<",
    "eq": "=",
}

# Search-key field -> numeric catalog column it filters on.
_NUMERIC_FIELD_TO_COLUMN: dict[str, str] = {
    "price": "price",
    "average_rating": "average_rating",
    "rating": "average_rating",
}


def _terms(text: str) -> list[str]:
    """Lowercase content tokens, dropping stopwords and 1-char noise."""
    return [
        token.lower()
        for token in TOKEN_RE.findall(text)
        if len(token) > 1 and token.lower() not in STOPWORDS
    ]


def _is_numeric_filter(value: object) -> bool:
    """A value is a numeric range filter iff it is a list of dicts whose keys
    are all recognized operators (``gte``/``lte``/``gt``/``lt``/``eq``)."""
    if not isinstance(value, list) or not value:
        return False
    for item in value:
        if not isinstance(item, dict) or not item:
            return False
        if any(key not in _OP_TO_SQL for key in item):
            return False
    return True


class Retriever:
    """Stateless (read-only) BM25 retriever over the product catalog."""

    def __init__(
        self,
        catalog: Catalog,
        weights: dict[str, float] | None = None,
        field_map: dict[str, tuple[str, ...]] | None = None,
        vector_index: "VectorIndex | None" = None,
        embedding_client: "EmbeddingClient | None" = None,
    ) -> None:
        self.catalog = catalog
        self.weights = {**_DEFAULT_WEIGHTS, **(weights or {})}
        self.field_map = {**_DEFAULT_FIELD_MAP, **(field_map or {})}
        self.vector_index = vector_index
        self.embedding_client = embedding_client

    @classmethod
    def with_vectors(
        cls,
        catalog: Catalog,
        cache_path: "str | Path" = None,  # type: ignore[assignment]
        weights: dict[str, float] | None = None,
        field_map: dict[str, tuple[str, ...]] | None = None,
    ) -> "Retriever":
        """Build a retriever with semantic search enabled *if possible*.

        Loads the on-disk embedding cache into a :class:`VectorIndex` and
        constructs an :class:`EmbeddingClient` for query embedding. If numpy /
        the cache / the endpoint are unavailable, the vector layer is simply
        left disabled and the retriever behaves exactly like a BM25-only one --
        :meth:`retrieve_vector` then returns ``[]``.
        """
        vector_index = None
        embedding_client = None
        try:
            from ..embeddings import DEFAULT_CACHE_PATH, EmbeddingClient, VectorIndex

            path = cache_path if cache_path is not None else DEFAULT_CACHE_PATH
            vector_index = VectorIndex.load(path)
            if vector_index is None:
                logger.info("No embedding cache at %s; vector retrieval disabled.", path)
            else:
                embedding_client = EmbeddingClient()
                if not embedding_client.available:
                    logger.info(
                        "Embedding client unavailable (%s); vector retrieval disabled.",
                        embedding_client.init_error,
                    )
                    embedding_client = None
                    vector_index = None
        except ImportError as exc:
            logger.info("numpy/embeddings unavailable (%s); vector retrieval disabled.", exc)

        return cls(
            catalog,
            weights=weights,
            field_map=field_map,
            vector_index=vector_index,
            embedding_client=embedding_client,
        )

    @property
    def has_vectors(self) -> bool:
        """True when semantic retrieval is wired up and usable."""
        return self.vector_index is not None and self.embedding_client is not None

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------
    def retrieve_bm25(
        self,
        search_key: dict[str, list],
        top_k: int = 10,
        preference_tags: list[str] | None = None,
    ) -> list[str]:
        """Return up to ``top_k`` ``parent_asin`` strings ranked by weighted
        BM25, filtered by any numeric range constraints in ``search_key``.

        Text fields (list-of-strings) drive the weighted BM25 score. Numeric
        fields (list-of-``{op: value}``) are hard range filters, except that a
        product with a NULL value for that column always passes (missing price
        should not exclude a product)."""
        search_key = search_key or {}

        match_expression = self._build_match_expression(search_key)
        where_clause, where_params = self._build_numeric_filter(search_key)

        weights = _weights_for_preference_tags(self.weights, list(preference_tags or []))
        bm25_args = ", ".join(
            str(weights.get(col, 0.0)) for col in _TEXT_COLUMNS
        )
        # parent_asin (col 0) is weight 0.0; trailing UNINDEXED numeric columns
        # are omitted -- bm25() only weights the text columns.
        rank = f"bm25({TABLE_NAME}, 0.0, {bm25_args})"

        if match_expression:
            sql = (
                f"SELECT parent_asin FROM {TABLE_NAME} WHERE {TABLE_NAME} MATCH ? "
                f"{where_clause} ORDER BY {rank} LIMIT ?"
            )
            params = [match_expression, *where_params, top_k]
        elif where_clause:
            # No text terms: numeric-filter-only query, best-rated first.
            sql = (
                f"SELECT parent_asin FROM {TABLE_NAME} "
                f"{where_clause.replace('AND', 'WHERE', 1)} "
                "ORDER BY average_rating DESC LIMIT ?"
            )
            params = [*where_params, top_k]
        else:
            return []

        rows = self.catalog.execute(sql, params)
        return [str(row[0]) for row in rows]

    def retrieve_vector(self, query_text: str, top_k: int = 10) -> list[str]:
        """Return up to ``top_k`` ``parent_asin`` strings ranked by semantic
        (cosine) similarity between ``query_text`` and each product's embedding.

        This is a standalone semantic path, independent of BM25: it embeds the
        raw query string and searches the in-memory vector index. Returns an
        empty list when the vector layer is unavailable (no cache / no numpy /
        no endpoint) or when embedding the query fails, so callers can fall
        back to :meth:`retrieve_bm25`.
        """
        if not self.has_vectors:
            return []
        query_text = (query_text or "").strip()
        if not query_text:
            return []

        try:
            query_vector = self.embedding_client.embed_one(query_text)
        except Exception as exc:  # noqa: BLE001 - never break retrieval on embed failure
            logger.warning("Query embedding failed; returning no vector results: %s", exc)
            return []

        hits = self.vector_index.search(query_vector, top_k=top_k)
        return [asin for asin, _score in hits]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def title_relevant_ids(
        self, terms: set[str] | list[str], top_k: int = 500
    ) -> set[str]:
        """``parent_asin``s whose ``title``/``categories`` prefix-match any of
        ``terms``.

        Used as a hard relevance gate on the whole-catalog fallback (when
        category bucket resolution fails): unlike ``retrieve_bm25``, whose
        terms are plain quoted phrase matches, prefix matching here means a
        singular query term ("dress") still matches plural/inflected catalog
        text ("dresses") without needing every caller to pre-stem its terms.
        """
        cleaned = [t.strip().lower() for t in terms if t and t.strip()]
        if not cleaned:
            return set()
        clauses = [
            f'{{title categories}}: {re.sub(r"[^a-z0-9]", "", term)}*'
            for term in cleaned
            if re.sub(r"[^a-z0-9]", "", term)
        ]
        if not clauses:
            return set()
        match_expression = " OR ".join(clauses)
        sql = (
            f"SELECT parent_asin FROM {TABLE_NAME} WHERE {TABLE_NAME} MATCH ? "
            f"LIMIT ?"
        )
        rows = self.catalog.execute(sql, [match_expression, top_k])
        return {str(row[0]) for row in rows}

    def _build_match_expression(self, search_key: dict[str, list]) -> str:
        """Build the FTS5 MATCH expression from the text fields, expanding
        each field's terms into its mapped columns via column-filter syntax.

        Each *distinct term* contributes exactly one clause, whose column
        set is the union of every field's mapped columns under which that
        term appears. This matters because SQLite FTS5's ``bm25()`` sums a
        contribution per matching phrase-clause, not per distinct term: if
        the same term were emitted as two separate OR'd clauses (e.g. once
        under a structured field like ``color`` and again under a
        ``keywords`` catch-all), a row matching both would have that term's
        contribution double-counted -- an artificial score boost unrelated
        to actual relevance. Deduping by term keeps each term's bm25
        contribution counted exactly once regardless of how many fields it
        was disclosed under."""
        term_columns: dict[str, set[str]] = {}
        term_order: list[str] = []
        for field, value in search_key.items():
            if field in _NUMERIC_FIELD_TO_COLUMN or _is_numeric_filter(value):
                continue
            if not isinstance(value, list):
                continue
            columns = self.field_map.get(field, _TEXT_COLUMNS)
            for raw in value:
                for term in dict.fromkeys(_terms(str(raw))):
                    if term not in term_columns:
                        term_columns[term] = set()
                        term_order.append(term)
                    term_columns[term].update(columns)

        clauses: list[str] = []
        for term in term_order:
            ordered_columns = [c for c in _TEXT_COLUMNS if c in term_columns[term]]
            column_filter = "{" + " ".join(ordered_columns) + "}"
            clauses.append(f'{column_filter}: "{term}"')
        # OR-combine every term's clause, mirroring the baseline's loose
        # recall-oriented matching.
        return " OR ".join(clauses)

    def _build_numeric_filter(self, search_key: dict[str, list]) -> tuple[str, list]:
        """Build the SQL WHERE fragment (leading ``AND``) and bound params for
        numeric range filters. NULL column values always pass."""
        conditions: list[str] = []
        params: list = []
        for field, value in search_key.items():
            if not _is_numeric_filter(value):
                continue
            column = _NUMERIC_FIELD_TO_COLUMN.get(field)
            if column is None:
                continue
            for item in value:
                for op, bound in item.items():
                    sql_op = _OP_TO_SQL[op]
                    conditions.append(f"({column} IS NULL OR {column} {sql_op} ?)")
                    params.append(bound)
        if not conditions:
            return "", []
        return "AND " + " AND ".join(conditions), params