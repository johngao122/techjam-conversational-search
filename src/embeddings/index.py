"""In-memory vector index with numpy cosine search.

Vectors are L2-normalized once at construction, so cosine similarity reduces to
a single matrix-vector dot product. ``search`` uses ``argpartition`` to pull the
top-k without a full sort of all 50k scores.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from .store import EmbeddingCache, load_cache

if TYPE_CHECKING:
    import numpy as np

_EPS = 1e-12


def _l2_normalize(matrix: "np.ndarray") -> "np.ndarray":
    import numpy as np

    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.maximum(norms, _EPS)
    return (matrix / norms).astype(np.float32, copy=False)


class VectorIndex:
    """Aligned ``asins`` + normalized ``matrix`` supporting cosine top-k search."""

    def __init__(self, asins: list[str], matrix: "np.ndarray") -> None:
        if len(asins) != matrix.shape[0]:
            raise ValueError(
                f"asins ({len(asins)}) and matrix rows ({matrix.shape[0]}) differ"
            )
        self.asins = asins
        self.matrix = _l2_normalize(matrix) if matrix.size else matrix
        self.dim = int(matrix.shape[1]) if matrix.ndim == 2 and matrix.size else 0

    def __len__(self) -> int:
        return len(self.asins)

    @classmethod
    def from_cache(cls, cache: EmbeddingCache) -> "VectorIndex":
        return cls(cache.asins, cache.vectors)

    @classmethod
    def load(cls, cache_path: str | Path) -> "VectorIndex | None":
        """Build an index from a saved cache, or ``None`` if unavailable."""
        cache = load_cache(cache_path)
        if cache is None or cache.vectors.size == 0:
            return None
        return cls.from_cache(cache)

    def search(self, query_vector: "np.ndarray", top_k: int = 10) -> list[tuple[str, float]]:
        """Return up to ``top_k`` ``(parent_asin, cosine_score)`` pairs, best first."""
        import numpy as np

        if len(self.asins) == 0 or query_vector is None or query_vector.size == 0:
            return []

        query = np.asarray(query_vector, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(query))
        if norm < _EPS:
            return []
        query = query / norm

        scores = self.matrix @ query  # (n,) cosine similarities

        k = min(top_k, scores.shape[0])
        if k <= 0:
            return []

        # argpartition -> top-k unordered, then sort just those k descending.
        top_unordered = np.argpartition(scores, -k)[-k:]
        top_sorted = top_unordered[np.argsort(scores[top_unordered])[::-1]]

        return [(self.asins[int(i)], float(scores[int(i)])) for i in top_sorted]
