"""Cosine search restricted to a subset of ASINs.

Wraps :class:`~src.embeddings.index.VectorIndex` without modifying it.
The reverse-lookup dict is built once at construction so mask building is
O(pool_size) rather than O(50k).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from src.embeddings.index import VectorIndex

_EPS = 1e-12


class FilteredVectorIndex:
    """Subset cosine search over an existing VectorIndex."""

    def __init__(self, vector_index: "VectorIndex") -> None:
        self._index = vector_index
        # asin -> row index in self._index.matrix, built once at construction
        self._asin_to_idx: dict[str, int] = {
            a: i for i, a in enumerate(vector_index.asins)
        }

    def search_filtered(
        self,
        query_vector: "np.ndarray",
        allowed_asins: set[str],
        top_k: int = 10,
    ) -> list[tuple[str, float]]:
        """Return up to ``top_k`` ``(parent_asin, cosine_score)`` pairs from
        ``allowed_asins`` only, best first.

        Scores only the rows corresponding to ``allowed_asins`` by slicing the
        matrix before the dot product -- far cheaper than scoring all 50k rows
        and then masking.
        """
        import numpy as np

        if not allowed_asins or query_vector is None:
            return []

        # Normalize query vector
        q = np.asarray(query_vector, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(q))
        if norm < _EPS:
            return []
        q = q / norm

        # Collect valid row indices for allowed pool (O(pool_size))
        valid_indices = np.array(
            [self._asin_to_idx[a] for a in allowed_asins if a in self._asin_to_idx],
            dtype=np.int64,
        )
        if valid_indices.size == 0:
            return []

        # Score only the valid rows: matrix[valid_indices] @ q  shape: (|pool|,)
        valid_scores = self._index.matrix[valid_indices] @ q

        k = min(top_k, int(valid_scores.shape[0]))
        if k <= 0:
            return []

        top_local = np.argpartition(valid_scores, -k)[-k:]
        top_sorted = top_local[np.argsort(valid_scores[top_local])[::-1]]

        return [
            (self._index.asins[int(valid_indices[i])], float(valid_scores[i]))
            for i in top_sorted
        ]
