"""On-disk cache for product embeddings.

Two files (both gitignored) side by side:

    <path>.npz         numpy archive holding the ``(n, dim)`` float32 matrix
                       under key ``vectors``.
    <path>.meta.json   ordering + provenance: model name, dim, and a per-row
                       list of {parent_asin, text_hash}.

``text_hash`` is a SHA-256 of the curated embed-text, so an incremental build
can re-embed only the products whose document (or the model) changed.

The matrix row order is authoritative and matches ``meta["items"]`` order.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

DEFAULT_CACHE_PATH = Path("data/embeddings.npz")


def text_hash(text: str) -> str:
    """Stable content hash of a curated embed-text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _meta_path(cache_path: str | Path) -> Path:
    cache_path = Path(cache_path)
    return cache_path.with_suffix(cache_path.suffix + ".meta.json")


@dataclass
class EmbeddingCache:
    """Loaded embedding cache: aligned ``asins`` / ``hashes`` / ``vectors``."""

    model: str
    dim: int
    asins: list[str]
    hashes: list[str]
    vectors: "np.ndarray"  # (n, dim) float32

    def hash_by_asin(self) -> dict[str, str]:
        return dict(zip(self.asins, self.hashes))


def save_cache(
    cache_path: str | Path,
    model: str,
    asins: list[str],
    hashes: list[str],
    vectors: "np.ndarray",
) -> None:
    """Persist the matrix (.npz) and metadata (.meta.json)."""
    import numpy as np

    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if not (len(asins) == len(hashes) == vectors.shape[0]):
        raise ValueError(
            f"row mismatch: asins={len(asins)} hashes={len(hashes)} "
            f"vectors={vectors.shape[0]}"
        )

    np.savez(cache_path, vectors=vectors.astype(np.float32, copy=False))
    # np.savez appends .npz if the path lacks it; normalise so the meta sidecar
    # is placed next to the file actually written.
    written = cache_path if cache_path.suffix == ".npz" else cache_path.with_suffix(".npz")

    meta = {
        "model": model,
        "dim": int(vectors.shape[1]) if vectors.size else 0,
        "count": len(asins),
        "items": [{"parent_asin": a, "text_hash": h} for a, h in zip(asins, hashes)],
    }
    _meta_path(written).write_text(json.dumps(meta), encoding="utf-8")


def load_cache(cache_path: str | Path = DEFAULT_CACHE_PATH) -> EmbeddingCache | None:
    """Load the cache, or ``None`` if either file is missing / inconsistent."""
    import numpy as np

    cache_path = Path(cache_path)
    if cache_path.suffix != ".npz":
        cache_path = cache_path.with_suffix(".npz")
    meta_file = _meta_path(cache_path)

    if not cache_path.exists() or not meta_file.exists():
        return None

    try:
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        with np.load(cache_path) as archive:
            vectors = archive["vectors"].astype(np.float32, copy=False)
        items = meta["items"]
        asins = [str(item["parent_asin"]) for item in items]
        hashes = [str(item["text_hash"]) for item in items]
    except (KeyError, ValueError, OSError, json.JSONDecodeError):
        return None

    if len(asins) != vectors.shape[0]:
        return None

    return EmbeddingCache(
        model=str(meta.get("model", "")),
        dim=int(meta.get("dim", vectors.shape[1] if vectors.size else 0)),
        asins=asins,
        hashes=hashes,
        vectors=vectors,
    )
