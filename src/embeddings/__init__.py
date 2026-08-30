"""Semantic embedding + vector retrieval support.

Public API:
    product_embed_text(product)       -> curated string to embed
    EmbeddingClient                   -> text -> vector over DOCKER_* endpoint
    VectorIndex                       -> in-memory numpy cosine search
    load_cache / save_cache           -> on-disk embedding cache (.npz + meta)

numpy and openai are optional dependencies, imported lazily so importing this
package never fails when they are absent.
"""

from .build_doc import product_embed_text
from .client import EmbeddingClient
from .index import VectorIndex
from .store import DEFAULT_CACHE_PATH, EmbeddingCache, load_cache, save_cache, text_hash

__all__ = [
    "product_embed_text",
    "EmbeddingClient",
    "VectorIndex",
    "EmbeddingCache",
    "load_cache",
    "save_cache",
    "text_hash",
    "DEFAULT_CACHE_PATH",
]
