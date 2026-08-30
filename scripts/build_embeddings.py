"""One-shot builder for the product embedding cache.

Reads ``data/catalog.jsonl``, builds a curated embed-text per product (see
``src.embeddings.build_doc``), embeds them in batches via the configured
OpenAI-compatible endpoint, and writes the cache (``data/embeddings.npz`` +
``.meta.json``).

Incremental: rows whose curated text hash and model already exist in the cache
are reused, so only new/changed products are re-embedded.

Usage
-----
    export DOCKER_MODEL_BASE_URL="http://localhost:12434/engines/v1"
    export DOCKER_MODEL_API_KEY="none"
    export DOCKER_EMBED_MODEL_NAME="ai/mxbai-embed-large"

    python -m scripts.build_embeddings                 # full 50k catalog
    python -m scripts.build_embeddings --limit 200     # smoke test
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from src.embeddings import (
    DEFAULT_CACHE_PATH,
    EmbeddingClient,
    load_cache,
    product_embed_text,
    save_cache,
    text_hash,
)


def _load_products(catalog_path: Path, limit: int | None) -> list[dict]:
    products: list[dict] = []
    with catalog_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            products.append(json.loads(line))
            if limit is not None and len(products) >= limit:
                break
    return products


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the product embedding cache")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--out", default=str(DEFAULT_CACHE_PATH))
    parser.add_argument("--limit", type=int, default=None, help="Only embed the first N products")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore any existing cache and re-embed everything",
    )
    args = parser.parse_args()

    import numpy as np  # fail fast with a clear message if numpy is missing

    client = EmbeddingClient(batch_size=args.batch_size)
    if not client.available:
        print(f"ERROR: embedding client unavailable: {client.init_error}", file=sys.stderr)
        print(
            "Set DOCKER_MODEL_BASE_URL / DOCKER_MODEL_API_KEY / DOCKER_EMBED_MODEL_NAME "
            "and ensure an embedding model is pulled (e.g. 'ai/mxbai-embed-large').",
            file=sys.stderr,
        )
        return 2

    catalog_path = Path(args.catalog)
    if not catalog_path.exists():
        print(f"ERROR: catalog not found at {catalog_path}", file=sys.stderr)
        return 2

    products = _load_products(catalog_path, args.limit)
    print(f"Loaded {len(products)} products from {catalog_path}")

    # Curated docs + content hashes.
    asins: list[str] = []
    docs: list[str] = []
    hashes: list[str] = []
    # Fold the document prefix into the hash so that changing the prefix (which
    # changes the embedding) correctly invalidates the incremental-reuse cache.
    doc_prefix = client.document_prefix
    for product in products:
        asin = str(product["parent_asin"])
        doc = product_embed_text(product)
        asins.append(asin)
        docs.append(doc)
        hashes.append(text_hash(f"{doc_prefix}\x00{doc}"))

    # Reuse unchanged rows from an existing cache (same model + same text hash).
    reused: dict[str, "np.ndarray"] = {}
    if not args.force:
        existing = load_cache(args.out)
        if existing is not None and existing.model == client.model:
            existing_hash = existing.hash_by_asin()
            existing_vec = {a: existing.vectors[i] for i, a in enumerate(existing.asins)}
            for asin, h in zip(asins, hashes):
                if existing_hash.get(asin) == h and asin in existing_vec:
                    reused[asin] = existing_vec[asin]
            print(f"Reusing {len(reused)} unchanged embeddings from existing cache")

    to_embed_idx = [i for i, a in enumerate(asins) if a not in reused]
    print(f"Embedding {len(to_embed_idx)} products (batch size {args.batch_size})...")

    fresh_vectors: dict[str, "np.ndarray"] = {}
    start_time = time.time()
    embedded = 0
    total = len(to_embed_idx)
    num_batches = (total + args.batch_size - 1) // args.batch_size
    for batch_no, start in enumerate(range(0, total, args.batch_size), start=1):
        batch_idx = to_embed_idx[start : start + args.batch_size]
        batch_docs = [docs[i] for i in batch_idx]
        matrix = client.embed(batch_docs, kind="document")
        for row, i in enumerate(batch_idx):
            fresh_vectors[asins[i]] = matrix[row]
        embedded += len(batch_idx)
        elapsed = time.time() - start_time
        rate = embedded / elapsed if elapsed > 0 else 0.0
        remaining = (total - embedded) / rate if rate > 0 else 0.0
        pct = 100.0 * embedded / total if total else 100.0
        # One newline-terminated line per batch: visible both interactively and
        # when piped/redirected to a log (a '\r' progress bar is invisible in
        # non-TTY contexts).
        print(
            f"  [batch {batch_no}/{num_batches}] {embedded}/{total} "
            f"({pct:.0f}%) | {rate:.1f} docs/s | ETA {remaining/60:.1f} min",
            flush=True,
        )

    # Assemble the final matrix in catalog order.
    all_vectors = []
    for asin in asins:
        vec = reused.get(asin)
        if vec is None:
            vec = fresh_vectors[asin]
        all_vectors.append(vec)
    matrix = np.asarray(all_vectors, dtype=np.float32)

    save_cache(args.out, model=client.model, asins=asins, hashes=hashes, vectors=matrix)
    print(f"Saved {matrix.shape[0]} embeddings (dim={matrix.shape[1]}) to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
