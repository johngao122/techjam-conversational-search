"""Interactive REPL to try semantic (vector) retrieval on your own query.

Prerequisites (see src/embeddings/README.md):
    1. pip install numpy openai
    2. An embedding model served over an OpenAI-compatible endpoint, e.g.
       Docker Model Runner with 'ai/mxbai-embed-large' pulled.
    3. Env vars set:
           export DOCKER_MODEL_BASE_URL="http://localhost:12434/engines/v1"
           export DOCKER_MODEL_API_KEY="none"
           export DOCKER_EMBED_MODEL_NAME="ai/mxbai-embed-large"
    4. Build the embedding cache once:
           python3 -m scripts.build_embeddings

Run from the repo root:
    python3 -m src.embeddings.try_it

Type a query, see the top matches, repeat. Ctrl+D or "quit" to exit.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from ..catalog import Catalog
from ..retrieval import Retriever

_CATALOG_PATH = "data/catalog.jsonl"
_DEFAULT_TOP_K = 10


def _load_titles(catalog_path: str) -> dict[str, str]:
    """Map parent_asin -> title, so results are human-readable."""
    titles: dict[str, str] = {}
    path = Path(catalog_path)
    if not path.exists():
        return titles
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            product = json.loads(line)
            titles[str(product["parent_asin"])] = str(product.get("title") or "")
    return titles


def main() -> int:
    if not Path(_CATALOG_PATH).exists():
        print(f"ERROR: {_CATALOG_PATH} not found.", file=sys.stderr)
        print(
            "Download the catalog first: gzip -dk catalog.jsonl.gz && "
            "mv catalog.jsonl data/catalog.jsonl",
            file=sys.stderr,
        )
        return 2

    print("Loading catalog + embedding index...")
    catalog = Catalog(_CATALOG_PATH)
    retriever = Retriever.with_vectors(catalog)

    if not retriever.has_vectors:
        print(
            "\nVector retrieval is NOT available. Falling back to nothing to show.\n"
            "Checklist:\n"
            "  - Did you build the cache?  python3 -m scripts.build_embeddings\n"
            "  - Are DOCKER_MODEL_BASE_URL / DOCKER_MODEL_API_KEY / "
            "DOCKER_EMBED_MODEL_NAME set?\n"
            "  - Is numpy installed?  pip install numpy\n"
            "See src/embeddings/README.md for full setup.",
            file=sys.stderr,
        )
        return 2

    titles = _load_titles(_CATALOG_PATH)
    print(
        f"Ready. Vector index holds {len(retriever.vector_index)} products.\n"
        "Type a query (or 'quit'):\n"
    )

    while True:
        try:
            query = input("> ").strip()
        except EOFError:
            break
        if not query or query.lower() in {"quit", "exit"}:
            break

        asins = retriever.retrieve_vector(query, top_k=_DEFAULT_TOP_K)
        if not asins:
            print("  (no results)\n")
            continue
        for rank, asin in enumerate(asins, start=1):
            title = titles.get(asin, "")
            print(f"  {rank:2d}. {asin}  {title}")
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
