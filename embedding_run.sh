#!/bin/bash
# Build the product embedding cache (if it does not already exist) using
# EmbeddingGemma over Docker Model Runner, then launch the interactive
# semantic-retrieval REPL. See src/embeddings/README.md for full setup.
#
# Usage:
#     ./embedding_run.sh            # build cache if missing, then run try_it
#     ./embedding_run.sh --force    # rebuild the cache from scratch first
set -e

# --- EmbeddingGemma configuration (override by exporting before you run) ------
export DOCKER_MODEL_BASE_URL="${DOCKER_MODEL_BASE_URL:-http://localhost:12434/engines/v1}"
export DOCKER_MODEL_API_KEY="${DOCKER_MODEL_API_KEY:-none}"
export DOCKER_EMBED_MODEL_NAME="${DOCKER_EMBED_MODEL_NAME:-ai/embeddinggemma}"
export DOCKER_EMBED_DOCUMENT_PREFIX="${DOCKER_EMBED_DOCUMENT_PREFIX:-title: none | text: }"
export DOCKER_EMBED_QUERY_PREFIX="${DOCKER_EMBED_QUERY_PREFIX:-task: search result | query: }"

CACHE_PATH="data/embeddings.npz"

# --force always rebuilds; otherwise only build when the cache is missing.
if [ "${1:-}" = "--force" ]; then
    echo "Rebuilding embedding cache (--force)..."
    python3 -m scripts.build_embeddings --force
elif [ ! -f "$CACHE_PATH" ]; then
    echo "No embedding cache at $CACHE_PATH; building it now..."
    python3 -m scripts.build_embeddings
else
    echo "Embedding cache found at $CACHE_PATH; skipping build."
fi

echo "Launching semantic retrieval REPL..."
python3 -m src.embeddings.try_it
