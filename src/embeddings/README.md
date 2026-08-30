# Semantic Vector Retrieval

A **standalone semantic retrieval path** alongside the BM25 (SQLite FTS5)
retriever. It embeds a *curated* representation of each product and matches an
input query by cosine similarity, using **EmbeddingGemma** served over an
OpenAI-compatible endpoint (Docker Model Runner).

- `product_embed_text(product)` — builds a compact document capturing the
  product's **core semantics** (title + categories + short description slice).
  Brand, features, and attribute details are deliberately excluded — that
  keyword signal is owned by BM25.
- `EmbeddingClient` — turns text into vectors via the endpoint, configured
  entirely by environment variables.
- `VectorIndex` — in-memory numpy cosine search over the product vectors.
- `Retriever.retrieve_vector(query, top_k)` — returns a ranked list of
  `parent_asin` strings.

Embeddings are computed once by a build script and cached to disk. If any
prerequisite is missing (numpy, the cache, or the endpoint), the vector layer
disables itself and `retrieve_vector` returns `[]` — BM25 is unaffected.

---

## Prerequisites

1. **Python packages** (into your active env):
   ```bash
   pip install numpy openai
   ```
   OR install all deps via `environment.yml`.

2. **Docker Desktop** with Docker Model Runner enabled.

---

## Setup: Docker Model Runner (EmbeddingGemma)

Docker Model Runner is built into Docker Desktop — no separate container needed.
A chat model (e.g. `ai/llama3.1`) **cannot** serve embeddings; you must pull a
dedicated embedding model.

```bash
# 1. Enable the OpenAI-compatible API on localhost:12434
docker desktop enable model-runner --tcp=12434

# 2. Pull EmbeddingGemma
docker model pull ai/embeddinggemma

# 3. Verify it can embed
curl http://localhost:12434/engines/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model":"ai/embeddinggemma","input":"black leather boots"}' | head -c 200
```

---

## Configuration

All connection details come from **environment variables** — never hard-code
credentials.

| Variable | Description | Value |
|---|---|---|
| `DOCKER_MODEL_BASE_URL` | OpenAI-compatible base URL | `http://localhost:12434/engines/v1` |
| `DOCKER_MODEL_API_KEY` | API key (DMR needs none) | `none` |
| `DOCKER_EMBED_MODEL_NAME` | Embedding model id | `ai/embeddinggemma` |
| `DOCKER_EMBED_DOCUMENT_PREFIX` | Task prefix for catalog documents | `title: none \| text: ` |
| `DOCKER_EMBED_QUERY_PREFIX` | Task prefix for search queries | `task: search result \| query: ` |

EmbeddingGemma is trained **asymmetrically** — documents and queries need
different task prefixes, or retrieval quality drops sharply. The document prefix
is folded into the cache's content hash, so changing it triggers a re-embed on
the next build. **Set the same env vars for both the build script and
`try_it.py`** so the two embedding spaces align.

```bash
export DOCKER_MODEL_BASE_URL="http://localhost:12434/engines/v1"
export DOCKER_MODEL_API_KEY="none"
export DOCKER_EMBED_MODEL_NAME="ai/embeddinggemma"
export DOCKER_EMBED_DOCUMENT_PREFIX="title: none | text: "
export DOCKER_EMBED_QUERY_PREFIX="task: search result | query: "
```

> `DOCKER_MODEL_BASE_URL` and `DOCKER_MODEL_API_KEY` are shared with the
> `LLMMessageParser`. The `DOCKER_EMBED_*` vars are specific to embeddings.

---

## Quick start

`embedding_run.sh` (repo root) builds the cache if it is missing, then launches
the interactive REPL:

```bash
./embedding_run.sh              # build cache if needed, then run try_it
./embedding_run.sh --force      # rebuild the cache from scratch first
```

Or run the two steps by hand:

```bash
# Build the cache once → data/embeddings.npz (+ .meta.json). Both gitignored.
python3 -m scripts.build_embeddings                # full 50k catalog
python3 -m scripts.build_embeddings --limit 200    # smoke test

# Try it interactively
python3 -m src.embeddings.try_it
```

The build is **incremental**: on re-run only products whose curated text (or the
model) changed are re-embedded. Useful flags: `--catalog PATH`, `--out PATH`,
`--batch-size N`, `--limit N`, `--force`.

Example session:

```text
Ready. Vector index holds 50000 products.
Type a query (or 'quit'):

> warm waterproof winter jacket for hiking
   1. B09XXXX1  Men's Waterproof Insulated Mountain Parka ...
   2. B07YYYY2  Columbia Powder Lite Hooded Winter Jacket ...

> quit
```

---

## Use it in code

```python
from src.catalog import Catalog
from src.retrieval import Retriever

catalog = Catalog("data/catalog.jsonl")

# Enables semantic search if cache + endpoint + numpy are present; otherwise
# silently degrades to a BM25-only retriever.
retriever = Retriever.with_vectors(catalog)

if retriever.has_vectors:
    asins = retriever.retrieve_vector("black leather ankle boots", top_k=10)
else:
    asins = retriever.retrieve_bm25({"keywords": ["black", "leather", "boots"]}, top_k=10)
```

`retrieve_vector` returns `list[str]` of `parent_asin`, the same contract as
`retrieve_bm25`, so the two paths are interchangeable at the call site.

---

## Design notes

- **Core semantics, complementary to BM25.** `build_doc.py` embeds only
  `title → categories → a bounded description slice`. Brand, `features`, and
  `details` (material, size, fit, closure) are excluded — FTS5 BM25 already
  matches those well. The vector layer instead captures the "what it is / what
  it's for" that survives paraphrase and vocabulary mismatch. Products without a
  description (~48% of the catalog) fall back to `title + categories`.
- **Asymmetric prefixes.** Document/query task prefixes are configurable (see
  *Configuration*), which is what makes EmbeddingGemma work correctly rather
  than silently underperforming.
- **Token-window safety.** Docs are char-capped to stay inside the model's
  window. As a backstop, `EmbeddingClient` detects the server's token-overflow
  error and recovers by splitting a batch and progressively truncating the
  offending item — one bad record never aborts a 50k build.
- **Graceful degradation.** No cache / no numpy / no endpoint ⇒
  `retriever.has_vectors is False` and `retrieve_vector` returns `[]`.

---

## How the cache works

| File | Contents |
|---|---|
| `data/embeddings.npz` | numpy `(n, dim)` float32 matrix (key `vectors`) |
| `data/embeddings.npz.meta.json` | model name, dim, per-row `{parent_asin, text_hash}` in matrix order |

Row order is authoritative and aligned with the metadata. `text_hash` is a
SHA-256 of the curated embed-text, enabling incremental rebuilds. Vectors are
L2-normalized on load, so a query ranks with a single normalized dot product.

---

## Troubleshooting

**`Vector retrieval is NOT available` in `try_it.py`**
Build the cache (`python3 -m scripts.build_embeddings`), set the env vars, and
`pip install numpy`.

**`Pooling type 'none' is not OAI compatible`**
`DOCKER_EMBED_MODEL_NAME` points at a chat model. Pull and use `ai/embeddinggemma`.

**`embedding client unavailable: missing env vars: ...`**
Set `DOCKER_MODEL_BASE_URL`, `DOCKER_MODEL_API_KEY`, `DOCKER_EMBED_MODEL_NAME`.

**`Connection error` / connection refused**
Enable TCP access and confirm the endpoint:
```bash
docker desktop enable model-runner --tcp=12434
curl http://localhost:12434/engines/v1/models
```

**`ModuleNotFoundError: No module named 'numpy'`**
`pip install numpy` into the active environment.

**`input (N tokens) is too large to process` during the build**
A document exceeded the token window. Docs are char-capped and the client
auto-recovers, so the build should not crash. If it still does, lower the cap:
`export DOCKER_EMBED_MAX_INPUT_CHARS=400`.
