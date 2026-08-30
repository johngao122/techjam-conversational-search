# Semantic Vector Retrieval

This module adds a **standalone semantic retrieval path** alongside the existing
BM25 (SQLite FTS5) retriever. It embeds a *curated* representation of each
product and matches an input query string by cosine similarity.

- `product_embed_text(product)` — builds a compact document capturing the
  product's **core semantics** (title + categories + a short description slice).
  It deliberately **excludes** brand, features, and attribute details — that
  keyword/attribute signal is owned by BM25 (see *Design notes*).
- `EmbeddingClient` — turns text into vectors via an OpenAI-compatible endpoint
  (e.g. Docker Model Runner), configured entirely by environment variables.
- `VectorIndex` — in-memory numpy cosine search over the product vectors.
- `Retriever.retrieve_vector(query, top_k)` — the retrieval entrypoint; returns
  a ranked list of `parent_asin` strings.

Embeddings are computed once by a build script and cached to disk, so startup
and per-query cost stay low. If any prerequisite is missing (numpy, the cache,
or the endpoint), the vector layer disables itself and `retrieve_vector`
returns `[]` — BM25 retrieval is unaffected.

---

## Prerequisites

1. **Python packages** (into your active env, e.g. conda `tiktok2026`):

   ```bash
   pip install numpy openai
   ```

2. **Docker Desktop** with Docker Model Runner enabled (see below).

---

## Setup: Docker Model Runner (embedding model)

Docker Model Runner is built into Docker Desktop — no separate container image
is needed. Note the chat model `ai/llama3.1` **cannot** serve embeddings
(`Pooling type 'none' is not OAI compatible`); you must pull a dedicated
embedding model.

### 1. Enable Docker Model Runner with TCP access

```bash
docker desktop enable model-runner --tcp=12434
```

This exposes the OpenAI-compatible API on `http://localhost:12434`.

### 2. Pull an embedding model

```bash
docker model pull ai/mxbai-embed-large
```

Other embedding models (browse [hub.docker.com/u/ai](https://hub.docker.com/u/ai)):
`ai/nomic-embed-text`, `ai/embeddinggemma`, etc.

### 3. Verify the endpoint

```bash
curl http://localhost:12434/engines/v1/models
# Then confirm the model can embed (should return a JSON with a "data" vector):
curl http://localhost:12434/engines/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model":"ai/mxbai-embed-large","input":"black leather boots"}' | head -c 200
```

---

## Configuration

All connection details are read from **environment variables** — never
hard-code credentials.

| Variable | Description | Value for Docker Model Runner |
|---|---|---|
| `DOCKER_MODEL_BASE_URL` | OpenAI-compatible base URL | `http://localhost:12434/engines/v1` |
| `DOCKER_MODEL_API_KEY` | API key (DMR does not require one) | `none` |
| `DOCKER_EMBED_MODEL_NAME` | Embedding model identifier | `ai/mxbai-embed-large` |
| `DOCKER_EMBED_DOCUMENT_PREFIX` | Task prefix prepended to catalog documents (optional) | *(empty)* |
| `DOCKER_EMBED_QUERY_PREFIX` | Task prefix prepended to search queries (optional) | *(empty)* |

```bash
export DOCKER_MODEL_BASE_URL="http://localhost:12434/engines/v1"
export DOCKER_MODEL_API_KEY="none"
export DOCKER_EMBED_MODEL_NAME="ai/mxbai-embed-large"
```

> `DOCKER_MODEL_BASE_URL` and `DOCKER_MODEL_API_KEY` are shared with the
> `LLMMessageParser`. Only `DOCKER_EMBED_MODEL_NAME` is specific to embeddings.

### Task prefixes (required for some models, e.g. EmbeddingGemma)

Some embedding models are trained **asymmetrically**: documents and queries
must be prefixed with different task instructions, or retrieval quality drops
sharply. Prefix-free models like `ai/mxbai-embed-large` need nothing here
(leave the prefixes empty).

**EmbeddingGemma** requires these prefixes:

```bash
export DOCKER_EMBED_MODEL_NAME="ai/embeddinggemma"
export DOCKER_EMBED_DOCUMENT_PREFIX="title: none | text: "
export DOCKER_EMBED_QUERY_PREFIX="task: search result | query: "
```

The document prefix is applied when building the cache; the query prefix is
applied per search. The document prefix is also folded into the cache's content
hash, so changing it triggers a re-embed on the next build. **Set the same env
vars for both `scripts.build_embeddings` and `src.embeddings.try_it`** — the two
sides must use matching configuration for their embedding spaces to align.

---

## Build the embedding cache

Embed all products once and persist to `data/embeddings.npz` (+ a
`.meta.json` sidecar). Both files are gitignored.

```bash
# Full 50k catalog:
python3 -m scripts.build_embeddings

# Smoke test on the first 200 products:
python3 -m scripts.build_embeddings --limit 200
```

The build is **incremental**: on re-run, only products whose curated text (or
the model) changed are re-embedded; unchanged rows are reused from the existing
cache. Use `--force` to rebuild everything.

Useful flags: `--catalog PATH`, `--out PATH`, `--batch-size N`, `--limit N`,
`--force`.

---

## Run retrieval interactively

```bash
python3 -m src.embeddings.try_it
```

Example session:

```text
Loading catalog + embedding index...
Ready. Vector index holds 50000 products.
Type a query (or 'quit'):

> warm waterproof winter jacket for hiking
   1. B09XXXX1  Men's Waterproof Insulated Mountain Parka ...
   2. B07YYYY2  Columbia Powder Lite Hooded Winter Jacket ...
   ...

> quit
```

If you see *"Vector retrieval is NOT available"*, work through the checklist it
prints (cache built? env vars set? numpy installed?).

---

## Use it in code

```python
from src.catalog import Catalog
from src.retrieval import Retriever

catalog = Catalog("data/catalog.jsonl")

# Enable semantic search if the cache + endpoint + numpy are all present;
# otherwise this silently degrades to a BM25-only retriever.
retriever = Retriever.with_vectors(catalog)

if retriever.has_vectors:
    asins = retriever.retrieve_vector("black leather ankle boots", top_k=10)
else:
    asins = retriever.retrieve_bm25({"keywords": ["black", "leather", "boots"]}, top_k=10)
```

`retrieve_vector` returns `list[str]` of `parent_asin`, the same output contract
as `retrieve_bm25`, so the two paths are interchangeable at the call site.

---

## How the cache works

| File | Contents |
|---|---|
| `data/embeddings.npz` | numpy `(n, dim)` float32 matrix (key `vectors`) |
| `data/embeddings.npz.meta.json` | model name, dim, and per-row `{parent_asin, text_hash}` in matrix order |

- Row order in the matrix is authoritative and aligned with the metadata items.
- `text_hash` is a SHA-256 of the curated embed-text, enabling incremental
  rebuilds.
- Vectors are L2-normalized once when loaded into `VectorIndex`, so a query is
  ranked with a single normalized dot product.

---

## Design notes

- **Asymmetric models supported.** Document/query task prefixes are
  configurable (see *Task prefixes* above), so models like EmbeddingGemma work
  correctly rather than silently underperforming. Defaults are empty for
  prefix-free models.
- **Core semantics, complementary to BM25.** `build_doc.py` embeds only
  `title → categories → a bounded description slice`. It deliberately **excludes**
  brand, `features`, and `details` (material, size, fit, closure, etc.). That
  keyword/attribute signal is already matched well by FTS5 BM25 over those
  fields; duplicating it in the embedding adds nothing. The vector layer instead
  captures the conceptual "what it is / what it's for" that survives paraphrase
  and vocabulary mismatch — which BM25 cannot. Products without a description
  (~48% of the catalog) fall back to `title + categories`.
- **Why not parse attributes from the description?** Considered and rejected.
  Running the message parser over descriptions reproduces exactly the
  material/category/brand signal BM25 already handles (recreating the redundancy
  we removed), and the parser — tuned for short customer messages — is noisy on
  long marketing prose. The raw description slice preserves the paraphrasable
  semantics that are this layer's value-add.
- **Token-window safety.** The description slice + overall doc are char-capped so
  a doc stays well inside a 512-token window. As a backstop for pathological
  records, `EmbeddingClient` detects the server's token-overflow error and
  recovers by splitting a batch (to isolate the offending item) and
  progressively truncating that item until it fits — so one bad record never
  aborts a 50k build.
- **Graceful degradation.** No cache / no numpy / no endpoint ⇒
  `retriever.has_vectors is False` and `retrieve_vector` returns `[]`.
- **Fusion / reranking is intentionally out of scope** here — this module
  provides an independent `retrieve_vector`; blending it with BM25 is left to a
  later step.

---

## Troubleshooting

**`Vector retrieval is NOT available` in `try_it.py`**
Build the cache (`python3 -m scripts.build_embeddings`), set the three env
vars, and `pip install numpy`.

**`Pooling type 'none' is not OAI compatible`**
You pointed `DOCKER_EMBED_MODEL_NAME` at a chat model (e.g. `ai/llama3.1`). Pull
and use a real embedding model such as `ai/mxbai-embed-large`.

**`embedding client unavailable: missing env vars: ...`**
Set `DOCKER_MODEL_BASE_URL`, `DOCKER_MODEL_API_KEY`, `DOCKER_EMBED_MODEL_NAME`.

**`LLM API call failed: Connection error.` / connection refused**
Enable TCP access and confirm the endpoint:
```bash
docker desktop enable model-runner --tcp=12434
curl http://localhost:12434/engines/v1/models
```

**`ModuleNotFoundError: No module named 'numpy'`**
`pip install numpy` into the active environment.

**`input (N tokens) is too large to process` during the build**
A single document exceeded the model's token window (e.g. 512). Curated docs are
char-capped to avoid this, and the client auto-recovers by truncating the
offending input, so the build should no longer crash. If you still see it,
lower the pre-filter cap: `export DOCKER_EMBED_MAX_INPUT_CHARS=400`.
```
