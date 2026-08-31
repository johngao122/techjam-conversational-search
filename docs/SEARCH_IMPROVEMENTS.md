# Search Performance Improvement Analysis

## Current Implementation Summary

### Architecture
```
┌─────────────────────────────────────────────────────────────────┐
│                      HybridRetriever                            │
├─────────────────────────────────────────────────────────────────┤
│  1. Constraint Pre-filter (Inverted Index)                      │
│     └── O(1) exact match, O(n) substring/token fallback         │
│                                                                 │
│  2. Parallel Search (ThreadPoolExecutor)                        │
│     ├── BM25 (SQLite FTS5)                                      │
│     └── Vector Search (numpy brute-force cosine)                │
│                                                                 │
│  3. Weighted RRF Fusion                                         │
│     └── weights: (6.0 constraint, 1.0 BM25, 1.0 vector)         │
└─────────────────────────────────────────────────────────────────┘
```

### Current Bottlenecks

| Component | Issue | Impact |
|-----------|-------|--------|
| **Vector Search** | Brute-force O(n) cosine similarity | Slow for large catalogs |
| **Constraint Fallback** | O(n) scan for substring/token matching | 50k products scanned per query |
| **Query Embedding** | Fresh embedding per query (no cache) | ~100-200ms API latency per query |
| **RRF Fusion** | Static weights, not learned | Suboptimal ranking combination |
| **No Re-ranking** | Single-stage retrieval | Missing cross-encoder precision boost |

---

## Improvement Recommendations

### Tier 1: Quick Wins (Low Effort, High Impact)

#### 1.1 Query Embedding Cache
**Current**: Fresh API call for every vector query  
**Improvement**: Cache query embeddings by normalized query text

```python
# src/retrieval/hybrid.py - Add query embedding cache
class HybridRetriever:
    def __init__(self, ...):
        ...
        self._query_cache: dict[str, np.ndarray] = {}
        self._cache_max = 1000
    
    def _embed_query(self, query: str) -> np.ndarray:
        key = query.strip().lower()
        if key in self._query_cache:
            return self._query_cache[key]
        vec = self._retriever.embedding_client.embed_one(query)
        if len(self._query_cache) >= self._cache_max:
            self._query_cache.pop(next(iter(self._query_cache)))
        self._query_cache[key] = vec
        return vec
```

**Expected Impact**: 50-80% reduction in vector search latency for repeat/similar queries

---

#### 1.2 HNSW Index for Vector Search
**Current**: Brute-force numpy dot product O(n)  
**Improvement**: Use HNSW (Hierarchical Navigable Small Worlds) for O(log n) ANN

```python
# Option A: hnswlib (lightweight, C++ bindings)
import hnswlib

class HNSWIndex:
    def __init__(self, dim: int, max_elements: int = 100000):
        self.index = hnswlib.Index(space='cosine', dim=dim)
        self.index.init_index(max_elements=max_elements, ef_construction=200, M=16)
        self.asins: list[str] = []
    
    def add(self, asins: list[str], vectors: np.ndarray):
        self.index.add_items(vectors, list(range(len(asins))))
        self.asins = asins
    
    def search(self, query: np.ndarray, k: int = 10) -> list[tuple[str, float]]:
        labels, distances = self.index.knn_query(query.reshape(1, -1), k=k)
        return [(self.asins[i], 1 - d) for i, d in zip(labels[0], distances[0])]

# Option B: faiss (Facebook, more features)
import faiss

class FAISSIndex:
    def __init__(self, dim: int):
        self.index = faiss.IndexHNSWFlat(dim, 32)  # 32 neighbors per node
        self.index.hnsw.efSearch = 64
        self.asins: list[str] = []
    
    def search(self, query: np.ndarray, k: int = 10) -> list[tuple[str, float]]:
        D, I = self.index.search(query.reshape(1, -1), k)
        return [(self.asins[i], float(D[0][j])) for j, i in enumerate(I[0]) if i >= 0]
```

**Expected Impact**: 10-50x faster vector search (50ms -> 1-5ms for 50k vectors)

---

#### 1.3 Inverted Index for Token Matching
**Current**: O(n) text scan for substring/token fallback  
**Improvement**: Already partially implemented! But not fully utilized.

```python
# Current: _token_inverted exists but fast_candidates still scans
# Fix: Use token inverted index for Tier 3 matching

def fast_candidates(self, constraints, exact_only=False) -> dict[str, float]:
    scores: dict[str, float] = {}
    
    for normalised, tokens, weight in constraints:
        # Tier 1: Exact match (already O(1))
        exact_matches = self._inverted.get(normalised, set())
        ...
        
        # Tier 2: Substring - still needs scan (hard to index)
        # Tier 3: Token containment - USE INVERTED INDEX!
        if tokens:
            # Intersect token posting lists instead of scanning
            token_candidates = None
            for token in tokens:
                posting = self._token_inverted.get(token, set())
                token_candidates = posting if token_candidates is None else (token_candidates & posting)
            for asin in (token_candidates or set()):
                if asin not in exact_matched and asin not in substring_matched:
                    scores[asin] = scores.get(asin, 0.0) + weight * TOKEN_WEIGHT
```

**Expected Impact**: 5-10x faster constraint filtering for token-based matches

---

### Tier 2: Medium Effort, High Impact

#### 2.1 Cross-Encoder Re-ranking
**Current**: Single-stage retrieval, no neural re-ranking  
**Improvement**: Add cross-encoder for top-k precision boost

```python
# src/reranker/cross_encoder.py
from sentence_transformers import CrossEncoder

class CrossEncoderReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name, max_length=512)
    
    def rerank(self, query: str, candidates: list[dict], top_k: int = 10) -> list[dict]:
        """Re-rank candidates using cross-encoder scores."""
        if not candidates:
            return []
        
        # Build query-document pairs
        pairs = [(query, c.get("title", "") + " " + c.get("description", "")[:200]) 
                 for c in candidates]
        
        # Score all pairs
        scores = self.model.predict(pairs)
        
        # Sort by score
        scored = list(zip(candidates, scores))
        scored.sort(key=lambda x: x[1], reverse=True)
        
        return [c for c, _ in scored[:top_k]]
```

**Integration point**: After RRF fusion, before returning top-k
**Expected Impact**: +2-5% MRR improvement, especially for ambiguous queries

---

#### 2.2 Learned Fusion Weights
**Current**: Static RRF weights (6.0, 1.0, 1.0)  
**Improvement**: Learn optimal weights from evaluation data

```python
# scripts/tune_rrf_weights.py
from scipy.optimize import minimize
import numpy as np

def tune_rrf_weights(eval_samples: list[dict], retriever: HybridRetriever):
    """Find optimal RRF weights via grid search or optimization."""
    
    def score_fn(weights):
        retriever._rrf_weights = tuple(weights)
        total_mrr = 0
        for sample in eval_samples:
            results = retriever.retrieve(...)
            # Calculate MRR
            target = sample["target_asin"]
            if target in results:
                rank = results.index(target) + 1
                total_mrr += 1.0 / rank
        return -total_mrr / len(eval_samples)  # Negative for minimization
    
    # Optimize
    result = minimize(
        score_fn,
        x0=[6.0, 1.0, 1.0],
        bounds=[(0, 10), (0, 10), (0, 10)],
        method='L-BFGS-B'
    )
    return tuple(result.x)
```

**Expected Impact**: +1-3% technical score improvement

---

#### 2.3 Query Expansion with LLM
**Current**: Raw user query used directly  
**Improvement**: Expand query with synonyms/related terms

```python
# src/retrieval/query_expansion.py

EXPANSION_PROMPT = """Given this product search query, generate 3-5 related search terms.
Query: {query}
Related terms (comma-separated):"""

def expand_query(query: str, llm_client) -> list[str]:
    """Expand query with LLM-generated related terms."""
    response = llm_client.complete(EXPANSION_PROMPT.format(query=query))
    terms = [t.strip() for t in response.split(",")]
    return [query] + terms[:5]

# Alternative: Static synonym expansion (faster, no LLM)
SYNONYMS = {
    "cheap": ["affordable", "budget", "inexpensive", "low-cost"],
    "good": ["quality", "excellent", "great", "top-rated"],
    "big": ["large", "spacious", "oversized", "xl"],
    ...
}
```

**Expected Impact**: +2-4% recall for paraphrased queries

---

### Tier 3: High Effort, Transformative

#### 3.1 Replace BM25 with Elasticsearch/OpenSearch

| Feature | SQLite FTS5 (Current) | Elasticsearch |
|---------|----------------------|---------------|
| Scalability | ~100k docs | Billions |
| Fuzzy matching | None | Built-in |
| Synonyms | Manual | Configurable |
| Faceted search | None | Built-in |
| Distributed | No | Yes |
| Learning to Rank | No | Plugin available |

```python
# src/retrieval/elasticsearch_retriever.py
from elasticsearch import Elasticsearch

class ElasticsearchRetriever:
    def __init__(self, index_name: str = "products"):
        self.es = Elasticsearch()
        self.index = index_name
    
    def search(self, query: str, filters: dict, top_k: int = 10) -> list[str]:
        body = {
            "query": {
                "bool": {
                    "must": [
                        {"multi_match": {
                            "query": query,
                            "fields": ["title^6", "categories^4", "features^2.5", "description"],
                            "fuzziness": "AUTO"
                        }}
                    ],
                    "filter": self._build_filters(filters)
                }
            },
            "size": top_k
        }
        results = self.es.search(index=self.index, body=body)
        return [hit["_source"]["parent_asin"] for hit in results["hits"]["hits"]]
```

**Expected Impact**: Better fuzzy matching, synonyms, scalability

---

#### 3.2 SPLADE: Learned Sparse Representations
**Current**: BM25 (term frequency based)  
**Improvement**: SPLADE learns query/document expansion

```python
# SPLADE produces sparse vectors that combine BM25's interpretability
# with neural network's semantic understanding

from transformers import AutoModelForMaskedLM, AutoTokenizer

class SPLADEEncoder:
    def __init__(self, model_name: str = "naver/splade-cocondenser-ensembledistil"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)
    
    def encode(self, text: str) -> dict[str, float]:
        """Encode text to sparse vector (term -> weight)."""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True)
        with torch.no_grad():
            logits = self.model(**inputs).logits
        
        # ReLU + log to get sparse weights
        weights = torch.log1p(torch.relu(logits)).max(dim=1).values.squeeze()
        
        # Extract non-zero terms
        sparse = {}
        for idx, weight in enumerate(weights):
            if weight > 0:
                term = self.tokenizer.decode([idx])
                sparse[term] = float(weight)
        return sparse
```

**Expected Impact**: +3-5% MRR, better semantic matching than BM25

---

#### 3.3 ColBERT: Late Interaction for Efficient Re-ranking
**Current**: No neural re-ranking  
**Improvement**: ColBERT provides cross-encoder quality at bi-encoder speed

```python
# ColBERT: Each query/doc token gets its own embedding
# Score = sum of max similarities (late interaction)

from colbert import Indexer, Searcher
from colbert.infra import Run, RunConfig

# Index products once
with Run().context(RunConfig(nranks=1)):
    indexer = Indexer(checkpoint="colbert-ir/colbertv2.0")
    indexer.index(name="products", collection=product_texts)

# Search
searcher = Searcher(index="products")
results = searcher.search(query, k=100)
```

**Expected Impact**: Cross-encoder quality (+2-5% MRR) at 10x speed

---

## Open-Source Search Engine Comparison

| Engine | Best For | Hybrid Search | Learning to Rank | Ease of Use |
|--------|----------|---------------|------------------|-------------|
| **Elasticsearch** | Enterprise, scale | Yes (RRF built-in) | Plugin | Medium |
| **Meilisearch** | Typo tolerance, speed | No (keyword only) | No | Easy |
| **Typesense** | Speed, simplicity | Yes (vector+keyword) | No | Easy |
| **Qdrant** | Vector-first, filtering | Yes | No | Medium |
| **Weaviate** | Hybrid, GraphQL | Yes (excellent) | No | Medium |
| **Vespa** | ML-native, scale | Yes (best-in-class) | Built-in | Hard |
| **LanceDB** | Embedded, serverless | Yes | No | Easy |

### Recommendation for This Project

**For hackathon/MVP**: 
1. **Qdrant** or **LanceDB** - Easy setup, good hybrid search, can run embedded

**For production scale**:
1. **Vespa** - Best hybrid search, built-in LTR, handles this exact use case
2. **Weaviate** - Great hybrid, easier than Vespa

---

## Implementation Priority Matrix

```
                    HIGH IMPACT
                        │
    ┌───────────────────┼───────────────────┐
    │                   │                   │
    │  HNSW Index       │  Cross-Encoder    │
    │  Query Cache      │  SPLADE           │
    │  Token Index Fix  │  ColBERT          │
    │                   │                   │
LOW ├───────────────────┼───────────────────┤ HIGH
EFFORT                  │                   EFFORT
    │                   │                   │
    │  Learned Weights  │  Elasticsearch    │
    │  Query Expansion  │  Vespa Migration  │
    │                   │                   │
    └───────────────────┼───────────────────┘
                        │
                    LOW IMPACT
```

## Quick Start: Implementing HNSW

```bash
# Install hnswlib
pip install hnswlib

# Or faiss (more features, harder install on Mac)
pip install faiss-cpu  # or faiss-gpu
```

```python
# src/embeddings/hnsw_index.py
import hnswlib
import numpy as np
from pathlib import Path

class HNSWIndex:
    """HNSW index for approximate nearest neighbor search."""
    
    def __init__(self, dim: int = 1536, max_elements: int = 100000):
        self.dim = dim
        self.index = hnswlib.Index(space='cosine', dim=dim)
        self.index.init_index(max_elements=max_elements, ef_construction=200, M=16)
        self.index.set_ef(64)  # Query time accuracy/speed tradeoff
        self.asins: list[str] = []
    
    def add(self, asins: list[str], vectors: np.ndarray):
        """Add vectors to index."""
        self.index.add_items(vectors, list(range(len(asins))))
        self.asins = asins
    
    def search(self, query: np.ndarray, top_k: int = 10) -> list[tuple[str, float]]:
        """Search for nearest neighbors."""
        labels, distances = self.index.knn_query(query.reshape(1, -1), k=min(top_k, len(self.asins)))
        # hnswlib returns distances, convert to similarities for cosine
        return [(self.asins[int(i)], 1.0 - float(d)) for i, d in zip(labels[0], distances[0])]
    
    def save(self, path: str | Path):
        """Save index to disk."""
        self.index.save_index(str(path))
    
    @classmethod
    def load(cls, path: str | Path, asins: list[str], dim: int = 1536) -> "HNSWIndex":
        """Load index from disk."""
        idx = cls(dim=dim, max_elements=len(asins))
        idx.index.load_index(str(path), max_elements=len(asins))
        idx.asins = asins
        return idx
```

---

## Metrics to Track

| Metric | Current | Target | Measurement |
|--------|---------|--------|-------------|
| Hit Rate @10 | 0.98 | 0.99+ | % queries where target in top 10 |
| MRR | 0.86 | 0.90+ | Mean reciprocal rank |
| MTTC | 2.78 | 2.5 | Mean turns to conversion |
| Technical Score | 0.91 | 0.95+ | Weighted composite |
| P95 Latency | ~500ms | <200ms | 95th percentile response time |

---

## Next Steps

1. **Immediate** (this week):
   - [ ] Add query embedding cache
   - [ ] Fix token inverted index usage in `fast_candidates`
   - [ ] Benchmark current latency

2. **Short-term** (next sprint):
   - [ ] Implement HNSW index
   - [ ] Add cross-encoder re-ranking (optional, behind flag)
   - [ ] Tune RRF weights on eval set

3. **Medium-term** (next month):
   - [ ] Evaluate Qdrant/LanceDB for embedded vector DB
   - [ ] Implement query expansion
   - [ ] A/B test improvements

4. **Long-term** (future):
   - [ ] SPLADE or ColBERT for neural sparse/dense
   - [ ] Vespa migration for production scale
   - [ ] Learning to Rank integration
