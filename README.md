# Conversational Shopping Search Agent

**TechJam 2026 — Track 4.** A conversational product-search agent that finds a
customer's hidden target product within ten turns by asking useful clarifying
questions and re-ranking as it learns.

On the 200 public evaluation sessions it reaches a **technical score of 0.969**
(HitRate@10 1.00, MRR 0.968, 2.07 mean turns to conversion), against 0.107 for
the supplied BM25 starter.

---

## Project overview

The task is not purely retrieval. The evaluator freezes a session's reciprocal
rank the moment the target first appears in the top ten, so a weak candidate
shown too early permanently caps the score. We therefore treat each session as
**progressively reducing uncertainty**: show one strong candidate, elicit one
more preference, re-rank, and only widen the list once the intent is clear.

The pipeline is **deterministic and fully offline — no language-model call sits
on the scored path**. A message flows through five stages:

| Stage | What it does |
|---|---|
| **Intent router** | Rule-based parser (catalogue-derived vocabulary) → `buying` / `browsing` / `intent_override` / `boundary`, plus any attributes mentioned. |
| **Ledger + constraint memory** | Per-session state; the customer's disclosed preferences accumulate *in their original wording*, with contradiction-aware supersession (a mind-change replaces only the contradicted preference). |
| **Category-bucket retrieval** | Resolve the coarse product category from turn one, restrict the pool to that bucket, and rank it by weighted string match against the accumulated verbatim constraints; popularity and rating style break ties. |
| **Clarification** | A clarifying question every turn until the customer is exhausted; the question targets whichever attributes are still undisclosed. |
| **Exposure gate** | One candidate on turns 1–2, the full top-ten from turn three (or once intent is exhausted / on the final turn). |

Retrieval or ranking failure falls back to a popularity ordering, so the agent
never raises and always returns a list.

**Agent entry point:** `src/agent.py` exports `Agent`, implementing the
`reset()` / `respond()` contract in `docs/agent_api_contract.json`.

---

## Results

200 public sessions, commit `88daecf`.

| Metric | BM25 baseline | This system |
|---|---|---|
| HitRate@10 | 0.790 | **1.000** |
| MRR | 0.495 | **0.968** |
| Mean turns to conversion | 4.18 | **2.07** |
| Technical score | 0.680 | **0.969** |

| Scenario | n | HitRate@10 | MRR | Turns |
|---|---|---|---|---|
| Buying | 80 | 1.000 | 0.990 | 1.61 |
| Browsing | 80 | 1.000 | 0.957 | 1.90 |
| Intent override | 30 | 1.000 | 0.942 | 3.60 |
| Boundary | 10 | 1.000 | 0.950 | 2.50 |

The target is the first result in 189 of 200 sessions and within the top three
in 197. Cost, tokens, and network calls on the scored path are all zero.

**What the score comes from.** Removing category-first retrieval (searching the
whole catalogue by keyword instead) drops the score to 0.467. The exposure gate
and the contradiction-aware constraint memory are each worth about 0.07.
Overlays that add BM25 or dense-vector retrieval on top of the bucket were tested
and all scored lower, so the shipped path uses neither.

---

## Setup and installation

Python **3.11** (3.10+ should also work).

```bash
git clone <this-repo> && cd techjam-conversational-search
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

**Catalogue.** Download `catalog.jsonl.gz` from the repository's GitHub Release,
verify it against the published `SHA256SUMS`, then:

```bash
gzip -dk catalog.jsonl.gz
mv catalog.jsonl data/catalog.jsonl
```

`data/public_set.jsonl` (200 sessions) is already in the repo.

**No credentials, API keys, network access, or GPU are required.** The `ollama`
and `openai` packages are imported for optional local components (a conversation
summariser and a dense-embedding retriever); both degrade to no-ops when their
backend is absent, which is the configuration every reported number was measured
in.

---

## Steps to reproduce the results

All commands run from the repo root. The evaluator writes per-session results
and aggregate metrics to a JSON file.

```bash
# Headline score → results.json
python3 -m evaluator.local_evaluator

# Component ablations — one switch per run, diffed against a baseline run
python3 scripts/ab_eval.py --label baseline
RETRIEVAL_MODE=legacy      python3 scripts/ab_eval.py --label legacy      --vs baseline
EXPOSURE_GATE=0            python3 scripts/ab_eval.py --label no-exposure --vs baseline
OVERRIDE_POLICY=evict_all  python3 scripts/ab_eval.py --label evict-all   --vs baseline

# Paraphrase robustness — reworded customer replies at two severities
python3 scripts/paraphrase_stress.py --level mild
python3 scripts/paraphrase_stress.py --level aggressive

# Latency and memory profile
python3 -m scripts.benchmark_latency

# Interactive session — type the customer's side yourself
python3 scripts/try_agent.py

# Test suite
python3 -m pytest -q      # or: python3 -m unittest discover -s tests -q
```

Expected headline output: `hit_rate_at_10 1.0`, `mrr ≈ 0.968`, `mttc ≈ 2.07`,
`recommended_technical_score ≈ 0.969`, `reported_token_usage` all zero.

Environment used for the reported figures: Python 3.11, single CPU core, ~1 GB
RAM, commit `88daecf`. The pipeline is deterministic, so the score is
reproducible from a frozen commit.

---

## Limitations, and what we would improve with more time

1. **Paraphrase generalisation.** The score holds under light rewording (0.905)
   but the constraint extractor is tuned to the public set's template phrasing.
   A held-out set that phrases requests very differently would degrade retrieval
   before ranking is even reached — we expect a private-set score nearer
   0.88–0.91. *Next:* widen the constraint parser and make marker-free
   disclosures land as reliably as marked ones.

2. **Category resolution is a single point of failure.** If the opening message
   does not name a resolvable category, the pool becomes the whole catalogue and
   quality drops sharply. *Next:* a semantic category classifier as a fallback,
   rather than falling straight through to whole-catalogue matching.

3. **Budget constraints are effectively inert.** Price is missing for ~79% of
   the catalogue, so a stated budget rarely filters. *Next:* impute or
   range-bucket price, or weight it as a soft signal instead of a hard filter.

4. **No semantic fallback.** BM25 and dense-vector overlays were tested and all
   scored lower, so the shipped path is lexical only. A request whose wording
   shares nothing with the catalogue has nothing to fall back on. *Next:*
   revisit a semantic layer that is gated to only fire when the lexical path is
   weak, rather than always contributing.

5. **Packaging.** `ollama` was missing from `requirements.txt` (fixed here); the
   summariser it feeds is disabled in the response path and the import should be
   made lazy. Our local `evaluator/local_evaluator.py` also carries small edits
   (a `dotenv` import, verbose logging) — the official run uses the frozen
   evaluator, so local and official scores should be reconciled against that.

---

## Repository layout

```text
src/agent.py                             Agent entry point (reset / respond)
src/intent_router/                       rule-based parser + scenario detection
src/intent_router/constraint_memory.py   verbatim constraints + contradiction handling
src/ledger/                              per-session state
src/retrieval/                           category buckets, constraint index, BM25 (ablation only)
src/reranker/                            constraint scoring, popularity tie-breaks
src/confidence/                          ask-vs-recommend policy, exposure gate, fail-open
evaluator/local_evaluator.py             public-set simulator and scorer
scripts/ab_eval.py                       A/B harness (records config, diffs churn)
scripts/paraphrase_stress.py             reworded-reply robustness harness
scripts/benchmark_latency.py             latency / memory profile
scripts/try_agent.py                     interactive REPL for the full pipeline
```

## Data

Derived from Amazon Reviews 2023 (McAuley Lab, UCSD). See
[`DATA_ATTRIBUTION.md`](DATA_ATTRIBUTION.md) before using or redistributing.
