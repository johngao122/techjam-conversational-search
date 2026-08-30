"""Standalone latency benchmark -- NOT part of the graded evaluator.

``evaluator/local_evaluator.py`` is the frozen, official grading harness and
must not be modified or instrumented in place. This script reuses its public
data-loading/simulation helpers to replay the same public sessions through
``Agent.respond()`` while timing startup and per-turn latency with stdlib
``time.perf_counter()``. Correctness is intentionally not scored here --
use ``./run.sh eval`` for that.

Usage:
    python3 -m scripts.benchmark_latency [--catalog PATH] [--dataset PATH] [--output PATH]
"""

from __future__ import annotations

import argparse
import json
import resource
import statistics
import sys
import time
import uuid
from pathlib import Path

from evaluator.local_evaluator import (
    MAX_TURNS,
    TOP_K,
    catalog_index,
    coarse_category,
    customer_reply,
    initial_message,
    load_jsonl,
    materialize_hidden_fields,
    normalize_recommendations,
)
from src.agent import Agent


def run_benchmark(catalog_path: str, dataset_path: str) -> dict:
    samples = load_jsonl(dataset_path)

    # Split apart: catalog_index() is the evaluator's own parse, not a cost the
    # agent imposes. Only agent_init_s is ours.
    start = time.perf_counter()
    catalog_ids, categories, products = catalog_index(catalog_path)
    catalog_index_s = time.perf_counter() - start
    start = time.perf_counter()
    agent = Agent(catalog_path)
    agent_init_s = time.perf_counter() - start
    startup_s = catalog_index_s + agent_init_s

    turn_latencies_s: list[float] = []
    reset_latencies_s: list[float] = []
    session_wall_s: list[float] = []

    for sample in samples:
        session_id = f"bench_{uuid.uuid4().hex}"
        reset_start = time.perf_counter()
        agent.reset(session_id, sample["user_profile"])
        reset_latencies_s.append(time.perf_counter() - reset_start)
        target = str(sample["ground_truth"]["parent_asin"])
        effective_intent_card, effective_behavior = materialize_hidden_fields(sample, products)
        effective_sample = {**sample, "intent_card": effective_intent_card, "behavior": effective_behavior}
        disclosed: set[str] = set()
        boundary_used = False
        override_applied = sample["scenario_type"] != "intent_override"
        user_message = initial_message(effective_sample, coarse_category(categories.get(target, [])), disclosed)

        session_start = time.perf_counter()
        for turn in range(1, MAX_TURNS + 1):
            turn_start = time.perf_counter()
            try:
                response = agent.respond(session_id, user_message, turn, TOP_K)
            except Exception:
                response = {"message": "", "ask_attribute": None, "recommendations": []}
            turn_latencies_s.append(time.perf_counter() - turn_start)

            if not isinstance(response, dict) or not isinstance(response.get("message"), str):
                response = {"message": "", "ask_attribute": None, "recommendations": []}
            ranked = normalize_recommendations(response.get("recommendations"), catalog_ids)
            if override_applied and target in ranked:
                break
            if turn == MAX_TURNS:
                break
            override = effective_sample.get("behavior", {}).get("override") or {}
            if not override_applied and turn + 1 == int(override.get("turn", 3)):
                override_applied = True
                new_value = str(override.get("new_value", ""))
                if new_value:
                    disclosed.add(new_value)
                user_message = str(override.get("message", "Actually, please ignore my earlier preference."))
            else:
                user_message, boundary_used = customer_reply(
                    effective_sample, response.get("ask_attribute"), disclosed, boundary_used
                )
        session_wall_s.append(time.perf_counter() - session_start)

    total_s = startup_s + sum(session_wall_s)

    def ms(values: list[float]) -> dict:
        """mean/max plus percentiles -- a lazily-built index landing inside the
        first turn is invisible in a mean but obvious at p99 and in max."""
        if not values:
            return {}
        ordered = sorted(v * 1000 for v in values)
        pct = lambda q: ordered[min(len(ordered) - 1, int(len(ordered) * q))]
        return {
            "mean_ms": round(1000 * statistics.fmean(values), 4),
            "p50_ms": round(pct(0.50), 4),
            "p90_ms": round(pct(0.90), 4),
            "p95_ms": round(pct(0.95), 4),
            "p99_ms": round(pct(0.99), 4),
            "max_ms": round(ordered[-1], 4),
        }

    # ru_maxrss is bytes on macOS, kilobytes on Linux.
    maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    divisor = 1024 * 1024 if sys.platform == "darwin" else 1024
    peak_rss_mb = round(maxrss / divisor, 1)

    return {
        "sample_count": len(samples),
        "catalog_index_s": round(catalog_index_s, 4),
        "agent_init_s": round(agent_init_s, 4),
        "startup_s": round(startup_s, 4),
        "total_eval_s": round(total_s, 4),
        "total_turns": len(turn_latencies_s),
        "peak_rss_mb": peak_rss_mb,
        "first_turn_ms": round(1000 * turn_latencies_s[0], 4) if turn_latencies_s else None,
        "turn": ms(turn_latencies_s),
        "steady_state_turn": ms(turn_latencies_s[1:]),
        "reset": ms(reset_latencies_s),
        "avg_turn_ms": round(1000 * statistics.fmean(turn_latencies_s), 4) if turn_latencies_s else None,
        "max_turn_ms": round(1000 * max(turn_latencies_s), 4) if turn_latencies_s else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Latency benchmark (dev tool, not graded)")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--output", default="docs/latency_baseline.json")
    args = parser.parse_args()

    result = run_benchmark(args.catalog, args.dataset)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
