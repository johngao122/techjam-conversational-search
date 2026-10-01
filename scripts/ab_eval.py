"""A/B evaluation harness with per-session churn diff.

Every pipeline change lands behind an env flag, and every run is recorded
append-only so a change that adds three hits while silently breaking one is
visible. Usage::

    python3 scripts/ab_eval.py --label baseline
    RETRIEVAL_MODE=bucket python3 scripts/ab_eval.py --label bucket --vs baseline

Runs are stored as ``runs/<label>.json`` (the evaluator's full output) and
summarised append-only in ``runs/log.jsonl``.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl  # noqa: E402

RUNS = Path(__file__).resolve().parent.parent / "runs"

# Env flags that define a configuration. Recorded with every run so a result
# can always be traced back to the exact switch settings that produced it.
CONFIG_FLAGS = (
    "RETRIEVAL_MODE",
    "IDF_WEIGHT",
    "EXPOSURE_GATE",
    "RELEASE_TURN",
    "OVERRIDE_POLICY",
    "TIER_BM25_WEIGHT",
    "TIER_VECTOR_WEIGHT",
    "TIER_RRF_K",
)


def current_config() -> dict[str, str]:
    return {flag: os.environ.get(flag, "") for flag in CONFIG_FLAGS}


def run_eval(catalog: str, dataset: str) -> dict:
    from src.agent import Agent

    samples = load_jsonl(dataset)
    catalog_ids, categories, products = catalog_index(catalog)
    return evaluate(Agent(catalog), samples, catalog_ids, categories, products)


def summarise(result: dict) -> dict:
    return {
        key: result[key]
        for key in ("hit_rate_at_10", "mrr", "mttc", "efficiency", "recommended_technical_score")
    }


def churn(baseline: dict, current: dict) -> dict:
    """Per-session diff: what this change actually broke, not just the net."""
    before = {s["sample_id"]: s for s in baseline["sessions"]}
    after = {s["sample_id"]: s for s in current["sessions"]}
    gained, lost, better, worse = [], [], [], []
    for sid, now in after.items():
        was = before.get(sid)
        if was is None:
            continue
        if now["hit"] and not was["hit"]:
            gained.append(sid)
        elif was["hit"] and not now["hit"]:
            lost.append(sid)
        elif now["hit"] and was["hit"]:
            if now["reciprocal_rank"] > was["reciprocal_rank"]:
                better.append((sid, was["best_rank"], now["best_rank"]))
            elif now["reciprocal_rank"] < was["reciprocal_rank"]:
                worse.append((sid, was["best_rank"], now["best_rank"]))
    return {"gained": gained, "lost": lost, "rank_better": better, "rank_worse": worse}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True, help="name for this run (runs/<label>.json)")
    parser.add_argument("--vs", default=None, help="baseline label to diff against")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--note", default="", help="what this run is testing")
    args = parser.parse_args()

    RUNS.mkdir(exist_ok=True)
    config = current_config()
    result = run_eval(args.catalog, args.dataset)
    metrics = summarise(result)

    (RUNS / f"{args.label}.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    with (RUNS / "log.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({
            "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
            "label": args.label, "note": args.note, "config": config,
            "dataset": args.dataset, **metrics,
        }) + "\n")

    print(f"\n=== {args.label} ===")
    print("config:", {k: v for k, v in config.items() if v} or "(all defaults)")
    for key, value in metrics.items():
        print(f"  {key:32s} {value}")
    for name, scenario in sorted(result["scenario_metrics"].items()):
        print(f"    {name:16s} hit={scenario['hit_rate_at_10']:.3f} "
              f"mrr={scenario['mrr']:.3f} mttc={scenario['mttc']:.2f}")

    if args.vs:
        baseline_path = RUNS / f"{args.vs}.json"
        if not baseline_path.exists():
            print(f"\n! no baseline run at {baseline_path}")
            return
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        delta = metrics["recommended_technical_score"] - baseline["recommended_technical_score"]
        print(f"\n=== churn vs {args.vs} ===")
        print(f"  score {baseline['recommended_technical_score']:.4f} -> "
              f"{metrics['recommended_technical_score']:.4f}  ({delta:+.4f})")
        diff = churn(baseline, result)
        print(f"  hits gained: {len(diff['gained'])}  hits LOST: {len(diff['lost'])}")
        if diff["lost"]:
            print("    lost:", ", ".join(diff["lost"]))
        print(f"  rank better: {len(diff['rank_better'])}  rank worse: {len(diff['rank_worse'])}")
        if diff["rank_worse"]:
            print("    worse:", ", ".join(f"{s}({a}->{b})" for s, a, b in diff["rank_worse"][:15]))


if __name__ == "__main__":
    main()
