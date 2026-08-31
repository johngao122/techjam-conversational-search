"""Record one clean multi-turn demo session for the deliverable video.

Replays a single public_set.jsonl sample through the real Agent using the same
simulated-customer logic as evaluator/local_evaluator.py (initial message,
attribute-targeted follow-ups, intent-override injection), and prints a
transcript formatted for screen capture. Also writes the same transcript to
a JSON file so it can be attached alongside the video.

Run from the repo root:
    python3 scripts/demo_session.py                     # default: public_0003 (intent_override, hits turn 3)
    python3 scripts/demo_session.py --sample public_0007 # pick any sample_id from data/public_set.jsonl
    python3 scripts/demo_session.py --list               # show good candidate sample_ids per scenario
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

from src.agent import Agent  # noqa: E402
from src.catalog.loader import load_catalog_rows  # noqa: E402
from evaluator.local_evaluator import (  # noqa: E402
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

# evaluator.local_evaluator sets the root logger to DEBUG on import for its
# own ad-hoc debugging; that's too noisy for a screen-recorded demo.
logging.getLogger().setLevel(logging.WARNING)
logging.getLogger("src.retrieval.strategies").setLevel(logging.WARNING)

_TITLE_MAX = 78
_DEFAULT_SAMPLE = "public_0003"


def _load_titles(catalog_path: str) -> dict[str, str]:
    return {row["parent_asin"]: row["title"] for row in load_catalog_rows(catalog_path)}


def _short_title(titles: dict[str, str], asin: str) -> str:
    title = titles.get(asin, "(title unavailable)")
    return title if len(title) <= _TITLE_MAX else title[: _TITLE_MAX - 1] + "…"


def list_candidates(dataset_path: str, results_path: str) -> None:
    samples = {s["sample_id"]: s for s in load_jsonl(dataset_path)}
    results_file = Path(results_path)
    if not results_file.exists():
        print(f"No {results_path} found -- run the evaluator first to get hit/turn data.")
        return
    sessions = json.loads(results_file.read_text())["sessions"]
    by_scenario: dict[str, list[dict]] = {}
    for session in sessions:
        if session["hit"] and session["sample_id"] in samples:
            by_scenario.setdefault(session["scenario_type"], []).append(session)
    for scenario, items in sorted(by_scenario.items()):
        items.sort(key=lambda s: (s["first_hit_turn"], s["best_rank"]))
        print(f"\n{scenario}:")
        for item in items[:5]:
            print(
                f"  {item['sample_id']}  hit_turn={item['first_hit_turn']}  "
                f"rank={item['best_rank']}"
            )


def run_demo(sample_id: str, catalog_path: str, dataset_path: str, transcript_out: str) -> None:
    samples = {s["sample_id"]: s for s in load_jsonl(dataset_path)}
    if sample_id not in samples:
        raise SystemExit(f"sample_id {sample_id!r} not found in {dataset_path}")
    sample = samples[sample_id]

    catalog_ids, categories, products = catalog_index(catalog_path)
    titles = _load_titles(catalog_path)

    print("Building the agent (FTS5 index + catalog load)...")
    agent = Agent(catalog_path)
    print("Ready.\n")

    target = str(sample["ground_truth"]["parent_asin"])
    intent_card, behavior = materialize_hidden_fields(sample, products)
    effective_sample = {**sample, "intent_card": intent_card, "behavior": behavior}
    scenario = sample["scenario_type"]

    print(f"{'=' * 72}")
    print(f"DEMO SESSION  sample_id={sample_id}  scenario={scenario}")
    print(f"Target product (hidden from the Agent): {target}  -- {_short_title(titles, target)}")
    print(f"{'=' * 72}\n")

    session_id = f"demo_{sample_id}"
    agent.reset(session_id, sample["user_profile"])

    disclosed: set[str] = set()
    boundary_used = False
    override_applied = scenario != "intent_override"
    user_message = initial_message(effective_sample, coarse_category(categories.get(target, [])), disclosed)

    transcript: list[dict] = []
    hit_turn: int | None = None
    best_rank: int | None = None

    for turn in range(1, MAX_TURNS + 1):
        print(f"--- Turn {turn} ---")
        print(f"Customer: {user_message}")

        response = agent.respond(session_id, user_message, turn, TOP_K)
        ranked = normalize_recommendations(response.get("recommendations"), catalog_ids)
        hit = target in ranked
        rank = ranked.index(target) + 1 if hit else None

        print(f"Agent: {response['message']}")
        if response.get("ask_attribute"):
            print(f"  (ask_attribute: {response['ask_attribute']})")
        print("  Recommendations:")
        for i, asin in enumerate(ranked[:5], start=1):
            marker = "  <-- TARGET" if asin == target else ""
            print(f"    {i}. {_short_title(titles, asin)}  [{asin}]{marker}")
        if not ranked:
            print("    (none)")
        print()

        transcript.append({
            "turn": turn,
            "customer_message": user_message,
            "agent_message": response["message"],
            "ask_attribute": response.get("ask_attribute"),
            "recommendations": ranked[:TOP_K],
            "target_rank_this_turn": rank,
        })

        if hit and override_applied:
            hit_turn, best_rank = turn, rank
            print(f"*** HIT on turn {turn}, rank {rank} (target={target}) ***\n")
            break

        if turn == MAX_TURNS:
            print("Reached the 10-turn cap without a hit.\n")
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

    summary = {
        "sample_id": sample_id,
        "scenario_type": scenario,
        "target": target,
        "hit": hit_turn is not None,
        "first_hit_turn": hit_turn,
        "best_rank": best_rank,
        "turns": transcript,
    }
    Path(transcript_out).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"{'=' * 72}")
    print(f"Transcript written to {transcript_out}")
    print(f"{'=' * 72}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run and record one demo multi-turn session")
    parser.add_argument("--sample", default=_DEFAULT_SAMPLE, help="sample_id from data/public_set.jsonl")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--results", default="results_ours.json", help="used only by --list")
    parser.add_argument("--output", default="runs/demo_transcript.json")
    parser.add_argument("--list", action="store_true", help="list good candidate sample_ids and exit")
    args = parser.parse_args()

    if args.list:
        list_candidates(args.dataset, args.results)
        return

    run_demo(args.sample, args.catalog, args.dataset, args.output)


if __name__ == "__main__":
    main()
