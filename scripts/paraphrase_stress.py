"""Paraphrase stress evaluation: run the full evaluator protocol against
reworded reply templates, at a selectable severity.

The public set uses fixed template strings; the private 800 explicitly reserves
the right to paraphrase (docs/competition_specification.md:40). This harness
monkeypatches the evaluator's ``initial_message`` / ``customer_reply`` to emit
reworded variants, then runs the real protocol (real disclosure schedule, real
override timing) so the score is directly comparable to the clean run.

Levels:
  none        the unmodified templates (sanity: should equal the clean eval)
  mild        reworded wrappers + reordered clauses, template MARKERS KEPT
              ("A key requirement is:", "what matters is:") -- byteme measured
              this class of perturbation as ~free
  aggressive  markers dropped, ";" -> "and also", synonym wrappers -- the worst
              case; exercises the transcript fallback rungs

Usage::

    python3 scripts/paraphrase_stress.py                       # aggressive (default)
    python3 scripts/paraphrase_stress.py --level mild
    python3 scripts/paraphrase_stress.py --level none --min-score 0.96
    python3 scripts/paraphrase_stress.py --level aggressive --output runs/stress.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import evaluator.local_evaluator as ev  # noqa: E402
from src.agent import Agent  # noqa: E402
from scripts.generate_paraphrase_dataset import generate as generate_dataset  # noqa: E402


# --------------------------------------------------------------------------
# Reword functions per level. Each returns a drop-in for the evaluator symbol.
# --------------------------------------------------------------------------

def _openers(level: str):
    def none(sample, category, disclosed):
        return ev._ORIG_INITIAL(sample, category, disclosed)

    def mild(sample, category, disclosed):
        scenario = sample["scenario_type"]
        if scenario == "buying" and sample["intent_card"].get("hard_constraints"):
            c = str(sample["intent_card"]["hard_constraints"][0])
            disclosed.add(c)
            # Marker KEPT, wrapper reworded, category moved.
            return f"Hi! A key requirement is: {c}. Anyway I want {category}."
        if scenario == "intent_override":
            old = str(sample["behavior"]["override"]["old_value"])
            return f"Show me {category}. {old}"
        return f"browsing {category} right now"

    def aggressive(sample, category, disclosed):
        scenario = sample["scenario_type"]
        if scenario == "buying" and sample["intent_card"].get("hard_constraints"):
            c = str(sample["intent_card"]["hard_constraints"][0])
            disclosed.add(c)
            # Marker DROPPED, wrapper reworded, clause reordered.
            return f"Hey, I really need {c} and I'm after some {category} today."
        if scenario == "intent_override":
            old = str(sample["behavior"]["override"]["old_value"])
            return f"Hi there, show me some {category} - {old}"
        return f"just browsing for {category} at the moment"

    return {"none": none, "mild": mild, "aggressive": aggressive}[level]


def _repliers(level: str):
    def _matches(sample, attribute, disclosed):
        if attribute not in ev.ALLOWED_ATTRIBUTES:
            attribute = "other"
        constraints = [
            *[str(v) for v in sample["intent_card"].get("hard_constraints", [])],
            *[str(v) for v in sample["intent_card"].get("soft_preferences", [])],
        ]
        m = [
            v for v in constraints
            if v not in disclosed
            and (attribute == "other" or ev.classify_constraint(v) == attribute)
        ][:2]
        return attribute, m

    def none(sample, ask_attribute, disclosed, boundary_used):
        return ev._ORIG_REPLY(sample, ask_attribute, disclosed, boundary_used)

    def mild(sample, ask_attribute, disclosed, boundary_used):
        attribute = ask_attribute if isinstance(ask_attribute, str) else None
        if sample["scenario_type"] == "boundary" and not boundary_used and attribute:
            return f"no strong feeling on {attribute}, your call", True
        if not attribute:
            return "those aren't right, ask me something specific", boundary_used
        attribute, matches = _matches(sample, attribute, disclosed)
        if not matches:
            return f"nothing else on {attribute}", boundary_used
        disclosed.update(matches)
        # Marker KEPT, only the join reworded.
        return "For that, what matters is: " + " and ".join(matches) + ".", boundary_used

    def aggressive(sample, ask_attribute, disclosed, boundary_used):
        attribute = ask_attribute if isinstance(ask_attribute, str) else None
        if sample["scenario_type"] == "boundary" and not boundary_used and attribute:
            return f"no strong feeling on {attribute}, your call", True
        if not attribute:
            return "hmm those aren't right, ask me something specific", boundary_used
        attribute, matches = _matches(sample, attribute, disclosed)
        if not matches:
            return f"nothing else on {attribute} sorry", boundary_used
        disclosed.update(matches)
        # Marker DROPPED, ";" -> "and also".
        return "well " + " and also ".join(matches) + " would be great", boundary_used

    return {"none": none, "mild": mild, "aggressive": aggressive}[level]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--level", choices=("none", "mild", "aggressive"),
                        default="aggressive", help="paraphrase severity")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--output", default=None, help="write full result JSON here")
    parser.add_argument("--min-score", type=float, default=0.80,
                        help="pass/fail threshold on recommended_technical_score")
    args = parser.parse_args()

    # Preserve the originals so level=none is a true passthrough.
    ev._ORIG_INITIAL = ev.initial_message
    ev._ORIG_REPLY = ev.customer_reply
    ev.initial_message = _openers(args.level)
    ev.customer_reply = _repliers(args.level)

    samples = ev.load_jsonl(args.dataset)
    catalog_ids, categories, products = ev.catalog_index(args.catalog)
    result = ev.evaluate(Agent(args.catalog), samples, catalog_ids, categories, products)

    print(f"=== paraphrase stress: level={args.level} ({len(samples)} sessions) ===")
    for k in ("hit_rate_at_10", "mrr", "mttc", "recommended_technical_score"):
        print(f"  {k:32s} {result[k]}")
    for name, s in sorted(result["scenario_metrics"].items()):
        print(f"    {name:16s} hit={s['hit_rate_at_10']:.3f} "
              f"mrr={s['mrr']:.3f} mttc={s['mttc']:.2f}")

    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    score = result["recommended_technical_score"]
    verdict = "PASS" if score >= args.min_score else f"BELOW {args.min_score}"
    print(f"\n{verdict}: score {score:.4f} (threshold {args.min_score})")
    sys.exit(0 if score >= args.min_score else 1)


if __name__ == "__main__":
    main()
