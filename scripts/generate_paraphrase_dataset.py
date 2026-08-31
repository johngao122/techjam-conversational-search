"""Generate a fresh evaluation dataset for paraphrase_stress.py.

Draws NEW sessions from data/catalog.jsonl -- distinct parent_asins from
those already used in data/public_set.jsonl -- so paraphrase robustness is
measured against unseen targets rather than the public dev set. Output
schema matches public_set.jsonl exactly (sample_id, scenario_type,
category_bucket, difficulty_bucket, ground_truth.parent_asin, user_profile);
evaluator.local_evaluator.materialize_hidden_fields() derives intent_card /
behavior from the catalog product at eval time, same as it does for the
public set.

Usage::

    python3 scripts/generate_paraphrase_dataset.py
    python3 scripts/generate_paraphrase_dataset.py --output data/paraphrase_set.jsonl --seed 20260831
    python3 scripts/paraphrase_stress.py --dataset data/paraphrase_set.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

SCENARIO_COUNTS = {
    "buying": 80,
    "browsing": 80,
    "intent_override": 30,
    "boundary": 10,
}

PREFERENCE_TAG_POOL = (
    "fit", "comfort", "durability", "style", "warmth", "weather",
    "performance", "material", "general shopping",
)

RATING_STYLE_WEIGHTS = (
    ("usually positive", 0.67, (4.0, 5.0)),
    ("critical", 0.225, (1.0, 2.0)),
    ("mixed", 0.105, (2.5, 3.5)),
)

DIFFICULTY_WEIGHTS = (("easy", 0.40), ("medium", 0.45), ("hard", 0.15))

EXCLUDE_CATEGORY_WORDS = {"clothing", "clothing shoes & jewelry", "clothing, shoes & jewelry"}


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def used_parent_asins(public_set_path: Path) -> set[str]:
    if not public_set_path.exists():
        return set()
    return {
        str(json.loads(line)["ground_truth"]["parent_asin"])
        for line in public_set_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def usable_products(catalog_path: Path, exclude: set[str]) -> list[dict]:
    products = []
    with catalog_path.open(encoding="utf-8") as handle:
        for line in handle:
            product = json.loads(line)
            parent_asin = str(product.get("parent_asin") or "")
            if not parent_asin or parent_asin in exclude:
                continue
            title = product.get("title")
            categories = product.get("categories") or []
            # Need enough signal for intent_card() to build real constraints,
            # and a real category for coarse_category() to name in prompts.
            if not title or not categories:
                continue
            if not (product.get("features") or product.get("details")):
                continue
            products.append(product)
    return products


def weighted_choice(rng: random.Random, weights: tuple) -> str:
    labels = [w[0] for w in weights]
    probs = [w[1] for w in weights]
    return rng.choices(labels, weights=probs, k=1)[0]


def coarse_category_bucket(categories: list[str]) -> str:
    for value in categories:
        for part in re.split(r",", value):
            part = part.strip().lower()
            if part and part not in EXCLUDE_CATEGORY_WORDS:
                return "clothing"
    return "clothing"


def build_user_profile(rng: random.Random) -> dict:
    labels = [w[0] for w in RATING_STYLE_WEIGHTS]
    probs = [w[1] for w in RATING_STYLE_WEIGHTS]
    ranges = {w[0]: w[2] for w in RATING_STYLE_WEIGHTS}
    rating_style = rng.choices(labels, weights=probs, k=1)[0]
    average_prior_rating = round(rng.uniform(*ranges[rating_style]), 1)
    tag_count = rng.choice([2, 3, 3, 4])
    tags = rng.sample(PREFERENCE_TAG_POOL, k=min(tag_count, len(PREFERENCE_TAG_POOL)))
    summary = f"Prior purchases emphasize {', '.join(tags)}; ratings are {rating_style}."
    return {
        "average_prior_rating": average_prior_rating,
        "preference_tags": tags,
        "purchase_frequency": "3-4 prior purchases",
        "rating_style": rating_style,
        "summary": summary,
    }


def build_sessions(products: list[dict], rng: random.Random) -> list[dict]:
    scenario_pool: list[str] = []
    for scenario, count in SCENARIO_COUNTS.items():
        scenario_pool.extend([scenario] * count)
    rng.shuffle(scenario_pool)

    total = len(scenario_pool)
    if len(products) < total:
        raise SystemExit(
            f"only {len(products)} usable catalog products available, need {total}"
        )
    chosen = rng.sample(products, k=total)

    sessions = []
    for index, (scenario, product) in enumerate(zip(scenario_pool, chosen), start=1):
        difficulty = weighted_choice(rng, DIFFICULTY_WEIGHTS)
        sessions.append({
            "category_bucket": coarse_category_bucket(product.get("categories") or []),
            "difficulty_bucket": difficulty,
            "ground_truth": {"parent_asin": str(product["parent_asin"])},
            "sample_id": f"paraphrase_{index:04d}",
            "scenario_type": scenario,
            "user_profile": build_user_profile(rng),
        })
    return sessions


def generate(
    catalog: str | Path,
    public_set: str | Path,
    output: str | Path,
    seed: int | None = None,
) -> list[dict]:
    """Build and write a fresh dataset; returns the sessions written."""
    rng = random.Random(seed) if seed is not None else random.Random()

    exclude = used_parent_asins(Path(public_set))
    products = usable_products(Path(catalog), exclude)
    sessions = build_sessions(products, rng)

    output_path = Path(output)
    with output_path.open("w", encoding="utf-8") as handle:
        for session in sessions:
            handle.write(json.dumps(session) + "\n")

    print(f"wrote {len(sessions)} sessions to {output_path}")
    print(f"excluded {len(exclude)} parent_asins already used in {public_set}")
    return sessions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", default=str(REPO_ROOT / "data" / "catalog.jsonl"))
    parser.add_argument("--public-set", default=str(REPO_ROOT / "data" / "public_set.jsonl"),
                        help="excluded so the generated targets are unseen, not copied")
    parser.add_argument("--output", default=str(REPO_ROOT / "data" / "paraphrase_set.jsonl"))
    parser.add_argument("--seed", type=int, default=None,
                        help="omit for a fresh, non-reproducible draw each run")
    args = parser.parse_args()
    generate(args.catalog, args.public_set, args.output, args.seed)


if __name__ == "__main__":
    main()
