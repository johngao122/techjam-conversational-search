"""Isolated follow-up-question inspection under contradiction + paraphrase
stress, proving the clarification component never touches product search.

Deliberately imports NOTHING from src.reranker, src.retrieval, src.catalog,
or src.ledger.ledger (LedgerService/search-key building) -- only:
  - src.intent_router          (message classification/extraction; no retrieval)
  - src.confidence.session_ledger.SessionLedger + policy.always_ask/next_unasked_topic
  - src.output.followup        (message phrasing)

always_ask(ledger) and next_unasked_topic(ledger, known_attrs) -- the two
functions that actually drive the shipped follow-up path -- take no
RankResult and call into no search code at all; this script's import list is
the structural proof, not just a claim (see PR #5).

Two scenarios, replaying the same customer-message constructions as
scripts/contradiction_stress.py (turn-1 decoy, override turn) and
scripts/paraphrase_stress.py (reworded templates, markers dropped) -- but
inspecting the follow-up question TEXT each turn produces, not retrieval
ranking (those scripts test the reranker/constraint-memory side; this one
tests the clarification side).

Run from the repo root:
    python3 scripts/followup_isolation_check.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.confidence.policy import always_ask, missing_topics, next_unasked_topic  # noqa: E402
from src.confidence.session_ledger import SessionLedger  # noqa: E402
from src.intent_router import detect_scenario, extract_attributes  # noqa: E402
from src.output.followup import (  # noqa: E402
    FollowUpContext,
    build_all_missing_ask_message,
    build_ask_message,
    build_recommend_message,
)


def _run_scenario(name: str, turns: list[tuple[int, str]]) -> None:
    print(f"=== {name} ===")
    ledger = SessionLedger(session_id="isolation-check")
    known_attrs: set[str] = set()
    history: list[dict] = []

    for turn, message in turns:
        scenario = detect_scenario(message, history)
        new_attrs = extract_attributes(message)
        known_attrs.update(new_attrs.keys())
        history.append({"turn": turn, "role": "user", "content": message})

        ledger.observe(message, turn)
        for value in new_attrs.values():
            ledger.add_constraint(str(value))

        payload = always_ask(ledger)
        if payload.ask_attribute:
            ledger.note_ask(payload.ask_attribute)

        topic = next_unasked_topic(ledger, known_attrs=known_attrs) if payload.clarify else None
        if topic:
            ledger.note_ask(topic)

        context = FollowUpContext(
            scenario=scenario,
            n_constraints_known=ledger.n_constraints_known,
            exhausted=ledger.exhausted,
            turn=turn,
            override_seen=ledger.override_seen,
            topic=topic,
        )
        text = build_ask_message(context) if payload.clarify else build_recommend_message(context)

        print(f"  turn {turn}: customer says {message!r}")
        print(f"    scenario={scenario!r} clarify={payload.clarify} ask_attribute={payload.ask_attribute!r} topic={topic!r}")
        print(f"    Agent: {text}")
    print()


def _run_scenario_bundled(name: str, turns: list[tuple[int, str]]) -> None:
    """Same as _run_scenario, but shows the current live default -- the
    bundled missing_topics + build_all_missing_ask_message path -- instead
    of the older single-topic next_unasked_topic + build_ask_message."""
    print(f"=== {name} (bundled/live) ===")
    ledger = SessionLedger(session_id="isolation-check-bundled")
    known_attrs: set[str] = set()
    history: list[dict] = []

    for turn, message in turns:
        scenario = detect_scenario(message, history)
        new_attrs = extract_attributes(message)
        known_attrs.update(new_attrs.keys())
        history.append({"turn": turn, "role": "user", "content": message})

        ledger.observe(message, turn)
        for value in new_attrs.values():
            ledger.add_constraint(str(value))

        payload = always_ask(ledger)
        if payload.ask_attribute:
            ledger.note_ask(payload.ask_attribute)

        missing = missing_topics(known_attrs) if payload.clarify else []

        context = FollowUpContext(
            scenario=scenario,
            n_constraints_known=ledger.n_constraints_known,
            exhausted=ledger.exhausted,
            turn=turn,
            override_seen=ledger.override_seen,
            missing_attrs=tuple(missing),
        )
        text = build_all_missing_ask_message(context) if payload.clarify else build_recommend_message(context)

        print(f"  turn {turn}: customer says {message!r}")
        print(f"    scenario={scenario!r} clarify={payload.clarify} ask_attribute={payload.ask_attribute!r} missing={missing!r}")
        print(f"    Agent: {text}")
    print()


def main() -> None:
    # Mirrors scripts/contradiction_stress.py's turns exactly: a turn-1 decoy
    # constraint, then a real override at turn 3 that contradicts it.
    contradiction_turns = [
        (1, "I'm looking for Boots. A key requirement is: cotton."),
        (3, "Actually, ignore my earlier preference. What I need is: leather."),
    ]

    # Mirrors scripts/paraphrase_stress.py's _reword_opening/_reword_reply:
    # markers dropped, clauses reordered, synonym wrappers, joined with
    # "and also" instead of ";" -- the private-set risk the team flagged.
    paraphrase_turns = [
        (1, "Hey, I really need cotton and I'm after some boots today."),
        (2, "no strong feeling on color, your call"),
        (3, "hmm those aren't right, ask me something specific"),
        (4, "well leather and also black would be great"),
    ]

    _run_scenario("contradiction (turn-1 decoy -> override)", contradiction_turns)
    _run_scenario("paraphrase (reworded templates)", paraphrase_turns)
    _run_scenario_bundled("contradiction (turn-1 decoy -> override)", contradiction_turns)
    _run_scenario_bundled("paraphrase (reworded templates)", paraphrase_turns)


if __name__ == "__main__":
    main()
