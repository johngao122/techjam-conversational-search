"""Full pipeline agent.

Wires the components into the flow described in ``docs/diagrams/architecture.md``::

    Intent Router -> Ledger -> Retrieval+Reranker -> Confidence (decision) -> Output

Every turn returns a top-10 recommendation list; the confidence component (the
decision gate) only decides whether to *also* attach a clarifying question.
Retrieval/rerank failures fall back to a popularity ordering so ``respond``
never raises and always emits recommendations.

The :class:`~src.ledger.ledger.LedgerService` tracks structured
constraints/turn/history; a parallel :class:`~src.confidence.session_ledger.SessionLedger`
tracks the exhaustion/override signals the confidence policy consumes.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

from src.confidence import SessionLedger, popularity_top10, safe_decide
from src.confidence.policy import DEFAULT_THETA, exposure, missing_topics
from src.intent_router import attributes_of, build_search_key, parse_message, warm_parser
from src.intent_router.constraint_memory import ConstraintMemory
from src.ledger.ledger import LedgerService
from src.output import FollowUpContext, OutputFormatter
from src.reranker import build_reranker, default_query
from src.reranker.rank import retrieval_mode


@dataclass
class PriceConstraint:
    operator: str  # "<" | "<=" | ">" | ">=" | "~"
    amount: float


_PRICE_RE = re.compile(
    r"(?:"
    r"(?P<op1>under|less\s+than|below|cheaper\s+than|max|maximum|no\s+more\s+than|at\s+most)\s*\$?(?P<amt1>[\d,]+(?:\.\d+)?)"
    r"|(?P<op2>over|more\s+than|above|at\s+least|minimum|min)\s*\$?(?P<amt2>[\d,]+(?:\.\d+)?)"
    r"|(?P<op3>around|about|approximately|budget\s+(?:is|of)?|~)\s*\$?(?P<amt3>[\d,]+(?:\.\d+)?)"
    r"|\$?(?P<amt4>[\d,]+(?:\.\d+)?)\s*(?P<op4>or\s+less|or\s+under|and\s+under|and\s+below|-)"
    r"|\$(?P<amt5>[\d,]+(?:\.\d+)?)"
    r")",
    re.IGNORECASE,
)


def _parse_price_constraint(text: str) -> PriceConstraint | None:
    m = _PRICE_RE.search(text)
    if not m:
        return None

    def _clean(s: str | None) -> float | None:
        return float(s.replace(",", "")) if s else None

    if m.group("op1"):
        return PriceConstraint("<", _clean(m.group("amt1")))
    if m.group("op2"):
        op_word = re.sub(r"\s+", " ", m.group("op2").lower().strip())
        op = ">=" if op_word in ("at least", "minimum", "min") else ">"
        return PriceConstraint(op, _clean(m.group("amt2")))
    if m.group("op3"):
        return PriceConstraint("~", _clean(m.group("amt3")))
    if m.group("amt4"):
        return PriceConstraint("<=", _clean(m.group("amt4")))
    if m.group("amt5"):
        return PriceConstraint("~", _clean(m.group("amt5")))
    return None


def ask_policy() -> str:
    """Ship default is ``always_ask`` (the measured 0.968-TechScore champion,
    see docs/project_description.md). ``ASK_POLICY=attribute_cycle`` swaps in
    the specific-attribute-per-turn, never-repeat policy for A/B measurement;
    see ``decide_specific_attribute``'s docstring for the tradeoff."""
    return os.environ.get("ASK_POLICY", "always_ask").strip() or "always_ask"


class Agent:
    """Full pipeline agent: Intent -> Ledger -> Retrieval/Rerank -> Confidence -> Output."""

    def __init__(
        self,
        catalog_path: str | Path = "data/catalog.jsonl",
        theta: float = DEFAULT_THETA,
    ) -> None:
        self._catalog_path = str(catalog_path)
        self._theta = theta
        self._ledger = LedgerService()
        self._formatter = OutputFormatter()
        # Eager build: FTS5 index + catalog load happen once, up front. The
        # router's vocab scan is warmed here too -- left lazy it fired inside
        # the first respond() call and made turn 1 a multi-second outlier.
        self._reranker = build_reranker(self._catalog_path)
        self._popularity = popularity_top10(self._catalog_path)
        warm_parser(self._catalog_path)
        self._mode = retrieval_mode()
        self._ask_policy = ask_policy()
        # Confidence state, keyed by session_id (parallel to the structured ledger).
        self._sessions: dict[str, SessionLedger] = {}
        # Turn-1 opening message per session -- carries the verbatim coarse
        # category the bucket resolver keys off, disclosed once and reused.
        self._openings: dict[str, str] = {}
        # Cumulative verbatim constraint memory (evict-on-value-conflict).
        self._memory: dict[str, ConstraintMemory] = {}

    def reset(self, session_id: str, user_profile: dict) -> None:
        self._ledger.create(session_id, user_profile or {})
        self._sessions[session_id] = SessionLedger(session_id=session_id)
        self._openings.pop(session_id, None)
        self._memory[session_id] = ConstraintMemory()

    def respond(
        self,
        session_id: str,
        user_message: str,
        turn: int,
        top_k: int,
    ) -> dict:
        self._ledger.increment_turn(session_id)
        conf_ledger = self._sessions.setdefault(
            session_id, SessionLedger(session_id=session_id)
        )
        # The first message of the session carries the coarse category; keep it
        # for the bucket resolver, which needs the opening line, not later turns.
        opening = self._openings.setdefault(session_id, user_message)
        # Accumulate verbatim disclosed constraints (with value-conflict
        # supersession) before ranking this turn.
        memory = self._memory.setdefault(session_id, ConstraintMemory())
        memory.add_message(user_message, turn)

        # -- 1. Intent Router --------------------------------------------------
        session = self._ledger.read_ref(session_id)
        # One parse per turn: the intent and the attributes both come off it.
        parsed = parse_message(user_message)
        scenario = parsed.intent

        if scenario == "intent_override":
            self._ledger.set_intent(session_id, "buying")
        elif scenario == "boundary":
            asked = self._ledger.read_ref(session_id)["asked_attributes"]
            if asked:
                last_asked = asked[-1]
                # Remove the boundary attribute so it isn't searched.
                with self._ledger.session(session_id) as s:
                    s["constraints"].pop(last_asked, None)
            self._ledger.set_intent(session_id, "boundary")
        else:
            self._ledger.set_intent(session_id, scenario)

        # -- 2. Attribute Extraction ------------------------------------------
        new_attrs = attributes_of(parsed)
        price = _parse_price_constraint(user_message)

        for attr, value in new_attrs.items():
            self._ledger.set_constraint(session_id, attr, value)

        if price:
            with self._ledger.session(session_id) as s:
                s["price_constraint"] = {"operator": price.operator, "amount": price.amount}

        # -- 3. Update history -------------------------------------------------
        with self._ledger.session(session_id) as s:
            s.setdefault("history", []).append(
                {"turn": turn, "role": "user", "content": user_message}
            )

        # -- 4. Update confidence ledger --------------------------------------
        # observe() reads the raw reply for override / boundary / exhaustion.
        conf_ledger.observe(user_message, turn)
        session = self._ledger.read_ref(session_id)
        constraints = self._collect_constraints(session)
        # `category` is a search-scoping signal pulled from the (possibly
        # vague) opening line, not a disclosed answer to a clarifying
        # question -- it must not by itself satisfy the confidence gate's
        # zero-info forced-clarify check. Retrieval/coverage still use the
        # unfiltered `constraints` list below.
        disclosed_constraints = self._collect_constraints(session, exclude_attrs={"category"})
        added_new = False
        for value in disclosed_constraints:
            if conf_ledger.add_constraint(value):
                added_new = True
        if added_new:
            conf_ledger.reset_progress()

        # -- 5. Build search key + query --------------------------------------
        session = self._ledger.read_ref(session_id)
        if session.get("search_key"):
            search_key = session["search_key"]
        else:
            search_key = build_search_key(session)
            self._ledger.set_search_key(session_id, search_key)
        query = default_query(constraints, user_message)

        # -- 6. Retrieval + Rerank + Decision (never raises) ------------------
        if self._mode == "legacy":
            rank_fn = lambda: self._reranker.rank(query, constraints, top_k=top_k)
        else:
            # Bucket mode ranks against the verbatim constraint memory, not the
            # taxonomy-routed strings -- the disclosed strings are literal
            # slices of the target's metadata and match exactly within a bucket.
            verbatim = memory.constraints
            transcript = " ".join(
                str(h.get("content", ""))
                for h in session.get("history", [])
                if h.get("role") == "user"
            )
            rank_fn = lambda: self._reranker.rank_bucket(
                opening, verbatim, top_k=top_k, transcript=transcript
            )
        known_attrs = set(session.get("constraints", {}).keys())
        payload, recommendations = safe_decide(
            rank_fn,
            conf_ledger,
            self._popularity,
            theta=self._theta,
            policy=self._ask_policy,
            known_attrs=known_attrs,
        )
        if payload.ask_attribute:
            conf_ledger.note_ask(payload.ask_attribute)

        # Message-phrasing: every attribute not yet disclosed, bundled into
        # one question (see missing_topics's docstring). Independent of
        # payload.ask_attribute (the contract field, which stays "other"
        # under always_ask). Recomputed fresh from known_attrs each turn --
        # no per-session state, so a missing attribute keeps being asked
        # about until it's actually known.
        #
        # known_attrs_for_missing (not known_attrs itself, to leave
        # safe_decide/decide_specific_attribute's already-measured behaviour
        # untouched): also counts budget as known when price_constraint is
        # set. A dollar-sign-less disclosure ("under 50") bypasses extract_
        # attributes()'s own budget regex (which requires a literal "$"),
        # but _parse_price_constraint already understands it and the search
        # layer already uses it -- the follow-up question shouldn't keep
        # asking about something already effectively provided.
        known_attrs_for_missing = set(known_attrs)
        if session.get("price_constraint"):
            known_attrs_for_missing.add("budget")
        missing = missing_topics(known_attrs_for_missing) if payload.clarify else []

        # -- 7. Exposure gate + format ----------------------------------------
        # Reveal one candidate on turns 1-2, the full list from turn 3 (or once
        # the card is drained / on the final turn). Legacy mode keeps the old
        # unconditional full-list behaviour.
        if self._mode == "legacy":
            reveal = top_k
        else:
            reveal = exposure(turn, conf_ledger.exhausted, top_k)
        followup_context = FollowUpContext(
            scenario=scenario,
            n_constraints_known=conf_ledger.n_constraints_known,
            exhausted=conf_ledger.exhausted,
            turn=turn,
            override_seen=conf_ledger.override_seen,
            missing_attrs=tuple(missing),
        )
        return self._formatter.format(payload, recommendations[:reveal], context=followup_context)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _collect_constraints(session: dict, exclude_attrs: set[str] | None = None) -> list[str]:
        """Flatten ledger constraints (+ budget) into coverage constraint strings."""
        exclude_attrs = exclude_attrs or set()
        constraints: list[str] = []
        for attr, values in session.get("constraints", {}).items():
            if attr in exclude_attrs:
                continue
            for value in values:
                if attr == "color":
                    constraints.append(f"color: {value}")
                else:
                    constraints.append(str(value))
        price_c = session.get("price_constraint")
        if price_c:
            constraints.append(f"budget around ${price_c['amount']}")
        return constraints
