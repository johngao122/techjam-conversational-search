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

import re
from dataclasses import dataclass
from pathlib import Path

from src.confidence import SessionLedger, popularity_top10, safe_decide
from src.confidence.policy import DEFAULT_THETA, exposure
from src.intent_router import build_search_key, detect_scenario, extract_attributes
from src.intent_router.constraint_memory import ConstraintMemory
from src.ledger.ledger import LedgerService
from src.ledger.ollama_summarizer import OllamaSummarizer
from src.output import OutputFormatter
from src.reranker import build_reranker, default_query
from src.reranker.rank import retrieval_mode
from src.retrieval.retrieval import Retriever


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
        # Eager build: FTS5 index + catalog load happen once, up front.
        self._reranker = build_reranker(self._catalog_path)
        # Vector-capable retriever sharing the reranker's already-built in-memory
        # catalog (no second load). Enables BM25 + semantic retrieval as two
        # independent parallel sources; retrieve_vector degrades to [] when the
        # embedding cache / endpoint / numpy are unavailable.
        self._retriever = Retriever.with_vectors(self._reranker.catalog)
        self._popularity = popularity_top10(self._catalog_path)
        self._mode = retrieval_mode()
        # Confidence state, keyed by session_id (parallel to the structured ledger).
        self._sessions: dict[str, SessionLedger] = {}
        # Turn-1 opening message per session -- carries the verbatim coarse
        # category the bucket resolver keys off, disclosed once and reused.
        self._openings: dict[str, str] = {}
        # Cumulative verbatim constraint memory (evict-on-value-conflict).
        self._memory: dict[str, ConstraintMemory] = {}
        # LLM-based conversation summarizer using local Ollama (phi3:mini)
        self._summarizer = OllamaSummarizer(model="phi3:mini")
        # Cross-session summary storage: keyed by user_id from user_profile.
        self._summaries: dict[str, dict] = {}

    def reset(self, session_id: str, user_profile: dict) -> None:
        self._ledger.create(session_id, user_profile or {})
        self._sessions[session_id] = SessionLedger(session_id=session_id)
        self._openings.pop(session_id, None)
        self._memory[session_id] = ConstraintMemory()
        # Inject prior summary if this user_id has been seen before.
        user_id = (user_profile or {}).get("user_id")
        if user_id and user_id in self._summaries:
            self._ledger.set_conversation_summary(session_id, self._summaries[user_id])

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
        session = self._ledger.read(session_id)
        scenario = detect_scenario(user_message, session.get("history", []))

        if scenario == "intent_override":
            self._ledger.set_intent(session_id, "buying")
        elif scenario == "boundary":
            asked = self._ledger.read(session_id)["asked_attributes"]
            if asked:
                last_asked = asked[-1]
                # Remove the boundary attribute so it isn't searched.
                with self._ledger.session(session_id) as s:
                    s["constraints"].pop(last_asked, None)
            self._ledger.set_intent(session_id, "boundary")
        else:
            self._ledger.set_intent(session_id, scenario)

        # -- 2. Attribute Extraction ------------------------------------------
        new_attrs = extract_attributes(user_message)
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

        # -- 3b. Update conversation summary ---------------------------------
        session = self._ledger.read(session_id)
        current_summary = session.get("conversation_summary", {})
        summary = self._summarizer.summarize(
            history=session.get("history", []),
            constraints=session.get("constraints", {}),
            intent=session.get("intent"),
            session_summary=current_summary if current_summary else None,
        )
        self._ledger.set_conversation_summary(session_id, summary)
        # Store in cross-session cache keyed by user_id.
        user_id = session.get("user_profile", {}).get("user_id")
        if user_id:
            self._summaries[user_id] = summary

        # -- 3c. Update search key from summary --------------------------------
        summary_search_key = self._build_summary_search_key(summary)
        self._ledger.set_search_key(session_id, summary_search_key)

        # -- 4. Update confidence ledger --------------------------------------
        # observe() reads the raw reply for override / boundary / exhaustion.
        conf_ledger.observe(user_message, turn)
        session = self._ledger.read(session_id)
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
        # Union-join this turn's parser-derived key with the previous turn's so
        # the key is monotonic across turns: text attributes accumulate (union),
        # numeric range filters (price/rating) are updated to the latest value.
        session = self._ledger.read(session_id)
        current_key = build_search_key(session)
        previous_key = session.get("search_key") or {}
        search_key = self._union_search_key(previous_key, current_key)
        self._ledger.set_search_key(session_id, search_key)
        query = default_query(constraints, user_message)

        # -- 5b. Parallel retrieval sources (independent; NOT fallbacks) ------
        # Both run every turn. Results are intentionally left UNUSED for now --
        # this wires the parallel paths so they execute and are ready for a
        # future fusion/rerank step. The bucket pipeline below still drives the
        # emitted top-10 (unchanged behaviour). Note the rank_bucket rung-3 BM25
        # fallback is deliberately not relied upon; this is the first-class BM25.
        bm25_results = self._retriever.retrieve_bm25(search_key, top_k=top_k)
        # TODO: add a query-processing function here to turn the session/user
        # message into an optimal vector query string; for now pass the raw
        # user message straight through.
        vector_results = self._retriever.retrieve_vector(user_message, top_k=top_k)
        # del bm25_results, vector_results  # intentionally unused for now

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
        payload, recommendations = safe_decide(
            rank_fn,
            conf_ledger,
            self._popularity,
            theta=self._theta,
            policy="always_ask",
        )
        recommendations = bm25_results
        # recommendations = vector_results
        if payload.ask_attribute:
            conf_ledger.note_ask(payload.ask_attribute)

        # -- 7. Exposure gate + format ----------------------------------------
        # Reveal one candidate on turns 1-2, the full list from turn 3 (or once
        # the card is drained / on the final turn). Legacy mode keeps the old
        # unconditional full-list behaviour.
        if self._mode == "legacy":
            reveal = top_k
        else:
            reveal = exposure(turn, conf_ledger.exhausted, top_k)
        return self._formatter.format(payload, recommendations[:reveal])

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _union_search_key(previous: dict[str, list], current: dict[str, list]) -> dict[str, list]:
        """Merge the previous turn's search key with the current one.

        Text fields (list-of-strings) are *unioned* -- values accumulate across
        turns, de-duplicated, preserving first-seen order. Numeric range filters
        (``price``/``rating`` etc., list-of-``{op: value}``) are *updated*: the
        current turn's value replaces the previous one (latest budget wins),
        rather than unioning bounds which could impose contradictory limits.

        The result is monotonic for text (never loses a previously-known value)
        while staying correct for numeric constraints.
        """
        # Local import: reuse the retriever's numeric-shape classifier so this
        # stays in lock-step with how retrieve_bm25 interprets the key.
        from src.retrieval.retrieval import _is_numeric_filter

        merged: dict[str, list] = {}
        for field in (*previous.keys(), *current.keys()):
            if field in merged:
                continue
            cur_val = current.get(field)
            prev_val = previous.get(field)

            # Numeric range filter -> update to current if present, else keep prev.
            if _is_numeric_filter(cur_val) or _is_numeric_filter(prev_val):
                merged[field] = cur_val if cur_val is not None else prev_val
                continue

            # Text field -> union of value lists, de-duplicated, order-preserving.
            values: list = []
            for source in (prev_val, cur_val):
                if isinstance(source, list):
                    for v in source:
                        if v not in values:
                            values.append(v)
            merged[field] = values
        return merged

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

    @staticmethod
    def _build_summary_search_key(summary: dict) -> dict:
        """Build a search key from the conversation summary using LLM.

        Uses the summary text to create a simple search string like "blue umbrella".
        """
        search_key: dict = {}
        summary_text = summary.get("summary", "")

        # Use LLM to extract the search key string from the summary
        search_key_string = Agent._extract_search_key_with_llm(summary_text)
        search_key["_string"] = search_key_string

        return search_key

    @staticmethod
    def _extract_search_key_with_llm(summary_text: str) -> str:
        """Use LLM to extract a simple search string from the summary.

        Returns something like "blue umbrella" or "red leather boots".
        """
        try:
            import ollama

            prompt = f"""Extract a simple search string from this customer summary.

Summary: {summary_text}

Return ONLY a simple search string with attributes and product, like:
- "blue umbrella"
- "red boots"
- "leather shoes"
- "black iphone"

Do NOT include extra words or explanations. Just the search string:"""

            response = ollama.generate(
                model="phi3:mini",
                prompt=prompt,
                stream=False,
                options={"temperature": 0.1, "num_predict": 30}
            )

            search_string = (response.get("response", "") or "").strip()

            # Clean up - remove quotes and extra punctuation
            search_string = search_string.strip("\"'.")

            if search_string:
                return search_string
            else:
                return summary_text  # Fallback to summary if extraction fails

        except Exception as e:
            # Fallback: just return the summary text
            return summary_text

    @staticmethod
    def _extract_product_from_summary(summary_text: str) -> str | None:
        """Extract the main product/item the customer wants from summary text.

        Uses simple keyword matching to identify product names.
        """
        if not summary_text:
            return None

        # Common product keywords and their canonical forms
        product_keywords = {
            "boots": "boots",
            "shoes": "shoes",
            "umbrella": "umbrella",
            "umbrellas": "umbrella",
            "phone": "phone",
            "iphone": "iphone",
            "tv": "tv",
            "television": "tv",
            "shirt": "shirt",
            "pants": "pants",
            "jacket": "jacket",
            "coat": "coat",
            "dress": "dress",
            "skirt": "skirt",
            "sweater": "sweater",
            "shirt": "shirt",
            "hat": "hat",
            "cap": "cap",
            "bag": "bag",
            "purse": "purse",
            "wallet": "wallet",
            "watch": "watch",
            "glasses": "glasses",
            "sunglasses": "sunglasses",
            "belt": "belt",
            "scarf": "scarf",
            "gloves": "gloves",
            "socks": "socks",
            "lingerie": "lingerie",
        }

        for keyword, product in product_keywords.items():
            if keyword in summary_text:
                return product

        return None
