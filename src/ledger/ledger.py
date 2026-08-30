from __future__ import annotations

import copy
import json
import threading
from contextlib import contextmanager
from typing import Generator


ALLOWED_ATTRIBUTES = {
    "category", "material", "color", "size", "style", "brand",
    "budget", "feature", "use_case", "other",
}

# Stable priority order so the clarifier always asks the most useful attribute first.
ATTRIBUTE_PRIORITY = (
    "category", "use_case", "material", "color", "size",
    "style", "brand", "budget", "feature", "other",
)


def _empty_entry(session_id: str, user_profile: dict) -> dict:
    return {
        "session_id": session_id,
        "user_profile": copy.deepcopy(user_profile),
        "turn": 0,
        "intent": None,
        "constraints": {},
        "user_preference": [],
        "asked_attributes": [],
        "search_key": {},
    }


class LedgerService:
    """Thread-safe in-memory session ledger keyed by session_id."""

    def __init__(self) -> None:
        self._store: dict[str, dict] = {}
        self._locks: dict[str, threading.RLock] = {}
        self._global_lock = threading.Lock()

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def create(self, session_id: str, user_profile: dict) -> None:
        with self._global_lock:
            self._locks[session_id] = threading.RLock()
            self._store[session_id] = _empty_entry(session_id, user_profile)

    def read(self, session_id: str) -> dict:
        with self._session_lock(session_id):
            return copy.deepcopy(self._store[session_id])

    def delete(self, session_id: str) -> None:
        with self._global_lock:
            self._store.pop(session_id, None)
            self._locks.pop(session_id, None)

    def exists(self, session_id: str) -> bool:
        with self._global_lock:
            return session_id in self._store

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    @contextmanager
    def managed_session(self, session_id: str, user_profile: dict) -> Generator[None, None, None]:
        """Create a session, yield control, then delete it automatically."""
        self.create(session_id, user_profile)
        try:
            yield
        finally:
            self.delete(session_id)

    @contextmanager
    def session(self, session_id: str) -> Generator[dict, None, None]:
        """Yield a mutable snapshot and write it back atomically on clean exit."""
        lock = self._session_lock(session_id)
        with lock:
            snapshot = copy.deepcopy(self._store[session_id])
            try:
                yield snapshot
            except Exception:
                raise
            else:
                self._store[session_id] = copy.deepcopy(snapshot)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def increment_turn(self, session_id: str) -> None:
        with self._session_lock(session_id):
            self._store[session_id]["turn"] += 1

    def set_intent(self, session_id: str, intent: str) -> None:
        with self._session_lock(session_id):
            self._store[session_id]["intent"] = intent

    def add_constraint(self, session_id: str, attribute: str, value: str) -> None:
        """Append a value to an attribute (e.g. user adds another color)."""
        if attribute not in ALLOWED_ATTRIBUTES:
            raise ValueError(f"Invalid attribute '{attribute}'. Must be one of {ALLOWED_ATTRIBUTES}")
        with self._session_lock(session_id):
            existing = self._store[session_id]["constraints"].get(attribute, [])
            if value not in existing:
                existing.append(value)
            self._store[session_id]["constraints"][attribute] = existing

    def set_constraint(self, session_id: str, attribute: str, value: str) -> None:
        """Overwrite an attribute entirely (e.g. user changes their mind)."""
        if attribute not in ALLOWED_ATTRIBUTES:
            raise ValueError(f"Invalid attribute '{attribute}'. Must be one of {ALLOWED_ATTRIBUTES}")
        with self._session_lock(session_id):
            self._store[session_id]["constraints"][attribute] = [value]

    def clear_constraints(self, session_id: str) -> None:
        """Wipe constraints and user preferences on intent override."""
        with self._session_lock(session_id):
            self._store[session_id]["constraints"].clear()
            self._store[session_id]["user_preference"].clear()

    def add_user_preference(self, session_id: str, preference: str) -> None:
        with self._session_lock(session_id):
            prefs = self._store[session_id]["user_preference"]
            if preference not in prefs:
                prefs.append(preference)

    def mark_attribute_asked(self, session_id: str, attribute: str) -> None:
        if attribute not in ALLOWED_ATTRIBUTES:
            raise ValueError(f"Invalid attribute '{attribute}'. Must be one of {ALLOWED_ATTRIBUTES}")
        with self._session_lock(session_id):
            asked = self._store[session_id]["asked_attributes"]
            if attribute not in asked:
                asked.append(attribute)

    def set_search_key(self, session_id: str, search_key: dict[str, list]) -> None:
        with self._session_lock(session_id):
            self._store[session_id]["search_key"] = search_key

    def next_unasked_attribute(self, session_id: str) -> str | None:
        with self._session_lock(session_id):
            asked = set(self._store[session_id]["asked_attributes"])
            constraints = set(self._store[session_id]["constraints"].keys())
        covered = asked | constraints
        for attr in ATTRIBUTE_PRIORITY:
            if attr not in covered:
                return attr
        return None

    # ------------------------------------------------------------------
    # Debug
    # ------------------------------------------------------------------

    def dump(self, session_id: str) -> None:
        print(json.dumps(self.read(session_id), indent=2))

    def __repr__(self) -> str:
        with self._global_lock:
            sessions = list(self._store.keys())
        return f"LedgerService(sessions={sessions})"

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _session_lock(self, session_id: str) -> threading.RLock:
        with self._global_lock:
            if session_id not in self._locks:
                raise KeyError(f"Session '{session_id}' not found in ledger.")
            return self._locks[session_id]
