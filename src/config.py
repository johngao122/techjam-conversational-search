"""Single source of truth for the pipeline's swappable mechanisms.

Every alternative mechanism the agent can run under (which message parser,
which retrieval core, the clarification-ask policy, the exposure gate, the
override-supersession policy, IDF weighting, the confidence threshold) is a
field on :class:`AgentConfig`. ``Agent`` takes one of these; the individual
components still read their own ``os.environ`` var as a *fallback default* so
older ``RETRIEVAL_MODE=… python …`` invocations and ``scripts/ab_eval.py``
keep working unchanged.

Defaults here are exactly the shipped behaviour:
    parser=rule, retrieval=bucket, ask_policy=always_ask, exposure gate on,
    release turn 3, override=evict_on_conflict, IDF off, theta 0.5
"""

from __future__ import annotations

import os
from dataclasses import dataclass, fields

PARSERS = ("rule", "llm")
RETRIEVAL_MODES = ("bucket", "legacy")
ASK_POLICIES = ("always_ask", "attribute_cycle")
OVERRIDE_POLICIES = ("evict_on_conflict", "evict_all", "keep")


def _env_flag(name: str, default: bool) -> bool:
    """``"0"``/``"false"``/``"no"`` -> False, ``"1"``/``"true"``/``"yes"`` -> True."""
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


@dataclass(frozen=True)
class AgentConfig:
    parser: str = "rule"
    retrieval_mode: str = "bucket"
    ask_policy: str = "always_ask"
    exposure_gate: bool = True
    release_turn: int = 3
    override_policy: str = "evict_on_conflict"
    idf_weight: bool = False
    theta: float = 0.5

    def __post_init__(self) -> None:
        if self.parser not in PARSERS:
            raise ValueError(f"parser must be one of {PARSERS}, got {self.parser!r}")
        if self.retrieval_mode not in RETRIEVAL_MODES:
            raise ValueError(f"retrieval_mode must be one of {RETRIEVAL_MODES}, got {self.retrieval_mode!r}")
        if self.ask_policy not in ASK_POLICIES:
            raise ValueError(f"ask_policy must be one of {ASK_POLICIES}, got {self.ask_policy!r}")
        if self.override_policy not in OVERRIDE_POLICIES:
            raise ValueError(f"override_policy must be one of {OVERRIDE_POLICIES}, got {self.override_policy!r}")
        if self.release_turn < 1:
            raise ValueError(f"release_turn must be >= 1, got {self.release_turn}")

    @classmethod
    def from_env(cls) -> "AgentConfig":
        """Build from the same env vars the individual components read.

        Any var left unset falls back to that field's default, so an empty
        environment reproduces the shipped configuration exactly.
        """
        d = cls()  # defaults
        return cls(
            parser=os.environ.get("PARSER_MODE", d.parser).strip().lower() or d.parser,
            retrieval_mode=os.environ.get("RETRIEVAL_MODE", d.retrieval_mode).strip().lower() or d.retrieval_mode,
            ask_policy=os.environ.get("ASK_POLICY", d.ask_policy).strip() or d.ask_policy,
            exposure_gate=_env_flag("EXPOSURE_GATE", d.exposure_gate),
            release_turn=_env_int("RELEASE_TURN", d.release_turn),
            override_policy=os.environ.get("OVERRIDE_POLICY", d.override_policy).strip() or d.override_policy,
            idf_weight=os.environ.get("IDF_WEIGHT", "").strip() == "1",
            theta=_env_float("THETA", d.theta),
        )

    def replace(self, **overrides: object) -> "AgentConfig":
        """Copy with some fields overridden (dataclasses.replace, kept local
        so callers don't import dataclasses just for this)."""
        current = {f.name: getattr(self, f.name) for f in fields(self)}
        current.update({k: v for k, v in overrides.items() if v is not None})
        return AgentConfig(**current)  # type: ignore[arg-type]

    def banner(self) -> str:
        return (
            f"parser={self.parser} retrieval={self.retrieval_mode} "
            f"ask_policy={self.ask_policy} exposure_gate={'on' if self.exposure_gate else 'off'} "
            f"release_turn={self.release_turn} override={self.override_policy} "
            f"idf={'on' if self.idf_weight else 'off'} theta={self.theta}"
        )


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    return int(raw) if raw.lstrip("-").isdigit() else default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    try:
        return float(raw) if raw else default
    except ValueError:
        return default
