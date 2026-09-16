"""Trusted application context supplied to each agent invocation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AgentContext:
    """Identity and LTM permission resolved by the application for one query."""

    user_id: str
    memory_enabled: bool
    long_term_memories: tuple[str, ...] = ()
    long_term_memories: tuple[str, ...] = ()
