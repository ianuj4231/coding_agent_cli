"""Agent tools for user-controlled long-term memory."""

from __future__ import annotations

from langchain.tools import ToolRuntime, tool

from ia_claude.agent.context import AgentContext
from ia_claude.memory.long_term import (
    LongTermMemoryWriteError,
    save_long_term_memory,
)


@tool
async def remember_user_information(
    memory: str,
    runtime: ToolRuntime[AgentContext],
) -> str:
    """Save one durable fact about the user for future conversations.

    Use this only for stable preferences, long-term goals, important
    achievements, ongoing projects, or explicit future instructions. Never
    use it for secrets, credentials, temporary details, or uncertain guesses.

    Args:
        memory: One clear, self-contained fact to remember about the user.
    """
    if not runtime.context.memory_enabled:
        return "Long-term memory is disabled; nothing was saved."

    if runtime.store is None:
        return "Long-term memory is temporarily unavailable; nothing was saved."

    try:
        memory_id = await save_long_term_memory(
            runtime.store,
            runtime.context.user_id,
            memory,
        )
    except ValueError:
        return "The memory was empty; nothing was saved."
    except LongTermMemoryWriteError:
        return "Long-term memory could not be saved."

    return f"Long-term memory saved. ID: {memory_id}"
