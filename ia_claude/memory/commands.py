"""Command handling for the persisted long-term-memory preference."""

from __future__ import annotations

import json

from langgraph.store.sqlite.aio import AsyncSqliteStore

from ia_claude.memory.long_term import (
    LongTermMemoryDeleteError,
    LongTermMemoryReadError,
    LongTermMemoryWriteError,
    delete_all_long_term_memories,
    delete_long_term_memory,
    list_long_term_memories,
    save_long_term_memory,
)
from ia_claude.memory.user_settings import (
    MemorySettingsError,
    is_memory_enabled,
    set_memory_enabled,
)


MEMORY_COMMAND_USAGE = "Usage: /memory <on|off|status>"
REMEMBER_COMMAND_USAGE = "Usage: /remember <text>"
FORGET_COMMAND_USAGE = "Usage: /forget <memory_id>"


def handle_memory_command(user_id: str, command: str) -> str:
    """Execute a ``/memory`` command and return a user-facing message."""
    parts = command.strip().lower().split()
    if len(parts) != 2 or parts[0] != "/memory":
        return MEMORY_COMMAND_USAGE

    action = parts[1]
    try:
        if action == "on":
            set_memory_enabled(user_id, True)
            return "Long-term memory enabled."
        if action == "off":
            set_memory_enabled(user_id, False)
            return "Long-term memory disabled."
        if action == "status":
            status = "enabled" if is_memory_enabled(user_id) else "disabled"
            return f"Long-term memory is {status}."
    except MemorySettingsError:
        return "Could not access the memory setting. Please try again."

    return MEMORY_COMMAND_USAGE


async def handle_show_ltm_command(
    store: AsyncSqliteStore | None,
    user_id: str,
) -> str:
    """Return a readable list of the current user's long-term memories."""
    if store is None:
        return "Long-term memory is temporarily unavailable."

    try:
        memories = await list_long_term_memories(store, user_id)
    except LongTermMemoryReadError:
        return "Could not read long-term memories. Please try again."

    if not memories:
        return "No long-term memories stored."

    lines = ["Long-term memories:"]
    for item in memories:
        namespace = tuple(item.namespace)
        sqlite_prefix = ".".join(namespace)
        stored_value = json.dumps(item.value, ensure_ascii=False, sort_keys=True)
        expires_at = getattr(item, "expires_at", None)
        ttl_minutes = getattr(item, "ttl_minutes", None)
        lines.extend(
            (
                "",
                f"Namespace: {namespace!r}",
                f"SQLite prefix: {sqlite_prefix}",
                f"User ID: {user_id}",
                f"Memory ID (key): {item.key}",
                f"Value: {stored_value}",
                f"Created: {item.created_at.isoformat()}",
                f"Updated: {item.updated_at.isoformat()}",
                f"Expires: {expires_at.isoformat() if expires_at else None}",
                f"TTL minutes: {ttl_minutes}",
            )
        )
    return "\n".join(lines)


async def handle_remember_command(
    store: AsyncSqliteStore | None,
    user_id: str,
    command: str,
) -> str:
    """Save one explicit user-provided long-term memory."""
    parts = command.strip().split(maxsplit=1)
    if len(parts) != 2 or parts[0].lower() != "/remember":
        return REMEMBER_COMMAND_USAGE

    if store is None:
        return "Long-term memory is temporarily unavailable."

    try:
        if not is_memory_enabled(user_id):
            return "Long-term memory is disabled. Use /memory on first."
    except MemorySettingsError:
        return "Could not verify the memory setting. Memory was not saved."

    try:
        memory_id = await save_long_term_memory(
            store,
            user_id,
            parts[1],
        )
    except ValueError:
        return REMEMBER_COMMAND_USAGE
    except LongTermMemoryWriteError:
        return "Could not save the long-term memory. Please try again."

    return f"Long-term memory saved. ID: {memory_id}"


async def handle_forget_command(
    store: AsyncSqliteStore | None,
    user_id: str,
    command: str,
) -> str:
    """Delete one explicit memory belonging to the current user."""
    parts = command.strip().split(maxsplit=1)
    if len(parts) != 2 or parts[0].lower() != "/forget":
        return FORGET_COMMAND_USAGE

    if store is None:
        return "Long-term memory is temporarily unavailable."

    try:
        deleted = await delete_long_term_memory(store, user_id, parts[1])
    except ValueError:
        return FORGET_COMMAND_USAGE
    except LongTermMemoryDeleteError:
        return "Could not delete the long-term memory. Please try again."

    if not deleted:
        return "Long-term memory not found."
    return "Long-term memory deleted."


async def handle_forget_all_command(
    store: AsyncSqliteStore | None,
    user_id: str,
) -> str:
    """Delete every long-term memory belonging to the current user."""
    if store is None:
        return "Long-term memory is temporarily unavailable."

    try:
        deleted_count = await delete_all_long_term_memories(store, user_id)
    except LongTermMemoryDeleteError:
        return "Could not delete long-term memories. Please try again."

    if deleted_count == 0:
        return "No long-term memories stored."
    return f"Deleted {deleted_count} long-term memory item(s)."








"""
Long explanation here.

In the current LangGraph abatch() API, there is no clearly defined universal maximum number of delete operations per call.
The practical limit comes from SQLite and the SQL generated by the installed AsyncSqliteStore implementation.
Our code submits:
await store.abatch(
    PutOp(namespace, item.key, None)
    for item in memories
)
Internally, LangGraph may group these delete operations into SQL statements. SQLite has a maximum number of bound SQL parameters, which varies by build:
Older/default SQLite builds often allow around 999 parameters.
Newer SQLite builds commonly allow up to 32,766 parameters.
The precise value should be checked in the installed SQLite runtime.
You can inspect your SQLite build with:
python -c "import sqlite3; db=sqlite3.connect(':memory:'); print([x for x in db.execute('PRAGMA compile_options') if 'MAX_VARIABLE_NUMBER' in x[0]]); db.close()"
However, relying on that maximum is unnecessary. A conservative production design would delete in batches of:
100 items per call
Flow:
fetch up to 100 at offset 0
→ batch-delete those 100
→ fetch up to 100 at offset 0
→ repeat until none remain
That keeps memory usage bounded and remains far below normal SQLite parameter limits.
So the short answer:
Documented LangGraph maximum: no fixed public limit
Recommended safe batch size: 100
Current implementation: all collected memories in one batch
For reliability, I would change /forget_all to delete 100 memories per batch.

"""