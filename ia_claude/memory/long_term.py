"""Shared foundations for cross-session, long-term memory."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4

from langgraph.store.sqlite.aio import AsyncSqliteStore
from langgraph.store.base import Item, PutOp

from ia_claude.config import config
from ia_claude.observability.logger import get_logger
from ia_claude.user_context import validate_user_id


LONG_TERM_MEMORY_NAMESPACE = "long_term_memories"
logger = get_logger(__name__)


class LongTermMemoryStoreError(RuntimeError):
    """Raised when the persistent long-term-memory Store cannot be opened."""


class LongTermMemoryWriteError(RuntimeError):
    """Raised when a long-term memory cannot be saved."""


class LongTermMemoryReadError(RuntimeError):
    """Raised when long-term memories cannot be read."""


class LongTermMemoryDeleteError(RuntimeError):
    """Raised when a long-term memory cannot be deleted."""


def get_long_term_memory_store_db_path() -> Path:
    """Return the application-owned SQLite Store path."""
    configured = config.get("memory", {}).get(
        "long_term_store_db_path",
        ".ia_claude_all_hidden/long_term_memory.sqlite",
    )
    return Path(configured)


@asynccontextmanager
async def open_long_term_memory_store() -> AsyncIterator[AsyncSqliteStore]:
    """Open and initialize the persistent LangGraph SQLite Store.

    The caller owns this context for its full usage lifetime. Store setup is
    idempotent, so opening an existing database is safe.
    """
    db_path = get_long_term_memory_store_db_path()

    try:
        db_path.parent.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        logger.exception("Could not prepare the long-term memory Store path")
        raise LongTermMemoryStoreError(
            "Could not prepare the long-term memory Store path"
        ) from exc

    store_context = AsyncSqliteStore.from_conn_string(str(db_path))
    store_opened = False
    try:
        store = await store_context.__aenter__()
        store_opened = True
        await store.setup()
    except Exception as exc:
        if store_opened:
            try:
                await store_context.__aexit__(type(exc), exc, exc.__traceback__)
            except Exception:
                logger.exception(
                    "Long-term memory Store cleanup after failed setup failed"
                )
        logger.exception("Long-term memory Store initialization failed")
        raise LongTermMemoryStoreError(
            "Could not initialize the long-term memory Store"
        ) from exc

    logger.info("Long-term memory Store ready at %s", db_path)
    try:
        yield store
    finally:
        try:
            await store_context.__aexit__(None, None, None)
        except Exception:
            logger.exception("Long-term memory Store shutdown failed")


def get_long_term_memory_namespace(user_id: str) -> tuple[str, str]:
    """Return the canonical LangGraph Store namespace for one user.

    Keeping namespace construction here ensures every future long-term-memory
    read and write uses the same validated, user-isolated location.
    """
    canonical_user_id = validate_user_id(user_id)
    return LONG_TERM_MEMORY_NAMESPACE, canonical_user_id
    
    # The comma creates a tuple containing two values:

async def save_long_term_memory(
    store: AsyncSqliteStore,
    user_id: str,
    memory: str,
) -> str:
    """Save one plain-text memory and return its generated Store key."""
    if not isinstance(memory, str) or not memory.strip():
        raise ValueError("memory must contain text")

    namespace = get_long_term_memory_namespace(user_id)
    memory_id = str(uuid4())

    try:
        await store.aput(
            namespace,
            memory_id,
            {"memory": memory.strip()},
        )
    except Exception as exc:
        logger.exception(
            "Could not save long-term memory: user_id=%s memory_id=%s",
            user_id,
            memory_id,
        )
        raise LongTermMemoryWriteError(
            "Could not save the long-term memory"
        ) from exc

    logger.info(
        "Long-term memory saved: user_id=%s memory_id=%s",
        user_id,
        memory_id,
    )
    return memory_id


async def list_long_term_memories(
    store: AsyncSqliteStore,
    user_id: str,
    limit: int = 50,
) -> list[Item]:
    """Return long-term memories from only the specified user's namespace."""
    if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
        raise ValueError("limit must be a positive integer")

    namespace = get_long_term_memory_namespace(user_id)
    try:
        return await store.asearch(namespace, limit=limit)
    except Exception as exc:
        logger.exception(
            "Could not list long-term memories: user_id=%s",
            user_id,
        )
        raise LongTermMemoryReadError(
            "Could not read long-term memories"
        ) from exc


async def delete_long_term_memory(
    store: AsyncSqliteStore,
    user_id: str,
    memory_id: str,
) -> bool:
    """Delete a memory from this user's namespace; return whether it existed."""
    if not isinstance(memory_id, str) or not memory_id.strip():
        raise ValueError("memory_id is required")

    namespace = get_long_term_memory_namespace(user_id)
    key = memory_id.strip()
    try:
        existing = await store.aget(namespace, key)
        if existing is None:
            return False
        await store.adelete(namespace, key)
    except Exception as exc:
        logger.exception(
            "Could not delete long-term memory: user_id=%s memory_id=%s",
            user_id,
            key,
        )
        raise LongTermMemoryDeleteError(
            "Could not delete the long-term memory"
        ) from exc

    logger.info(
        "Long-term memory deleted: user_id=%s memory_id=%s",
        user_id,
        key,
    )
    return True


async def delete_all_long_term_memories(
    store: AsyncSqliteStore,
    user_id: str,
) -> int:
    """Delete every memory in this user's exact namespace and return the count."""
    namespace = get_long_term_memory_namespace(user_id)

    try:
        memories: list[Item] = []
        offset = 0
        page_size = 100
        while True:
            page = await store.asearch(
                namespace,
                limit=page_size,
                offset=offset,
            )
            memories.extend(page)
            if len(page) < page_size:
                break
            offset += len(page)

        if not memories:
            return 0

        await store.abatch(
            PutOp(namespace, item.key, None)
            for item in memories
        )
    except Exception as exc:
        logger.exception(
            "Could not delete all long-term memories: user_id=%s",
            user_id,
        )
        raise LongTermMemoryDeleteError(
            "Could not delete all long-term memories"
        ) from exc

    logger.info(
        "All long-term memories deleted: user_id=%s count=%s",
        user_id,
        len(memories),
    )
    return len(memories)






# AsyncSqliteSaver  → short-term memory (STM)
# AsyncSqliteStore  → long-term memory (LTM)


# AsyncSqliteSaver stores LangGraph checkpoints for a particular conversation:
# thread_id = "session-123"



# Meaning of key
# The key uniquely identifies one memory inside a namespace.
# For example:
# key = "mem_a82f"
# A different memory receives a different key:
# key = "mem_b91c" key is like memory_id
# The true database identity is the combination:
# (prefix, key)


# Our proposed namespace is:
# (
#     "long_term_memories",
#     user_id,
# )
# Example:
# (
#     "long_term_memories",
#     "3b2eb43f-3a0e-41d7-82b7-cde6dc9357d8",
# )
# The SQLite Store converts that tuple into the prefix column by joining its parts with dots:
# long_term_memories.3b2eb43f-3a0e-41d7-82b7-cde6dc9357d8



# so is long_term_memories only one box for evryone?






# 1:02 PM




# so is long_term_memories only one box for evryone?

# Conceptually, long_term_memories is the shared outer box for the entire application.
# Inside it, every user has a separate private box:
# long_term_memories
# │
# ├── 3b2eb43f-...             ← user 1
# │   ├── memory-A
# │   └── memory-B
# │
# ├── 7c41a819-...             ← user 2
# │   ├── memory-C
# │   └── memory-D
# │
# └── f980321c-...             ← user 3
#     └── memory-E




# Yes—reliable long-term memory is generally a senior-level engineering feature.
# Calling store.put() and store.search() is straightforward. Building a trustworthy memory system around those calls is the difficult part.
# Senior-level concerns include:
# User isolation: no memory may leak across namespaces.
# Consent: disabled must mean no automatic reading or writing.
# Memory quality: distinguish durable facts from temporary conversation.
# Security: never retain passwords, tokens, secrets, or unsafe personal data.
# Transparency: users can list, inspect, modify, and delete memories.
# Contradictions: determine whether a new fact replaces an old one.
# Retrieval quality: select relevant memories without polluting the prompt.
# Prompt-injection resistance: stored text must be treated as untrusted data, not system instructions.
# Failure handling: /ask should still work when the memory Store fails.
# Concurrency: avoid conflicting updates from simultaneous requests.
# Lifecycle: migrations, backups, retention, expiration, and deletion.
# Observability: log operations without logging private memory content.
# Testing: prove isolation, opt-in behavior, persistence, and safe failure modes.

# There are roughly three maturity levels:
# Junior prototype
# └── Save arbitrary chat text and insert everything into the prompt

# Intermediate implementation
# └── Namespaced storage, retrieval, explicit commands, basic validation

# Senior/production implementation
# └── Consent, isolation, structured extraction, deduplication,
#     conflict resolution, security, observability, migrations,
#     deletion guarantees, evaluation, and graceful degradation
# Your current approach is moving in the senior direction because we are:
# Separating short-term and long-term memory.
# Keeping consent in user_settings.
# Reading consent once per query.
# Centralizing namespace construction.
# Using an established Store abstraction.
# Planning explicit inspection and deletion.
# Implementing small, reviewable steps.
# The senior-level part is not making the system remember. It is ensuring that it remembers the right information, for the correct user, only with permission, and remains controllable and safe.




# steps

# Up to this point, the following is complete:
# LTM enable/disable setting

# /memory on|off|status

# Per-user user_settings row

# LTM setting checked once per /ask

# Dedicated long_term_memory.sqlite

# AsyncSqliteStore initialization

# LangGraph Store schema

# Explicit namespace format:
# ("long_term_memories", user_id)

# Remaining steps:
# Add a simple function for saving one memory:
# await store.aput(
#     namespace,
#     memory_id,
#     {"memory": memory_text},
# )

# Add a simple function for listing the current user’s memories:
# await store.asearch(namespace)

# Add a CLI inspection command:
# /show_ltm

# Add a function for deleting one memory using its key:
# await store.adelete(namespace, memory_id)

# Add a CLI deletion command:
# /forget <memory_id>

# Give the LangGraph agent access to:
# AsyncSqliteStore
# user_id
# memory_enabled

# Add a simple memory-saving tool inspired by the provided example.

# Add a prompt instruction telling the agent what durable information may be remembered:
# Stable preferences
# Long-term goals
# Important achievements
# Ongoing projects

# Enforce memory_enabled inside the saving tool:
# disabled → reject Store writes
# enabled  → permit Store writes

# Retrieve the current user’s stored memories at the start of /ask when enabled.

# Add retrieved memories to the agent’s context.

# Ensure memory retrieval failure does not break /ask.

# Test persistence:
# Save memory
# Exit CLI
# Restart CLI
# Inspect memory

# Test user isolation:
# User A cannot see User B’s memories

# Add automatic memory detection only after explicit saving, listing, retrieval, and deletion are stable.

# Recommended next baby step:
# Add only the simple save-memory function.
# No CLI command, prompt change, automatic detection, or retrieval during that step.
