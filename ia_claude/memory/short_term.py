from pathlib import Path
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from ia_claude.config import config
from ia_claude.observability.logger import get_logger

logger = get_logger(__name__)

def get_checkpointer_db_path() -> str:
    db_path = config.get("memory", {}).get("db_path", "checkpoints.sqlite")
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Using SQLite checkpointer at {db_path}")
    return db_path

def get_checkpointer():
    """Return the SQLite saver context manager.

    ``AsyncSqliteSaver.from_conn_string`` opens its connection on context
    entry, so callers must use ``async with`` to obtain the saver instance.
    """
    db_path = get_checkpointer_db_path()
    return AsyncSqliteSaver.from_conn_string(db_path)

def get_summarization_middleware(llm):
    from langchain.agents.middleware import SummarizationMiddleware
    return SummarizationMiddleware(
        model=llm,
        trigger=("tokens", config.get("memory", {}).get("summarize_at_tokens", 4000)),
        keep=("messages", config.get("memory", {}).get("keep_last_messages", 10)),
    )



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

