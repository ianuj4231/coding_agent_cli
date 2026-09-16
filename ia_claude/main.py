import asyncio
from contextlib import AsyncExitStack
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.prompt import Prompt


# Load environment variables before importing project modules
load_dotenv(
    Path(__file__).resolve().parent.parent / ".env"
)


from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from ia_claude.config import config
from ia_claude.llm.factory import get_llm, get_embedder

from ia_claude.observability.logger import get_logger

from ia_claude.agent.orchestrator import handle_query
from ia_claude.agent.context import AgentContext
from ia_claude.agent.factory import build_agent

from ia_claude.context.indexers.factory import get_indexer

from ia_claude.memory.short_term import get_checkpointer_db_path
from ia_claude.memory.session import (
    SessionAccessError,
    get_current_session,
    new_session,
    switch_session,
)
from ia_claude.memory.session_registry import list_user_sessions
from ia_claude.memory.history import export_session_history
from ia_claude.memory.commands import (
    handle_forget_all_command,
    handle_forget_command,
    handle_memory_command,
    handle_remember_command,
    handle_show_ltm_command,
)
from ia_claude.memory.long_term import (
    LongTermMemoryReadError,
    LongTermMemoryStoreError,
    list_long_term_memories,
    open_long_term_memory_store,
)
from ia_claude.memory.user_settings import MemorySettingsError, is_memory_enabled
from ia_claude.user_context import prompt_for_user_id

from ia_claude.context.indexers.watcher import (
    start_watcher,
    stop_watcher,
)

# ---------------------------------------------------------
# Semantic cache
# ---------------------------------------------------------
from ia_claude.cache.semantic_cache import (
    build_semantic_cache,
    get_repo_domain,
)


console = Console()
logger = get_logger(__name__)


def get_or_create_index(user_id: str):
    """
    Load the existing vector index or create/update one using
    the configured vector-store backend.
    """

    repo_path = str(Path.cwd())

    provider = config["vector_store"]["provider"].lower()

    logger.info(
        f"Using vector store provider: {provider}"
    )

    console.print(
        f"[dim]Vector store: {provider}[/dim]"
    )

    index_codebase = get_indexer()

    logger.info(
        f"Preparing vector index for: {repo_path}"
    )

    console.print(
        f"[dim]Preparing index for {repo_path}...[/dim]"
    )

    vector_store = index_codebase(repo_path, user_id)

    return vector_store


async def initialize(checkpointer, user_id: str, long_term_store=None):
    """
    Initialize shared services once before starting the REPL.

    Initializes:
    - LLM
    - Embedder
    - Vector index
    - Semantic cache
    - Filesystem watcher
    - Agent
    - Conversation session
    """

    # ---------------------------------------------------------
    # LLM + embeddings
    # ---------------------------------------------------------
    llm = get_llm()
    embedder = get_embedder()

    console.print(
        f"[dim]LLM: "
        f"{config['llm']['provider']} / "
        f"{config['llm']['model']}[/dim]"
    )

    console.print(
        f"[dim]Embedder: "
        f"{config['embeddings']['provider']} / "
        f"{config['embeddings']['model']}[/dim]"
    )

    # ---------------------------------------------------------
    # Repository / vector index
    # ---------------------------------------------------------
    repo_path = str(Path.cwd())

    vector_store = get_or_create_index(user_id)

    # ---------------------------------------------------------
    # Semantic cache
    # ---------------------------------------------------------
    semantic_cache = await build_semantic_cache()

    cache_domain = (
        get_repo_domain(repo_path)
        if semantic_cache is not None
        else None
    )

    if semantic_cache is not None:
        console.print(
            "[dim]Semantic cache: enabled "
            f"(threshold={semantic_cache.threshold})[/dim]"
        )

        logger.info(
            "Semantic cache enabled "
            f"for domain: {cache_domain}"
        )

    else:
        console.print(
            "[dim]Semantic cache: disabled[/dim]"
        )

        logger.info(
            "Semantic cache disabled"
        )

    # ---------------------------------------------------------
    # Cache invalidation when repository files change
    # ---------------------------------------------------------
    loop = asyncio.get_running_loop()

    def _invalidate_cache_on_change():
        """
        Called by the filesystem watcher whenever source files change.

        The watcher normally runs in another thread, so schedule the
        async cache invalidation safely on the main asyncio loop.
        """

        if semantic_cache is None:
            return

        if cache_domain is None:
            return

        try:
            future = asyncio.run_coroutine_threadsafe(
                semantic_cache.invalidate_domain(
                    cache_domain
                ),
                loop,
            )

            # Retrieve async errors instead of silently losing them.
            def _handle_invalidation_result(result_future):
                try:
                    result_future.result()
                except Exception:
                    logger.exception(
                        "Failed to invalidate semantic cache "
                        "after repository change"
                    )

            future.add_done_callback(
                _handle_invalidation_result
            )

        except Exception:
            logger.exception(
                "Unable to schedule semantic cache invalidation"
            )

    # ---------------------------------------------------------
    # Filesystem watcher
    # ---------------------------------------------------------
    observer = start_watcher(
        repo_path,
        user_id,
        on_change=_invalidate_cache_on_change,
    )

    # ---------------------------------------------------------
    # Agent
    # ---------------------------------------------------------
    agent = await build_agent(
        checkpointer,
        user_id,
        long_term_store,
    )

    # ---------------------------------------------------------
    # Session
    # ---------------------------------------------------------
    session_id = get_current_session(user_id)

    console.print(
        f"[dim]Session: {session_id}[/dim]"
    )

    console.print(
        "[green]✓ Ready[/green]\n"
    )

    return (
        llm,
        embedder,
        vector_store,
        agent,
        session_id,
        observer,
        semantic_cache,
        cache_domain,
    )


async def run_async():
    """Run the interactive IA Claude command-line interface."""

    logger.info(
        "Starting IA Claudexxy"
    )

    console.print(
        "\n[bold blue]IA Claude[/bold blue] "
        "- RAG-powered code assistant"
    )

    user_id = prompt_for_user_id(console)

    async with AsyncExitStack() as resources:
        checkpointer = await resources.enter_async_context(
            AsyncSqliteSaver.from_conn_string(get_checkpointer_db_path())
        )
        try:
            long_term_store = await resources.enter_async_context(
                open_long_term_memory_store()
            )
        except LongTermMemoryStoreError:
            long_term_store = None
            logger.exception(
                "Long-term memory unavailable; continuing without it"
            )
            console.print(
                "[yellow]Long-term memory is temporarily unavailable. "
                "Other features will continue normally.[/yellow]"
            )

        (
            llm,
            embedder,
            vector_store,
            agent,
            session_id,
            observer,
            semantic_cache,
            cache_domain,
        ) = await initialize(
            checkpointer,
            user_id,
            long_term_store,
        )

        console.print(
            "Type [bold]'/exit'[/bold] to quit\n"
        )

        try:
            while True:
                user_input = Prompt.ask(
                    "[bold green]>[/bold green]"
                )

                user_input = user_input.strip()

                if not user_input:
                    continue

                # ---------------------------------------------------------
                # Exit
                # ---------------------------------------------------------
                if user_input.lower() in (
                    "/exit",
                    "/quit",
                ):
                    logger.info(
                        "Shutting down"
                    )

                    console.print(
                        "[dim]Goodbye![/dim]"
                    )

                    break

                # ---------------------------------------------------------
                # Ask
                # ---------------------------------------------------------
                elif user_input.startswith("/ask "):
                    question = (
                        user_input
                        .removeprefix("/ask ")
                        .strip()
                    )

                    if not question:
                        console.print(
                            "[yellow]Usage: "
                            "/ask <question>[/yellow]"
                        )
                        continue

                    # Resolve this once per query. The database remains the
                    # source of truth, while every stage of this request sees
                    # one consistent long-term-memory preference.
                    try:
                        memory_enabled = (
                            is_memory_enabled(user_id)
                            and long_term_store is not None
                        )
                    except MemorySettingsError:
                        memory_enabled = False
                        logger.exception(
                            "Could not read memory setting; disabling long-term "
                            "memory for this query"
                        )

                    logger.info(
                        "Long-term memory setting for query: enabled=%s",
                        memory_enabled,
                    )

                    long_term_memories = []
                    if memory_enabled and long_term_store is not None:
                        try:
                            long_term_memories = await list_long_term_memories(
                                long_term_store,
                                user_id,
                            )
                        except LongTermMemoryReadError:
                            logger.exception(
                                "Long-term memory retrieval failed; continuing "
                                "without memories"
                            )

                    logger.info(
                        "Long-term memories loaded for query: count=%s",
                        len(long_term_memories),
                    )

                    memory_texts = tuple(
                        item.value["memory"]
                        for item in long_term_memories
                        if isinstance(item.value, dict)
                        and isinstance(item.value.get("memory"), str)
                    )
                    agent_context = AgentContext(
                        user_id=user_id,
                        memory_enabled=memory_enabled,
                        long_term_memories=memory_texts,
                    )

                    logger.info(
                        f"Ask command received: {question}"
                    )

                    console.print(
                        f"[dim]Searching for: "
                        f"{question}...[/dim]"
                    )

                    response = await handle_query(
                        agent,
                        question,
                        session_id,
                        semantic_cache=semantic_cache,
                        cache_domain=cache_domain,
                        memory_enabled=memory_enabled,
                        agent_context=agent_context,
                        long_term_memories=long_term_memories,
                    )

                    console.print(
                        response
                    )

                # ---------------------------------------------------------
                # Persisted long-term-memory preference
                # ---------------------------------------------------------
                elif user_input.split(maxsplit=1)[0].lower() == "/memory":
                    message = handle_memory_command(user_id, user_input)
                    console.print(message)

                # ---------------------------------------------------------
                # Inspect this user's long-term memories
                # ---------------------------------------------------------
                elif user_input.lower() == "/show_ltm":
                    message = await handle_show_ltm_command(
                        long_term_store,
                        user_id,
                    )
                    console.print(message)

                # ---------------------------------------------------------
                # Save one explicit long-term memory
                # ---------------------------------------------------------
                elif user_input.split(maxsplit=1)[0].lower() == "/remember":
                    message = await handle_remember_command(
                        long_term_store,
                        user_id,
                        user_input,
                    )
                    console.print(message)

                # ---------------------------------------------------------
                # Delete one of this user's long-term memories
                # ---------------------------------------------------------
                elif user_input.split(maxsplit=1)[0].lower() == "/forget":
                    message = await handle_forget_command(
                        long_term_store,
                        user_id,
                        user_input,
                    )
                    console.print(message)

                # ---------------------------------------------------------
                # Delete all of this user's long-term memories
                # ---------------------------------------------------------
                elif user_input.lower() == "/forget_all":
                    confirmed = Prompt.ask(
                        "Delete all your long-term memories?",
                        choices=["y", "yes", "n", "no"],
                        default="n",
                    ).strip().lower()
                    if confirmed not in {"y", "yes"}:
                        console.print("Deletion cancelled.")
                        continue

                    message = await handle_forget_all_command(
                        long_term_store,
                        user_id,
                    )
                    console.print(message)

                # ---------------------------------------------------------
                # New session
                # ---------------------------------------------------------
                elif user_input == "/new_session":
                    session_id = new_session(user_id)

                    console.print(
                        f"[green]New session started: "
                        f"{session_id}[/green]"
                    )

                # ---------------------------------------------------------
                # List this user's registered sessions
                # ---------------------------------------------------------
                elif user_input == "/get_sessions":
                    sessions = list_user_sessions(user_id)

                    if not sessions:
                        console.print(
                            "[dim]No registered sessions found for this user.[/dim]"
                        )
                        continue

                    console.print("[bold]Your sessions:[/bold]")
                    for session in sessions:
                        current_marker = (
                            " [green](current)[/green]"
                            if session.session_id == session_id
                            else ""
                        )
                        console.print(
                            f"  {session.session_id}{current_marker}\n"
                            f"    created: {session.created_at}"
                        )

                # ---------------------------------------------------------
                # Export current conversation as readable JSON
                # ---------------------------------------------------------
                elif user_input == "/export_history":
                    try:
                        export_path = await export_session_history(
                            agent,
                            session_id,
                            user_id,
                            Path.cwd(),
                        )
                    except ValueError as exc:
                        console.print(f"[red]Cannot export history: {exc}[/red]")
                        continue

                    console.print(
                        f"[green]History exported:[/green] {export_path}"
                    )

                # ---------------------------------------------------------
                # Switch session
                # ---------------------------------------------------------
                elif user_input.startswith("/switch "):
                    target = (
                        user_input
                        .removeprefix("/switch ")
                        .strip()
                    )

                    if not target:
                        console.print(
                            "[yellow]Usage: "
                            "/switch <session_id>[/yellow]"
                        )
                        continue

                    try:
                        session_id = switch_session(target, user_id)
                    except SessionAccessError as exc:
                        console.print(f"[red]Cannot switch session: {exc}[/red]")
                        continue

                    console.print(
                        f"[green]Switched to session: "
                        f"{session_id}[/green]"
                    )

                # ---------------------------------------------------------
                # Show current session
                # ---------------------------------------------------------
                elif user_input == "/session":
                    console.print(
                        f"[dim]Current session: "
                        f"{session_id}[/dim]"
                    )

                # ---------------------------------------------------------
                # Reindex
                # ---------------------------------------------------------
                elif user_input == "/reindex":
                    logger.info(
                        "Manual reindex requested"
                    )

                    console.print(
                        "[dim]Running delta-sync "
                        "reindex on codebase...[/dim]"
                    )

                    vector_store = get_or_create_index(user_id)

                    # Reindexing can make cached answers stale.
                    # Clear only this repository's semantic-cache domain.
                    if (
                        semantic_cache is not None
                        and cache_domain is not None
                    ):
                        try:
                            await semantic_cache.invalidate_domain(
                                cache_domain
                            )

                            console.print(
                                "[dim]Semantic cache invalidated "
                                "for this repository.[/dim]"
                            )

                            logger.info(
                                "Semantic cache invalidated "
                                f"for domain: {cache_domain}"
                            )

                        except Exception:
                            logger.exception(
                                "Failed to invalidate semantic "
                                "cache after manual reindex"
                            )

                            console.print(
                                "[yellow]Warning: codebase was "
                                "reindexed, but semantic cache "
                                "invalidation failed.[/yellow]"
                            )

                    console.print(
                        "[green]✓ Codebase reindexed "
                        "successfully[/green]"
                    )

                # ---------------------------------------------------------
                # Show semantic/vector index
                # ---------------------------------------------------------
                elif user_input == "/show_semantic_index":
                    logger.info(
                        "Showing vector index"
                    )

                    provider = (
                        config["vector_store"]["provider"]
                        .lower()
                    )

                    try:
                        # -------------------------------------------------
                        # Qdrant
                        # -------------------------------------------------
                        if provider == "qdrant":
                            retrieval_mode = (
                                config["vector_store"]
                                .get(
                                    "retrieval_mode",
                                    "dense",
                                )
                                .lower()
                            )

                            if retrieval_mode in (
                                "hybrid",
                                "sparse",
                            ):
                                from ia_claude.context.indexers.hybrid_qdrant import (
                                    show_index,
                                )

                            else:
                                from ia_claude.context.indexers.semantic_qdrant import (
                                    show_index,
                                )

                            if retrieval_mode in ("hybrid", "sparse"):
                                show_index(vector_store, user_id)
                            else:
                                show_index(vector_store)

                        # -------------------------------------------------
                        # Chroma
                        # -------------------------------------------------
                        elif provider == "chroma":
                            from ia_claude.context.indexers.semantic_chroma import (
                                show_index,
                            )

                            show_index(
                                vector_store
                            )

                        # -------------------------------------------------
                        # Unsupported vector store
                        # -------------------------------------------------
                        else:
                            console.print(
                                "[red]Unsupported vector "
                                f"store: {provider}[/red]"
                            )

                    except ImportError as e:
                        logger.error(
                            "Failed to import index "
                            f"viewer for {provider}: {e}"
                        )

                        console.print(
                            "[red]Could not load index "
                            f"viewer module: {e}[/red]"
                        )

                    except Exception as e:
                        logger.exception(
                            "Failed to show vector index"
                        )

                        console.print(
                            "[red]Failed to show vector "
                            f"index: {e}[/red]"
                        )

                # ---------------------------------------------------------
                # Unknown command
                # ---------------------------------------------------------
                else:
                    logger.warning(
                        "Unknown command received: "
                        f"{user_input}"
                    )

                    console.print(
                        "[yellow]Unknown command. "
                        "Try:[/yellow]"
                    )

                    console.print(
                        "  [bold]/ask <question>[/bold] "
                        "— ask a question about the codebase"
                    )

                    console.print(
                        "  [bold]/reindex[/bold] "
                        "— force refresh/delta-sync "
                        "the codebase index"
                    )

                    console.print(
                        "  [bold]/show_semantic_index[/bold] "
                        "— show chunks + embeddings"
                    )

                    console.print(
                        "  [bold]/new_session[/bold] "
                        "— start a fresh conversation"
                    )

                    console.print(
                        "  [bold]/switch <session_id>[/bold] "
                        "— resume a past session"
                    )

                    console.print(
                        "  [bold]/get_sessions[/bold] — list your registered sessions"
                    )

                    console.print(
                        "  [bold]/export_history[/bold] — export current session to JSON"
                    )

                    console.print(
                        "  [bold]/memory <on|off|status>[/bold] "
                        "— manage long-term memory"
                    )

                    console.print(
                        "  [bold]/show_ltm[/bold] "
                        "— inspect your long-term memories"
                    )

                    console.print(
                        "  [bold]/remember <text>[/bold] "
                        "— save an explicit long-term memory"
                    )

                    console.print(
                        "  [bold]/forget <memory_id>[/bold] "
                        "— delete one long-term memory"
                    )

                    console.print(
                        "  [bold]/forget_all[/bold] "
                        "— delete all your long-term memories"
                    )

                    console.print(
                        "  [bold]/session[/bold] "
                        "— show current session id"
                    )

                    console.print(
                        "  [bold]/exit[/bold] "
                        "— quit IA Claude"
                    )

        finally:
            stop_watcher(
                observer
            )


def run():
    """Synchronous console-script entry point."""

    asyncio.run(
        run_async()
    )


if __name__ == "__main__":
    run()
