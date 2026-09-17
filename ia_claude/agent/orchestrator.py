import time

from rich.console import Console
from rich.prompt import Prompt

from ia_claude.cache.semantic_cache import CacheTiming
from ia_claude.config import config
from ia_claude.observability.logger import get_logger
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Command
from langgraph.store.base import Item

from ia_claude.agent.context import AgentContext
from ia_claude.agent.schemas import AgentResponse

logger = get_logger(__name__)
console = Console()


def _collect_human_decisions(interrupts) -> list[dict[str, str]]:
    """Ask the CLI user to approve or reject interrupted tool calls."""
    decisions = []
    for interrupt in interrupts:
        request = interrupt.value
        for action in request["action_requests"]:
            console.print("\n[yellow]Approval required[/yellow]")
            console.print(action["description"])
            approved = Prompt.ask(
                "Run this command? (yes/no)",
                choices=["y", "yes", "n", "no"],
                default="n",
            ).strip().lower()
            if approved in {"y", "yes"}:
                decisions.append({"type": "approve"})
                logger.info("HITL decision collected: approve")
            else:
                decisions.append(
                    {"type": "reject", "message": "Command rejected by the user."}
                )
                logger.info("HITL decision collected: reject")
    return decisions


async def handle_query(
    agent,
    question: str,
    thread_id: str,
    semantic_cache=None,
    cache_domain: str | None = None,
    memory_enabled: bool = False,
    agent_context: AgentContext | None = None,
    long_term_memories: list[Item] | None = None,
) -> str:
    """Run one query with the application-wide compiled agent.

    ``main`` owns the checkpointer lifetime and builds this agent once during
    startup. Passing it here keeps conversation state and expensive setup
    outside the per-query path.
    """

    request_started = time.perf_counter()
    logger.info("Handling query for session %s: %s", thread_id, question)
    logger.info(
        "Long-term memory enabled for query: %s",
        memory_enabled,
    )
    logger.info(
        "Long-term memories received for query: count=%s",
        len(long_term_memories or []),
    )
    # ``model`` remains the actual LLM model. The cache owns a separate
    # ``agent_namespace`` filter configured during its construction.
    model = config.get("llm", {}).get("model", "claude-3-5-sonnet-latest")
    agent_config = {"configurable": {"thread_id": thread_id}}

    # A shared answer cache is safe only for a new conversation. Later turns
    # can depend on the thread's prior messages, so they always invoke the
    # agent and use the SQLite-backed conversation state.
    cache_eligible = False
    cache_lookup_ms: float | None = None
    cached_hit = None
    if (
        semantic_cache is not None
        and cache_domain is not None
        and not memory_enabled
    ):
        try:
            state = await agent.aget_state(agent_config)
            cache_eligible = not state.values.get("messages", [])
        except Exception as e:
            # Fail closed: an unknown conversation state must not receive a
            # cross-session cached answer.
            logger.warning(
                "Could not inspect session state; skipping semantic cache: %s", e
            )

    if cache_eligible:
        try:
            logger.info(
                "Semantic cache lookup START: domain=%s agent_namespace=%s model=%s",
                cache_domain,
                semantic_cache.agent_namespace,
                model,
            )
            cache_lookup_started = time.perf_counter()
            cached_hit = await semantic_cache.get_with_metadata(
                question, domain=cache_domain, model=model
            )
            cache_lookup_ms = (time.perf_counter() - cache_lookup_started) * 1000
            logger.info(
                "Semantic cache lookup COMPLETE: outcome=%s duration_ms=%.1f",
                "HIT" if cached_hit is not None else "MISS",
                cache_lookup_ms,
            )
        except Exception as e:
            logger.warning(f"Semantic cache lookup failed, falling back to agent: {e}")
            cached_hit = None
        if cached_hit is not None:
            try:
                # The cache bypasses ``ainvoke``, so explicitly persist the
                # logical turn. This makes a later follow-up see the cached
                # exchange in the same SQLite checkpoint as normal turns.
                await agent.aupdate_state(
                    agent_config,
                    {
                        "messages": [
                            HumanMessage(content=question),
                            AIMessage(content=cached_hit.response),
                        ]
                    },
                )
            except Exception as e:
                # Do not return a response that was not recorded in session
                # memory; instead fall back to the normal agent path.
                logger.warning(
                    "Could not persist semantic cache hit; falling back to agent: %s",
                    e,
                )
            else:
                hit_total_ms = (time.perf_counter() - request_started) * 1000
                if cached_hit.timing is not None:
                    estimated_agent_time_saved_ms = max(
                        cached_hit.timing.agent_ms - hit_total_ms, 0.0
                    )
                    estimated_total_time_saved_ms = max(
                        cached_hit.timing.end_to_end_ms - hit_total_ms, 0.0
                    )
                    logger.info(
                        "Semantic cache HIT: lookup_ms=%.1f total_ms=%.1f "
                        "origin_agent_ms=%.1f origin_total_ms=%.1f "
                        "estimated_agent_time_saved_ms=%.1f "
                        "estimated_total_time_saved_ms=%.1f",
                        cache_lookup_ms,
                        hit_total_ms,
                        cached_hit.timing.agent_ms,
                        cached_hit.timing.end_to_end_ms,
                        estimated_agent_time_saved_ms,
                        estimated_total_time_saved_ms,
                    )
                else:
                    logger.info(
                        "Semantic cache HIT: lookup_ms=%.1f total_ms=%.1f "
                        "(no miss baseline stored)",
                        cache_lookup_ms,
                        hit_total_ms,
                    )
                logger.info(
                    "Semantic cache HIT - persisted turn and skipped agent/tool calls"
                )
                return cached_hit.response

    logger.info(
        "Agent invocation START: kind=initial thread_id=%s agent_id=%s "
        "semantic_cache_outcome=MISS_OR_BYPASSED",
        thread_id,
        id(agent),
    )
    agent_started = time.perf_counter()
    try:
        response = await agent.ainvoke(
            {"messages": [{"role": "user", "content": question}]},
            agent_config,
            context=agent_context,
        )

        resume_count = 0
        while interrupts := response.get("__interrupt__"):
            resume_count += 1
            logger.info(
                "Agent invocation INTERRUPTED: thread_id=%s agent_id=%s "
                "resume_count=%s interrupt_count=%s",
                thread_id,
                id(agent),
                resume_count,
                len(interrupts),
            )
            decisions = _collect_human_decisions(interrupts)
            logger.info(
                "Agent invocation RESUME: thread_id=%s agent_id=%s "
                "resume_count=%s decisions=%s",
                thread_id,
                id(agent),
                resume_count,
                [decision["type"] for decision in decisions],
            )
            response = await agent.ainvoke(
                Command(resume={"decisions": decisions}),
                agent_config,
                context=agent_context,
            )

        logger.info(
            "Agent invocation COMPLETE: thread_id=%s agent_id=%s resumes=%s",
            thread_id,
            id(agent),
            resume_count,
        )
        
        structured_response = response.get("structured_response")
        if isinstance(structured_response, AgentResponse):
            answer = structured_response.answer
            logger.info(
                "Structured agent response accepted: thread_id=%s "
                "schema=%s answer_length=%s",
                thread_id,
                type(structured_response).__name__,
                len(answer),
            )
        else:
            # Preserve compatibility with old checkpoints or providers that
            # return a normal final message instead of structured output.
            answer = response["messages"][-1].content
            logger.warning(
                "Structured response unavailable; using final message: "
                "thread_id=%s received_type=%s",
                thread_id,
                type(structured_response).__name__,
            )
    except Exception as exc:
        logger.exception("Agent error while handling query")
        return f"Error: {exc}"

    agent_ms = (time.perf_counter() - agent_started) * 1000
    pre_store_total_ms = (time.perf_counter() - request_started) * 1000

    # Store the fresh answer in the semantic cache on a miss
    if cache_eligible:
        cache_store_ms = None
        try:
            ttl = config.get("semantic_cache", {}).get("ttl", 86400)
            cache_store_started = time.perf_counter()
            await semantic_cache.put(
                question,
                answer,
                domain=cache_domain,
                model=model,
                ttl=ttl,
                timing=CacheTiming(
                    agent_ms=agent_ms,
                    end_to_end_ms=pre_store_total_ms,
                ),
            )
            cache_store_ms = (time.perf_counter() - cache_store_started) * 1000
        except Exception as e:
            logger.warning(f"Semantic cache store failed: {e}")
        total_ms = (time.perf_counter() - request_started) * 1000
        logger.info(
            "Semantic cache MISS: lookup_ms=%.1f agent_ms=%.1f "
            "total_ms=%.1f cache_store_ms=%s",
            cache_lookup_ms or 0.0,
            agent_ms,
            total_ms,
            f"{cache_store_ms:.1f}" if cache_store_ms is not None else "failed",
        )
    else:
        total_ms = (time.perf_counter() - request_started) * 1000
        logger.info(
            "Semantic cache BYPASSED: reason=%s agent_ms=%.1f total_ms=%.1f",
            (
                "long_term_memory_enabled"
                if memory_enabled
                else "session_history"
                if semantic_cache is not None
                else "disabled"
            ),
            agent_ms,
            total_ms,
        )

    return answer





# Keeping your static configuration logic inside a factory file (like factory.py) is standard practice because it keeps your system prompt, tool definitions, and builder functions clean and centralized.

# However, the file name itself doesn't automatically control caching. Anthropic achieves caching by looking at the exact order of the text sent over the API network payload.

# The Setup: Your build_agent function inside your factory compiles the static full_prompt and the tools list.

# The API Payload: When your orchestrator invokes the agent, LangChain automatically translates that factory output so it sits at the absolute beginning of the prompt structure sent to Anthropic.

# Why the Factory Matters: Because your factory generates that exact same string every single time the app boots or builds an agent, Anthropic's servers see a matching fingerprint at the top of the request and serve it from cache.

# You don't need to change where your code lives—your current structure in the factory file is already in the right place to take advantage of prompt caching, as long as you keep dynamic queries out of the static SYSTEM_PROMPT.

##
# it is not mandatory to explicitly mention the ttl.If you omit it, LangChain and Anthropic will automatically use a safe default—typically 5 minutes ("5m").  You only need to specify a ttl parameter if you want to explicitly change that default behavior:"5m" (Default): Keeps your cache alive for 5 minutes of inactivity. This is ideal for normal active chat sessions and standard coding workflows.  "1h" (Optional override): Keeps your cache alive for 1 hour. This is useful if your users tend to step away from their keyboards for long periods between coding questions, though keep in mind that longer TTLs can sometimes carry slightly different pricing or tier rules depending on your Anthropic account setup.  For most standard setups, leaving out the ttl argument entirely and just writing AnthropicPromptCachingMiddleware() (or using default middleware configurations) works completely fine.



# Stable Chunk IDs: Generate deterministic string IDs for every chunk (filepath::name::start_line) so Qdrant can overwrite existing points instead of accumulating duplicates.

# Metadata Timestamp Tracking (mtime): Inject the file's modification timestamp (os.path.getmtime) directly into each chunk's Qdrant payload metadata.

# State Inspection: Query Qdrant's existing payload data on startup to build a map of stored files and their last-indexed mtime values.

# Delta Execution Matrix:

# Unchanged files: Skipped entirely (saves embedding costs).

# Modified files: Delete old points matching the source path, then re-parse and upsert fresh chunks.

# New files: Parse and insert fresh chunks.

# Deleted files: Purge points associated with the missing file path.

# and

# Where the Data Lives

# Stored in Qdrant (Local or Remote): The mtime metadata lives inside your Qdrant database payload. Whether your Qdrant instance is running locally on your laptop (e.g., via a local Docker container or local path storage) or hosted remotely (e.g., Qdrant Cloud), the mtime values are stored right there inside the database server alongside your vector embeddings.

# Stored on your Laptop's Filesystem: The actual source files (models.py, main.py, etc.) live on your laptop's hard drive, and their physical file system modification timestamps (mtime) are managed directly by your operating system's file directory structure.

# When your agent runs, it compares the mtime sitting on your laptop's filesystem against the mtime stored inside your Qdrant database (whether that database is on your local machine or hosted remotely in the cloud).  


# and


# The Flow: From Code to Point

# File Scan & Parse: The agent reads your Python file (email_client.py), and tree-sitter splits it into logical code blocks (like the EmailMessage class).

# Payload Creation: The system packages the raw code into page_content and attaches contextual metadata (source, name, type, start_line, end_line, mtime).

# Dual Vector Generation:

# The code text is sent to your HuggingFace model to generate the Dense Vector.

# The same text is sent to BM25 (FastEmbed) to generate the Sparse Vector (langchain-sparse).

# Storage: Qdrant combines the metadata payload and both vectors into a single database record (the Point).



# You provide the previous thread_id to invoke.
# The checkpointer loads the old state.


# {
#     "messages": [
#         HumanMessage(
#             content="Create divide.py and then delete it"
#         ),
#         AIMessage(
#             content="",
#             tool_calls=[
#                 {
#                     "name": "delete_file",
#                     "args": {
#                         "file_path": "divide.py"
#                     },
#                     "id": "tool_call_123"
#                 }
#             ],
#         ),
#     ],

#     # LangGraph execution metadata
#     "next": ["HumanInTheLoopMiddleware.after_model"],

#     # Interrupt payload
#     "__interrupt__": [
#         {
#             "action_requests": [
#                 {
#                     "name": "delete_file",
#                     "args": {
#                         "file_path": "divide.py"
#                     },
#                     "description":
#                         "This will permanently delete a file:\n\ndivide.py",
#                 }
#             ],
#             "review_configs": [
#                 {
#                     "allowed_decisions": [
#                         "approve",
#                         "reject",
#                     ]
#                 }
#             ],
#         }
#     ],
# }



# SQLite checkpointer
# ├── thread_id: session-A
# │   └── messages and graph checkpoints for conversation A
# ├── thread_id: session-B
# │   └── messages and graph checkpoints for conversation B
# └── thread_id: session-C
#     └── messages and graph checkpoints for conversation C


# Checkpoint = saved state/data.
# Checkpointer = component that saves and loads checkpoints.
# The graph state records what has already happened for one particular thread_id.
