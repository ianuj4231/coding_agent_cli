from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, List, Optional
from fastapi import FastAPI, HTTPException, Depends, status
from pydantic import BaseModel, Field

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from ia_claude.config import config
from ia_claude.llm.factory import get_llm, get_embedder
from ia_claude.observability.logger import get_logger
from ia_claude.agent.orchestrator import handle_query
from ia_claude.agent.context import AgentContext
from ia_claude.agent.evaluation_trace import finish_trace, start_trace
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
from ia_claude.memory.user_settings import MemorySettingsError, is_memory_enabled
from ia_claude.memory.long_term import (
    LongTermMemoryReadError,
    LongTermMemoryStoreError,
    list_long_term_memories,
    open_long_term_memory_store,
)
from ia_claude.cache.semantic_cache import (
    build_semantic_cache,
    get_repo_domain,
)
from ia_claude.context.indexers.watcher import (
    start_watcher,
    stop_watcher,
)
from ia_claude.user_context import DEFAULT_USER_ID

logger = get_logger(__name__)

# Global runtime state for the API server
app_state: dict[str, Any] = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup initialization
    logger.info("Starting IA Claude API server lifespan initialization")
    repo_path = str(Path.cwd())
    
    # Initialize checkpointer and long term memory store
    checkpointer_db = get_checkpointer_db_path()
    checkpointer_saver = AsyncSqliteSaver.from_conn_string(checkpointer_db)
    checkpointer = await checkpointer_saver.__aenter__()
    
    try:
        long_term_store = await open_long_term_memory_store().__aenter__()
    except Exception:
        long_term_store = None
        logger.exception("Long-term memory store unavailable during API startup")

    # Default bootstrap user for API operations if not provided
    default_user_id = DEFAULT_USER_ID
    
    provider = config["vector_store"]["provider"].lower()
    logger.info(f"API using vector store provider: {provider}")
    index_codebase = get_indexer()
    vector_store = index_codebase(repo_path, default_user_id)
    
    semantic_cache = await build_semantic_cache()
    cache_domain = get_repo_domain(repo_path) if semantic_cache is not None else None

    # Watcher
    observer = start_watcher(
        repo_path,
        default_user_id,
        on_change=lambda: None, # no-op or custom invalidation
    )

    agent = await build_agent(
        checkpointer,
        default_user_id,
        long_term_store,
    )

    app_state["checkpointer"] = checkpointer
    app_state["checkpointer_saver"] = checkpointer_saver
    app_state["long_term_store"] = long_term_store
    app_state["vector_store"] = vector_store
    app_state["agent"] = agent
    app_state["semantic_cache"] = semantic_cache
    app_state["cache_domain"] = cache_domain
    app_state["observer"] = observer
    app_state["repo_path"] = repo_path

    logger.info("IA Claude API server initialized successfully")
    yield
    
    # Shutdown cleanup
    logger.info("Shutting down IA Claude API server lifespan")
    stop_watcher(observer)
    if long_term_store is not None:
        try:
            await long_term_store.__aexit__(None, None, None)
        except Exception:
            pass
    if checkpointer_saver is not None:
        try:
            await checkpointer_saver.__aexit__(None, None, None)
        except Exception:
            pass
    logger.info("IA Claude API server shutdown complete")

app = FastAPI(
    title="IA Claude RAG Agent API",
    description="REST API for interacting with the IA Claude code assistant, vector search, and agent tools.",
    version="0.1.0",
    lifespan=lifespan,
)

# Pydantic models for requests and responses
class AskRequest(BaseModel):
    question: str = Field(..., description="The coding question or instruction for the agent.")
    user_id: Optional[str] = Field(DEFAULT_USER_ID, description="User ID for session and memory scoping.")
    session_id: Optional[str] = Field(None, description="Optional session ID to continue a specific thread.")
    include_evaluation_details: bool = False


class EvaluationDetails(BaseModel):
    candidates: list[dict[str, str]]
    selected: list[dict[str, str]]

class AskResponse(BaseModel):
    answer: str
    session_id: str
    user_id: str
    evaluation_details: Optional[EvaluationDetails] = None

class SessionCreateRequest(BaseModel):
    user_id: str = Field(DEFAULT_USER_ID, description="User ID owning the new session.")

class SessionResponse(BaseModel):
    session_id: str
    user_id: str
    created_at: Optional[str] = None

class ReindexResponse(BaseModel):
    status: str
    message: str


@app.get("/health", tags=["System"])
async def health_check():
    """Health check endpoint to verify the API is running."""
    return {"status": "healthy", "service": "ia-claude-api"}


@app.post(
    "/ask",
    response_model=AskResponse,
    response_model_exclude_none=True,
    tags=["Agent"],
)
async def ask_question(payload: AskRequest):
    """Ask a coding question to the RAG agent."""
    agent = app_state.get("agent")
    semantic_cache = app_state.get("semantic_cache")
    cache_domain = app_state.get("cache_domain")
    long_term_store = app_state.get("long_term_store")

    if not agent:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Agent not initialized")

    user_id = payload.user_id or DEFAULT_USER_ID
    session_id = payload.session_id or get_current_session(user_id)

    # Determine memory status
    try:
        memory_enabled = (
            not payload.include_evaluation_details
            and is_memory_enabled(user_id)
            and long_term_store is not None
        )
    except Exception:
        memory_enabled = False

    long_term_memories = []
    if memory_enabled and long_term_store is not None:
        try:
            long_term_memories = await list_long_term_memories(long_term_store, user_id)
        except Exception:
            logger.exception("Failed to load long-term memories")

    memory_texts = tuple(
        item.value["memory"]
        for item in long_term_memories
        if isinstance(item.value, dict) and isinstance(item.value.get("memory"), str)
    )
    agent_context = AgentContext(
        user_id=user_id,
        memory_enabled=memory_enabled,
        long_term_memories=memory_texts,
    )

    trace_token = start_trace() if payload.include_evaluation_details else None
    try:
        answer = await handle_query(
            agent,
            payload.question,
            session_id,
            semantic_cache=None if payload.include_evaluation_details else semantic_cache,
            cache_domain=None if payload.include_evaluation_details else cache_domain,
            memory_enabled=memory_enabled,
            agent_context=agent_context,
            long_term_memories=long_term_memories,
        )
    except Exception as exc:
        logger.exception("Error handling ask request in API")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))
    finally:
        evaluation_details = (
            finish_trace(trace_token) if trace_token is not None else None
        )

    return AskResponse(
        answer=answer,
        session_id=session_id,
        user_id=user_id,
        evaluation_details=evaluation_details,
    )


@app.post("/sessions", response_model=SessionResponse, tags=["Sessions"])
async def create_new_session(payload: SessionCreateRequest):
    """Start a new conversation session for a user."""
    try:
        sid = new_session(payload.user_id)
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    return SessionResponse(session_id=sid, user_id=payload.user_id)


@app.get("/sessions/{user_id}", response_model=List[SessionResponse], tags=["Sessions"])
async def list_sessions(user_id: str):
    """List all registered sessions for a user."""
    try:
        sessions = list_user_sessions(user_id)
        return [
            SessionResponse(
                session_id=s.session_id,
                user_id=user_id,
                created_at=str(s.created_at) if hasattr(s, "created_at") else None
            )
            for s in sessions
        ]
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))


@app.post("/reindex", response_model=ReindexResponse, tags=["Codebase"])
async def trigger_reindex(user_id: str = DEFAULT_USER_ID):
    """Trigger a delta-sync reindex of the codebase and invalidate semantic cache."""
    try:
        index_codebase = get_indexer()
        index_codebase(app_state["repo_path"], user_id)
        
        semantic_cache = app_state.get("semantic_cache")
        cache_domain = app_state.get("cache_domain")
        if semantic_cache and cache_domain:
            await semantic_cache.invalidate_domain(cache_domain)
            
        return ReindexResponse(
            status="success",
            message="Codebase successfully reindexed and semantic cache invalidated."
        )
    except Exception as exc:
        logger.exception("Reindex failed via API")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))
