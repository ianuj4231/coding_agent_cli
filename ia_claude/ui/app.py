import asyncio
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage

from ia_claude.config import config
from ia_claude.llm.factory import get_llm, get_embedder
from ia_claude.context.indexers.factory import get_indexer
from ia_claude.cache.semantic_cache import build_semantic_cache, get_repo_domain
from ia_claude.agent.factory import build_agent
from ia_claude.agent.context import AgentContext
from ia_claude.agent.orchestrator import handle_query
from ia_claude.memory.short_term import get_checkpointer_db_path
from ia_claude.memory.session import get_current_session, new_session
from ia_claude.memory.user_settings import is_memory_enabled
from ia_claude.memory.long_term import open_long_term_memory_store, list_long_term_memories
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

st.set_page_config(
    page_title="IA Claude - RAG-powered Code Assistant",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 IA Claude")
st.markdown("RAG-powered code assistant with vector search, semantic caching, and long-term memory.")

# Sidebar for configuration and sessions
st.sidebar.header("Configuration & Session")
user_id = st.sidebar.text_input("User ID", value="default_user")

if "current_session_id" not in st.session_state:
    st.session_state.current_session_id = get_current_session(user_id)

st.sidebar.text(f"Session: {st.session_state.current_session_id}")

if st.sidebar.button("New Session"):
    st.session_state.current_session_id = new_session(user_id)
    st.rerun()

# Initialize backend resources cached in session state
@st.cache_resource
def init_backend(uid: str):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    repo_path = str(Path.cwd())
    index_codebase = get_indexer()
    vector_store = index_codebase(repo_path, uid)
    
    # Checkpointer
    checkpointer_path = get_checkpointer_db_path()
    # We will instantiate checkpointer in async calls or use a loop wrapper
    return vector_store

vector_store = init_backend(user_id)

# Chat history storage in streamlit session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Ask a question about your codebase..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            async def run_query():
                async with AsyncSqliteSaver.from_conn_string(get_checkpointer_db_path()) as checkpointer:
                    try:
                        async with open_long_term_memory_store() as ltm_store:
                            long_term_store = ltm_store
                    except Exception:
                        long_term_store = None

                    agent = await build_agent(checkpointer, user_id, long_term_store)
                    semantic_cache = await build_semantic_cache()
                    cache_domain = get_repo_domain(str(Path.cwd())) if semantic_cache is not None else None

                    memory_enabled = is_memory_enabled(user_id) if long_term_store is not None else False
                    long_term_memories = []
                    if memory_enabled and long_term_store is not None:
                        try:
                            long_term_memories = await list_long_term_memories(long_term_store, user_id)
                        except Exception:
                            pass

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

                    response = await handle_query(
                        agent,
                        prompt,
                        st.session_state.current_session_id,
                        semantic_cache=semantic_cache,
                        cache_domain=cache_domain,
                        memory_enabled=memory_enabled,
                        agent_context=agent_context,
                        long_term_memories=long_term_memories,
                    )
                    return response

            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                response_text = loop.run_until_complete(run_query())
                loop.close()
            except Exception as e:
                response_text = f"Error executing query: {e}"

            st.markdown(response_text)
            st.session_state.messages.append({"role": "assistant", "content": response_text})
