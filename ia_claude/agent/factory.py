from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware, SummarizationMiddleware

from ia_claude.agent.tools import create_search_codebase_tool
from ia_claude.agent.context import AgentContext
from ia_claude.agent.schemas import AgentResponse
from ia_claude.agent.memory_tools import remember_user_information
from ia_claude.agent.memory_prompt import add_long_term_memory_context
from ia_claude.agent.guardrails import (
    ContentSafetyMiddleware,
    describe_file_deletion,
    describe_terminal_action,
    requires_file_deletion_approval,
    requires_terminal_approval,
)
from ia_claude.llm.factory import get_llm
from ia_claude.observability.logger import get_logger

from ia_claude.tools.terminal_tools import run_command, run_in_directory
from ia_claude.tools.filesystem_tools import (
    read_file,
    write_file,
    append_file,
    delete_file,
    list_directory,
    file_exists,
)
from ia_claude.mcp.mcp_client import get_mcp_tools

from ia_claude.skills.skill_tools import build_skills_prompt,load_skill

logger = get_logger(__name__)

SYSTEM_PROMPT = """
You are a senior software engineer with deep knowledge of codebases.

Always use the search_codebase or file/terminal tools before answering questions about the codebase.

When answering:
- Reference specific file names.
- Reference class and function names.
- Reference line numbers when available.
- Base your answers on the retrieved code.
- Do not guess or invent information.
- If you cannot find the answer in the codebase, say so clearly.
"""

async def build_agent(checkpointer, user_id: str, long_term_store=None):
    """Build coding agent with full terminal/filesystem tools and summarization memory middleware."""

    llm = get_llm()

    
    mcp_tools =  await get_mcp_tools()
    
    skills_prompt = build_skills_prompt()
    full_prompt = SYSTEM_PROMPT
    if skills_prompt:
        full_prompt = SYSTEM_PROMPT + "\n\n" + skills_prompt


    
    
    
    search_codebase = create_search_codebase_tool(user_id)

    tools = [
        search_codebase,
        remember_user_information,
        load_skill,
        *mcp_tools,
        run_command,
        run_in_directory,
        read_file,
        write_file,
        append_file,
        delete_file,
        list_directory,
        file_exists,
    ]

    # 2. Define the Summarization Middleware with budget-friendly values
    summarization = SummarizationMiddleware(
        model=llm,
        trigger=("tokens", 4000),
        keep=("messages", 10),
    )
    terminal_approval = HumanInTheLoopMiddleware(
        interrupt_on={
            "delete_file": {
                "allowed_decisions": ["approve", "reject"],
                "when": requires_file_deletion_approval,##if returns true, needs hitl approval 
                "description": describe_file_deletion,
            },
            "run_command": {
                "allowed_decisions": ["approve", "reject"],
                "when": requires_terminal_approval,
                "description": describe_terminal_action,
            },
            "run_in_directory": {
                "allowed_decisions": ["approve", "reject"],
                "when": requires_terminal_approval,
                "description": describe_terminal_action,
            },
        }
    )
    middleware = [
        add_long_term_memory_context,
        ContentSafetyMiddleware(),
        summarization,
        terminal_approval,
    ]

    logger.info(
        "Building coding agent: tool_count=%s middleware_count=%s "
        "response_schema=%s strategy=automatic",
        len(tools),
        len(middleware),
        AgentResponse.__name__,
    )

    agent = create_agent(
        model=llm,
        tools=tools,
        response_format=AgentResponse,
        system_prompt=full_prompt,
        checkpointer=checkpointer,
        store=long_term_store,
        context_schema=AgentContext,
        middleware=middleware,
    )

    logger.info(
        "Coding agent built: agent_id=%s checkpointer_type=%s store_type=%s "
        "response_schema=%s",
        id(agent),
        type(checkpointer).__name__,
        type(long_term_store).__name__ if long_term_store is not None else "None",
        AgentResponse.__name__,
    )

    return agent
