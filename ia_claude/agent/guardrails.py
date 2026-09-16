"""Deterministic safety checks for agent requests and terminal tool calls."""

from __future__ import annotations

import re
from typing import Any

from langchain.agents.middleware import AgentMiddleware, AgentState, hook_config
from langchain.messages import AIMessage
from langgraph.runtime import Runtime


BLOCKED_RESPONSE = "Sorry, I cannot help with that request."

# These are intentionally checked before a request is sent to a model. They are
# not a substitute for the terminal approval gate below, which protects against
# unsafe commands even when a harmless-looking prompt produces one.
_BLOCKED_REQUEST_PATTERNS = (
    r"\bhow (?:to )?(?:hack|break into|exploit)\b",
    r"\b(?:write|create|build|make) (?:a |an )?(?:malware|ransomware|keylogger|virus)\b",
    r"\b(?:destroy|sabotage)\b",
    r"\b(?:make|build|buy|use) (?:a |an )?(?:gun|firearm|weapon|bomb|explosive)\b",
)

_DESTRUCTIVE_COMMAND_PATTERNS = (
    r"(?:^|[;&|]\s*)rm\s+",
    r"(?:^|[;&|]\s*)(?:del|erase|rmdir|rd|remove-item|ri|unlink|shred)\b",
    r"\bgit\s+reset\s+--hard\b",
    r"\bgit\s+clean\s+-[^\s]*f",
    r"\b(?:format|mkfs)\b",
    r"\bdd\s+if=",
)


def _message_text(message: Any) -> str:
    """Extract plain text from LangChain's string or block message content."""
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            block if isinstance(block, str) else str(block.get("text", ""))
            for block in content
        )
    return str(content)


class ContentSafetyMiddleware(AgentMiddleware):
    """Stop disallowed user requests before any model invocation."""

    @hook_config(can_jump_to=["end"])
    def before_model(
        self, state: AgentState, runtime: Runtime
    ) -> dict[str, Any] | None:
        latest_user_message = next(
            (message for message in reversed(state["messages"]) if message.type == "human"),
            None,
        )
        if latest_user_message is None:
            return None

        request = _message_text(latest_user_message).lower()
        if any(re.search(pattern, request) for pattern in _BLOCKED_REQUEST_PATTERNS):
            return {
                "messages": [AIMessage(content=BLOCKED_RESPONSE)],
                "jump_to": "end",
            }
        return None


def requires_terminal_approval(request: Any) -> bool:
    """Return whether a terminal tool call can delete or irreversibly alter data."""
    command = str(request.tool_call["args"].get("command", "")).lower()
    return any(re.search(pattern, command) for pattern in _DESTRUCTIVE_COMMAND_PATTERNS)


def describe_terminal_action(tool_call: Any, _state: Any, _runtime: Any) -> str:
    """Build the approval text shown to the CLI user."""
    command = tool_call.get("args", {}).get("command", "")
    return f"This command can delete or permanently alter data:\n\n{command}"


def requires_file_deletion_approval(_request: Any) -> bool:
    """Require approval for every call to the direct filesystem delete tool."""
    return True


def describe_file_deletion(tool_call: Any, _state: Any, _runtime: Any) -> str:
    """Build the approval text shown before deleting a file directly."""
    file_path = tool_call.get("args", {}).get("file_path", "")
    return f"This will permanently delete a file:\n\n{file_path}"



# LLM Decision: The agent sends the conversation history and available tools to the LLM. The LLM decides to invoke a tool, such as generating a request to run run_command(command="...").

# Middleware Interception: Before Python actually executes the tool function, the request passes through your registered middleware stack (ContentSafetyMiddleware → SummarizationMiddleware → HumanInTheLoopMiddleware).

# Conditional Evaluation (when): The HumanInTheLoopMiddleware catches the incoming tool call and runs your requires_terminal_approval function against the command arguments.

# The Branching Point (Safe vs. Destructive):
# If Safe (e.g., pytest): The condition returns False. The middleware lets the request pass right through without stopping, and the tool executes immediately.
# If Destructive (e.g., rm or del): The condition returns True. The agent graph instantly pauses (interrupts) right before running the code.

# Human Prompt: The CLI stops execution, displays your warning description (describe_terminal_action), and waits for your input.

# User Action & Resumption:
# Approve: The graph resumes, and the terminal tool runs your command via subprocess.
# Reject: The tool call is canceled, and a rejection message is fed back into the agent's memory so the LLM knows you blocked it and can try an alternative approach.
