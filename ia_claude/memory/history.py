"""Human-readable JSON exports for LangGraph conversation state."""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

from ia_claude.memory.session_registry import session_belongs_to_user
from ia_claude.user_context import validate_user_id


HISTORY_EXPORT_DIR_NAME = ".ia_claude_history_exports"


def _readable_content(content) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content)

    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and isinstance(block.get("text"), str):
            parts.append(block["text"])
        else:
            parts.append(json.dumps(block, ensure_ascii=False, default=str))
    return "\n".join(part for part in parts if part)


async def export_session_history(
    agent,
    session_id: str,
    user_id: str,
    output_root: str | Path,
) -> Path:
    """Export one owned session's human/assistant messages to JSON."""
    canonical_user_id = validate_user_id(user_id)
    canonical_session_id = str(uuid.UUID(session_id.strip()))
    if not session_belongs_to_user(canonical_session_id, canonical_user_id):
        raise ValueError("Session does not exist or is not available to this user.")

    state = await agent.aget_state(
        {"configurable": {"thread_id": canonical_session_id}}
    )
    messages = (state.values or {}).get("messages", [])

    exported_messages = []
    for message in messages:
        message_type = getattr(message, "type", "")
        if message_type not in {"human", "ai"}:
            continue
        exported_messages.append(
            {
                "role": "user" if message_type == "human" else "assistant",
                "content": _readable_content(getattr(message, "content", "")),
            }
        )

    export_dir = Path(output_root).resolve() / HISTORY_EXPORT_DIR_NAME
    export_dir.mkdir(parents=True, exist_ok=True)
    destination = export_dir / f"session_{canonical_session_id}.json"
    temporary = destination.with_suffix(".json.tmp")
    document = {
        "session_id": canonical_session_id,
        "user_id": canonical_user_id,
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "messages": exported_messages,
    }
    temporary.write_text(
        json.dumps(document, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary, destination)
    return destination
