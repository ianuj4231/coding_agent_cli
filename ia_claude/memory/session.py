import uuid
from pathlib import Path

from ia_claude.config import config
from ia_claude.observability.logger import get_logger
from ia_claude.memory.session_registry import (
    register_session,
    session_belongs_to_user,
)
from ia_claude.user_context import validate_user_id


logger = get_logger(__name__)


def get_legacy_session_file() -> Path:
    """Return the pre-multi-user active-session pointer path."""
    return Path(config.get("memory", {}).get("session_file", ".ia_claude_all_hidden/current_session"))


def get_session_file(user_id: str) -> Path:
    """Return the active-session pointer dedicated to one user."""
    canonical_user_id = validate_user_id(user_id)
    legacy_path = get_legacy_session_file()
    return legacy_path.parent / "users" / canonical_user_id / legacy_path.name


def get_current_session(user_id: str) -> str:
    """Return the current session/thread ID, or create a new one."""

    session_file = get_session_file(user_id)
    if session_file.exists():
        session_id = session_file.read_text().strip()

        if session_id:
            logger.info(
                f"Resuming session: {session_id}"
            )

            return session_id

    # Preserve a legacy current session only when the registry proves that it
    # belongs to this user. Unregistered or cross-user pointers are ignored.
    legacy_file = get_legacy_session_file()
    if legacy_file.exists():
        legacy_session_id = legacy_file.read_text().strip()
        if session_belongs_to_user(legacy_session_id, user_id):
            session_file.parent.mkdir(parents=True, exist_ok=True)
            session_file.write_text(legacy_session_id)
            logger.info(
                "Migrated legacy current session for user: %s",
                legacy_session_id,
            )
            return legacy_session_id

    return new_session(user_id)


def new_session(user_id: str) -> str:
    """Create and persist a new session/thread ID."""

    session_id = str(uuid.uuid4())

    session_file = get_session_file(user_id)
    session_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    register_session(session_id, user_id)
    session_file.write_text(session_id)

    logger.info(
        f"Started new session: {session_id}"
    )

    return session_id


class SessionAccessError(ValueError):
    """Raised when a session is unavailable to the current user."""


def switch_session(session_id: str, user_id: str) -> str:
    """Switch only to a session registered to the current user."""

    target = session_id.strip() if session_id else ""
    if not session_belongs_to_user(target, user_id):
        raise SessionAccessError(
            "Session does not exist or is not available to this user."
        )

    session_file = get_session_file(user_id)
    session_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    session_file.write_text(target)

    logger.info(
        f"Switched to session: {target}"
    )

    return target
