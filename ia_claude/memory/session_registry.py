"""Application-owned registry mapping new sessions to their users."""

from __future__ import annotations

import sqlite3
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from ia_claude.config import config
from ia_claude.observability.logger import get_logger
from ia_claude.user_context import validate_user_id


logger = get_logger(__name__)


@dataclass(frozen=True)
class SessionRecord:
    session_id: str
    created_at: str
    updated_at: str


def get_session_registry_db_path() -> Path:
    """Return the separate SQLite path for application session metadata."""
    configured = config.get("memory", {}).get(
        "session_registry_db_path",
        ".ia_claude_all_hidden/session_registry.sqlite",
    )
    return Path(configured)


def _connect() -> sqlite3.Connection:
    db_path = get_session_registry_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path, timeout=10)
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def _ensure_schema(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_sessions_user_id
        ON sessions(user_id)
        """
    )


def register_session(session_id: str, user_id: str) -> None:
    """Persist ownership metadata for a newly created session."""
    if not session_id or not session_id.strip():
        raise ValueError("session_id is required")

    canonical_user_id = validate_user_id(user_id)
    timestamp = datetime.now(timezone.utc).isoformat()

    with closing(_connect()) as connection:
        with connection:
            _ensure_schema(connection)
            connection.execute(
                """
                INSERT INTO sessions (session_id, user_id, created_at, updated_at)
                VALUES (?, ?, ?, ?)
                """,
                (session_id.strip(), canonical_user_id, timestamp, timestamp),
            )

    logger.info("Registered session ownership: session_id=%s", session_id)


def list_user_sessions(user_id: str) -> list[SessionRecord]:
    """Return only sessions registered to ``user_id``, newest first."""
    canonical_user_id = validate_user_id(user_id)

    with closing(_connect()) as connection:
        with connection:
            _ensure_schema(connection)
        rows = connection.execute(
            """
            SELECT session_id, created_at, updated_at
            FROM sessions
            WHERE user_id = ?
            ORDER BY updated_at DESC, created_at DESC
            """,
            (canonical_user_id,),
        ).fetchall()

    return [SessionRecord(*row) for row in rows]


def session_belongs_to_user(session_id: str, user_id: str) -> bool:
    """Return whether the registered session belongs to the given user."""
    if not session_id or not session_id.strip():
        return False

    canonical_user_id = validate_user_id(user_id)
    with closing(_connect()) as connection:
        with connection:
            _ensure_schema(connection)
        row = connection.execute(
            """
            SELECT 1
            FROM sessions
            WHERE session_id = ? AND user_id = ?
            LIMIT 1
            """,
            (session_id.strip(), canonical_user_id),
        ).fetchone()

    return row is not None
