"""Persisted, per-user application settings."""

from __future__ import annotations

import sqlite3
from contextlib import closing
from datetime import datetime, timezone

from ia_claude.memory.session_registry import get_session_registry_db_path
from ia_claude.user_context import validate_user_id


class MemorySettingsError(RuntimeError):
    """Raised when the persisted memory preference cannot be accessed."""


def _connect() -> sqlite3.Connection:
    db_path = get_session_registry_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(db_path, timeout=10)


def _ensure_schema(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS user_settings (
            user_id TEXT PRIMARY KEY,
            memory_enabled INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            CHECK (memory_enabled IN (0, 1))
        )
        """
    )


def set_memory_enabled(user_id: str, enabled: bool) -> None:
    """Enable or disable long-term memory for a user."""
    canonical_user_id = validate_user_id(user_id)
    if not isinstance(enabled, bool):
        raise TypeError("enabled must be a boolean")

    timestamp = datetime.now(timezone.utc).isoformat()
    try:
        with closing(_connect()) as connection:
            with connection:
                _ensure_schema(connection)
                connection.execute(
                    """
                    INSERT INTO user_settings (
                        user_id, memory_enabled, created_at, updated_at
                    )
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(user_id) DO UPDATE SET
                        memory_enabled = excluded.memory_enabled,
                        updated_at = excluded.updated_at
                    """,
                    (canonical_user_id, int(enabled), timestamp, timestamp),
                )
    except (OSError, sqlite3.Error) as exc:
        raise MemorySettingsError("Could not update memory setting") from exc


def is_memory_enabled(user_id: str) -> bool:
    """Return the persisted setting; a missing row defaults to disabled."""
    canonical_user_id = validate_user_id(user_id)
    try:
        with closing(_connect()) as connection:
            with connection:
                _ensure_schema(connection)
            row = connection.execute(
                """
                SELECT memory_enabled
                FROM user_settings
                WHERE user_id = ?
                """,
                (canonical_user_id,),
            ).fetchone()
    except (OSError, sqlite3.Error) as exc:
        raise MemorySettingsError("Could not read memory setting") from exc

    return bool(row[0]) if row is not None else False
