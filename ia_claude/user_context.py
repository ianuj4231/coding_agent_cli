"""Startup user identity used to isolate Qdrant data."""

from __future__ import annotations

import uuid

from rich.console import Console
from rich.prompt import Prompt


DEFAULT_USER_ID = "3b2eb43f-3a0e-41d7-82b7-cde6dc9357d8"


def validate_user_id(value: str) -> str:
    """Return a canonical UUID string, or raise ``ValueError``."""
    if not value or not value.strip():
        raise ValueError("user_id is required")
    return str(uuid.UUID(value.strip()))


def prompt_for_user_id(console: Console) -> str:
    """Prompt until the user supplies a valid UUID."""
    while True:
        try:
            raw_value = Prompt.ask("[bold]Enter user ID (UUID)[/bold]")
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Goodbye![/dim]")
            raise SystemExit(0) from None

        try:
            user_id = validate_user_id(raw_value)
        except (ValueError, AttributeError):
            console.print("[red]Invalid UUID. Please enter a valid UUID.[/red]")
            continue

        console.print(f"[green]User ID accepted:[/green] {user_id}")
        return user_id
