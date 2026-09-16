import os
import re
import json
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

_CONFIG_PATH = Path(__file__).parent.parent / "mcp_servers.json"

def load_mcp_configs() -> dict:
    """Return mcp_servers dict from mcp_servers.json with env vars resolved."""
    raw = json.loads(_CONFIG_PATH.read_text())
    environment = {**os.environ, "CWD": str(Path.cwd())}
    resolved = _resolve_environment_variables(raw, environment)
    return resolved.get("mcp_servers", {})


def _resolve_environment_variables(value: Any, environment: dict[str, str]) -> Any:
    """Resolve ``${NAME}`` placeholders without re-parsing JSON text."""
    if isinstance(value, str):
        return re.sub(
            r"\$\{(\w+)\}",
            lambda match: environment.get(match.group(1), ""),
            value,
        )
    if isinstance(value, list):
        return [_resolve_environment_variables(item, environment) for item in value]
    if isinstance(value, dict):
        return {
            key: _resolve_environment_variables(item, environment)
            for key, item in value.items()
        }
    return value
