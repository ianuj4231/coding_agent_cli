from __future__ import annotations

import os
import re
from pathlib import Path

import yaml
from dotenv import load_dotenv


PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent

CONFIG_PATH = PACKAGE_DIR / "config.yaml"
ENV_PATH = PROJECT_ROOT / ".env"

_ENV_VAR_PATTERN = re.compile(
    r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}"
)


def _expand_environment_variables(
    text: str,
) -> str:
    """
    Expand ${VARIABLE} references using environment variables.

    Raises a clear error when a referenced variable is missing.
    """

    def replace(match: re.Match[str]) -> str:
        variable_name = match.group(1)

        value = os.getenv(
            variable_name
        )

        if value is None:
            raise RuntimeError(
                "Missing required environment variable: "
                f"{variable_name}"
            )

        return value

    return _ENV_VAR_PATTERN.sub(
        replace,
        text,
    )


def load_config() -> dict:
    """
    Load application settings from config.yaml.

    The project .env file is loaded first so config.yaml may safely
    reference secrets using syntax such as ${REDIS_URL}.
    """

    load_dotenv(
        ENV_PATH
    )

    raw_config = CONFIG_PATH.read_text(
        encoding="utf-8"
    )

    expanded_config = (
        _expand_environment_variables(
            raw_config
        )
    )

    loaded_config = yaml.safe_load(
        expanded_config
    )

    if not isinstance(
        loaded_config,
        dict,
    ):
        raise ValueError(
            f"Invalid configuration in {CONFIG_PATH}: "
            "expected a YAML mapping."
        )

    return loaded_config


config = load_config()