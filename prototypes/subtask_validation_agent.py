"""Prototype: execute one task with lightweight, task-level validation.

This file is intentionally not connected to the production plan workflow.
"""

import asyncio
import sys
from dataclasses import dataclass
from pathlib import Path

from langchain.agents import create_agent
from pydantic import BaseModel, ConfigDict, Field

from ia_claude.llm.factory import get_llm
from ia_claude.long_running_tasks.schemas import TaskSpec
from ia_claude.tools.filesystem_tools import (
    append_file,
    file_exists,
    list_directory,
    read_file,
    write_file,
)
from ia_claude.tools.terminal_tools import run_command, run_in_directory


class CheckResult(BaseModel):
    """One check performed by the specialist after completing its task."""

    model_config = ConfigDict(extra="forbid")

    check: str = Field(min_length=1, max_length=300)
    passed: bool
    details: str = Field(min_length=1, max_length=1_000)


class SubtaskResult(BaseModel):
    """Validated final response from the specialist agent."""

    model_config = ConfigDict(extra="forbid")

    summary: str = Field(min_length=1, max_length=4_000)
    files_changed: list[str] = Field(default_factory=list)
    checks: list[CheckResult] = Field(min_length=1)


@dataclass(frozen=True)
class ValidationCommand:
    """One deterministic command required by the validation policy."""

    name: str
    command: tuple[str, ...]


DEFAULT_VALIDATION_COMMANDS = (
    ValidationCommand(
        name="Python syntax",
        command=(sys.executable, "-m", "compileall", "-q", "."),
    ),
    ValidationCommand(
        name="Unit tests",
        command=(sys.executable, "-m", "unittest", "discover", "-s", "tests"),
    ),
    ValidationCommand(
        name="Static security analysis",
        command=(sys.executable, "-m", "bandit", "-q", "-r", "."),
    ),
    ValidationCommand(
        name="Dependency vulnerability scan",
        command=(sys.executable, "-m", "pip_audit"),
    ),
    ValidationCommand(
        name="Secret scan",
        command=("gitleaks", "detect", "--no-git", "--source", "."),
    ),
)


SUBTASK_SYSTEM_PROMPT = """You are a focused specialist coding worker.
Complete only the single assigned task and preserve unrelated work.

Before finishing, perform at least one relevant check using the available tools.
Examples include running a focused test, building the changed code, validating
syntax, reading the resulting file, or confirming that an expected file exists.
If a check fails, try to fix the problem and run the check again.

Your final response must follow the required SubtaskResult schema. Report checks
truthfully. A failed final check must have passed=false.
"""


async def run_validated_subtask_agent(
    task: TaskSpec,
    dependency_summaries: dict[str, str],
) -> SubtaskResult:
    """Execute one task and reject its result when a reported check failed."""

    dependency_context = "\n".join(
        f"- {task_id}: {summary}"
        for task_id, summary in dependency_summaries.items()
    ) or "- None; this task has no dependencies."

    agent = create_agent(
        model=get_llm(),
        tools=[
            run_command,
            run_in_directory,
            read_file,
            write_file,
            append_file,
            list_directory,
            file_exists,
        ],
        system_prompt=SUBTASK_SYSTEM_PROMPT,
        response_format=SubtaskResult,
    )

    state = await agent.ainvoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        f"Task ID: {task.id}\n"
                        f"Task: {task.prompt}\n\n"
                        "Completed dependency results:\n"
                        f"{dependency_context}\n\n"
                        "Complete and validate this task only."
                    ),
                }
            ]
        }
    )

    result = SubtaskResult.model_validate(state.get("structured_response"))
    failed_checks = [check.check for check in result.checks if not check.passed]
    if failed_checks:
        raise ValueError(
            "Subtask validation failed: " + ", ".join(failed_checks)
        )
    return result


async def run_deterministic_checks(
    project_directory: Path,
    commands: tuple[ValidationCommand, ...] = DEFAULT_VALIDATION_COMMANDS,
    timeout_seconds: int = 300,
) -> list[CheckResult]:
    """Run mandatory checks outside the author agent and return their evidence."""

    project_directory = project_directory.resolve()
    results: list[CheckResult] = []

    for validation in commands:
        try:
            process = await asyncio.create_subprocess_exec(
                *validation.command,
                cwd=project_directory,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=timeout_seconds,
                )
            except TimeoutError:
                process.kill()
                await process.communicate()
                results.append(
                    CheckResult(
                        check=validation.name,
                        passed=False,
                        details=f"Timed out after {timeout_seconds} seconds.",
                    )
                )
                continue

            output = (stdout + stderr).decode(errors="replace").strip()
            results.append(
                CheckResult(
                    check=validation.name,
                    passed=process.returncode == 0,
                    details=(output or f"Exited with code {process.returncode}.")[-1_000:],
                )
            )
        except FileNotFoundError:
            results.append(
                CheckResult(
                    check=validation.name,
                    passed=False,
                    details=f"Required command was not found: {validation.command[0]}",
                )
            )

    return results


async def execute_and_validate_subtask(
    task: TaskSpec,
    dependency_summaries: dict[str, str],
    project_directory: Path,
) -> SubtaskResult:
    """Run the author agent, then enforce independent deterministic checks."""

    result = await run_validated_subtask_agent(task, dependency_summaries)
    independent_checks = await run_deterministic_checks(project_directory)
    failed_checks = [check.check for check in independent_checks if not check.passed]
    if failed_checks:
        raise ValueError(
            "Independent validation failed: " + ", ".join(failed_checks)
        )

    return result.model_copy(
        update={"checks": [*result.checks, *independent_checks]}
    )
