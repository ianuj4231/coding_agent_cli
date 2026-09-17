from __future__ import annotations

import json
import sqlite3
import time
from contextlib import closing
from pathlib import Path
from typing import Callable, Literal, TypeVar
from uuid import uuid4

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from rich.console import Console
from rich.table import Table

from ia_claude.long_running_tasks.schemas import MAX_TASKS, PlanSpec, TaskSpec

PLANS_DB_PATH = Path(".ia_claude_all_hidden/plans.db")
DEFAULT_LOCK_RETRY_DELAYS = (0.1, 0.25, 0.5)

WriteResult = TypeVar("WriteResult")


def validate_dag(plan: PlanSpec) -> None:
    """Validate relationships that cannot be checked by field types alone."""

    task_ids = [task.id for task in plan.tasks]
    known_ids = set(task_ids)
    if len(task_ids) != len(known_ids):
        raise ValueError("Task IDs must be unique.")

    for task in plan.tasks:
        missing = set(task.depends_on) - known_ids
        if missing:
            raise ValueError(
                f"Task '{task.id}' has unknown dependencies: "
                f"{', '.join(sorted(missing))}."
            )
        if task.id in task.depends_on:
            raise ValueError(f"Task '{task.id}' cannot depend on itself.")

    remaining = {task.id: set(task.depends_on) for task in plan.tasks}
    while ready := {task_id for task_id, deps in remaining.items() if not deps}:
        for task_id in ready:
            remaining.pop(task_id)
        for dependencies in remaining.values():
            dependencies.difference_update(ready)

    if remaining:
        raise ValueError(
            "The plan contains a dependency cycle involving: "
            f"{', '.join(sorted(remaining))}."
        )


class PlanStore:
    """Small SQLite store for plans awaiting execution."""

    def __init__(
        self,
        db_path: Path = PLANS_DB_PATH,
        *,
        busy_timeout: float = 2,
        lock_retry_delays: tuple[float, ...] = DEFAULT_LOCK_RETRY_DELAYS,
    ) -> None:
        self.db_path = db_path
        self.busy_timeout = busy_timeout
        self.lock_retry_delays = lock_retry_delays

    def _connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.db_path, timeout=self.busy_timeout)
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    @staticmethod
    def _is_lock_error(exc: sqlite3.OperationalError) -> bool:
        error_code = getattr(exc, "sqlite_errorcode", None)
        if isinstance(error_code, int) and error_code & 0xFF in {
            sqlite3.SQLITE_BUSY,
            sqlite3.SQLITE_LOCKED,
        }:
            return True
        message = str(exc).lower()
        return "locked" in message or "busy" in message

    def _write(
        self,
        operation: Callable[[sqlite3.Connection], WriteResult],
    ) -> WriteResult:
        """Run and commit a write, retrying the full transaction on contention."""

        delays = (*self.lock_retry_delays, None)
        for delay in delays:
            try:
                with closing(self._connect()) as connection, connection:
                    return operation(connection)
            except sqlite3.OperationalError as exc:
                if not self._is_lock_error(exc) or delay is None:
                    raise
                time.sleep(delay)

        raise AssertionError("SQLite write retry loop exited unexpectedly.")

    def initialize(self) -> None:
        def create_schema(connection: sqlite3.Connection) -> None:
            # WAL lets readers continue while another CLI instance writes. SQLite
            # still permits only one writer, which _write handles with retries.
            connection.execute("PRAGMA journal_mode = WAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS plans (
                    id TEXT PRIMARY KEY,
                    request TEXT NOT NULL,
                    title TEXT NOT NULL,
                    status TEXT NOT NULL CHECK (
                        status IN ('approved', 'rejected')
                    )
                );

                CREATE TABLE IF NOT EXISTS tasks (
                    plan_id TEXT NOT NULL,
                    id TEXT NOT NULL,
                    prompt TEXT NOT NULL,
                    depends_on TEXT NOT NULL,
                    status TEXT NOT NULL CHECK (
                        status IN ('pending', 'running', 'completed', 'failed')
                    ),
                    summary TEXT,
                    error TEXT,
                    PRIMARY KEY (plan_id, id),
                    FOREIGN KEY (plan_id) REFERENCES plans(id) ON DELETE CASCADE
                );
                """
            )

        self._write(create_schema)

    def save(
        self,
        request: str,
        plan: PlanSpec,
        decision: Literal["approved", "rejected"],
    ) -> str:
        plan_id = uuid4().hex[:12]

        def insert_plan(connection: sqlite3.Connection) -> None:
            connection.execute(
                "INSERT INTO plans (id, request, title, status) VALUES (?, ?, ?, ?)",
                (plan_id, request, plan.title, decision),
            )
            connection.executemany(
                """
                INSERT INTO tasks
                    (plan_id, id, prompt, depends_on, status, summary, error)
                VALUES (?, ?, ?, ?, 'pending', NULL, NULL)
                """,
                [
                    (
                        plan_id,
                        task.id,
                        task.prompt,
                        json.dumps(task.depends_on),
                    )
                    for task in plan.tasks
                ],
            )

        self._write(insert_plan)
        return plan_id

    def claim_ready_tasks(
        self,
        plan_id: str,
        ready_task_ids: list[str],
        limit: int = 1,
    ) -> list[str]:
        """Atomically claim only the supplied tasks that are still ready."""

        if limit < 1:
            raise ValueError("Task claim limit must be at least 1.")

        def claim(connection: sqlite3.Connection) -> list[str]:
            connection.execute("BEGIN IMMEDIATE")
            plan_row = connection.execute(
                "SELECT status FROM plans WHERE id = ?",
                (plan_id,),
            ).fetchone()
            if plan_row is None:
                raise ValueError("Plan not found.")
            if plan_row[0] != "approved":
                return []

            rows = connection.execute(
                "SELECT id, depends_on, status FROM tasks WHERE plan_id = ?",
                (plan_id,),
            ).fetchall()
            completed = {
                task_id for task_id, _, status in rows if status == "completed"
            }
            task_states = {
                task_id: (json.loads(dependencies), status)
                for task_id, dependencies, status in rows
            }
            ready = []
            for task_id in ready_task_ids:
                task_state = task_states.get(task_id)
                if task_state is None:
                    continue
                dependencies, status = task_state
                if status == "pending" and set(dependencies) <= completed:
                    ready.append(task_id)
                if len(ready) == limit:
                    break

            for task_id in ready:
                connection.execute(
                    """
                    UPDATE tasks
                    SET status = 'running'
                    WHERE plan_id = ? AND id = ? AND status = 'pending'
                    """,
                    (plan_id, task_id),
                )
            return ready

        return self._write(claim)

    def get_task_statuses(self, plan_id: str) -> dict[str, str]:
        """Return the current status of every task in a plan."""

        with closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT id, status FROM tasks WHERE plan_id = ?",
                (plan_id,),
            ).fetchall()
        return {task_id: status for task_id, status in rows}

    def recover_running_tasks(self, plan_id: str) -> int:
        """Return interrupted tasks to pending so an approved plan can resume."""

        def recover(connection: sqlite3.Connection) -> int:
            plan_row = connection.execute(
                "SELECT status FROM plans WHERE id = ?",
                (plan_id,),
            ).fetchone()
            if plan_row is None:
                raise ValueError("Plan not found.")
            if plan_row[0] != "approved":
                raise ValueError("Only approved plans can be resumed.")

            cursor = connection.execute(
                """
                UPDATE tasks
                SET status = 'pending', summary = NULL, error = NULL
                WHERE plan_id = ? AND status = 'running'
                """,
                (plan_id,),
            )
            return cursor.rowcount

        return self._write(recover)

    def get_task(self, plan_id: str, task_id: str) -> TaskSpec:
        """Load one task for execution."""

        with closing(self._connect()) as connection:
            row = connection.execute(
                """
                SELECT id, prompt, depends_on
                FROM tasks
                WHERE plan_id = ? AND id = ?
                """,
                (plan_id, task_id),
            ).fetchone()
        if row is None:
            raise ValueError(f"Task '{task_id}' was not found.")
        return TaskSpec(id=row[0], prompt=row[1], depends_on=json.loads(row[2]))

    def get_dependency_summaries(
        self,
        plan_id: str,
        dependency_ids: list[str],
    ) -> dict[str, str]:
        """Load summaries produced by a task's completed dependencies."""

        if not dependency_ids:
            return {}

        placeholders = ", ".join("?" for _ in dependency_ids)
        with closing(self._connect()) as connection:
            rows = connection.execute(
                f"""
                SELECT id, summary
                FROM tasks
                WHERE plan_id = ?
                  AND id IN ({placeholders})
                  AND status = 'completed'
                """,
                (plan_id, *dependency_ids),
            ).fetchall()
        summaries = {task_id: summary or "" for task_id, summary in rows}
        return {
            task_id: summaries[task_id]
            for task_id in dependency_ids
            if task_id in summaries
        }

    def complete_task(self, plan_id: str, task_id: str, summary: str) -> None:
        """Save a successful task result."""

        def complete(connection: sqlite3.Connection) -> None:
            cursor = connection.execute(
                """
                UPDATE tasks
                SET status = 'completed', summary = ?, error = NULL
                WHERE plan_id = ? AND id = ? AND status = 'running'
                """,
                (summary, plan_id, task_id),
            )
            if cursor.rowcount != 1:
                raise ValueError(f"Running task '{task_id}' was not found.")

        self._write(complete)

    def fail_task(self, plan_id: str, task_id: str, error: str) -> None:
        """Save a failed task result."""

        def fail(connection: sqlite3.Connection) -> None:
            cursor = connection.execute(
                """
                UPDATE tasks
                SET status = 'failed', summary = NULL, error = ?
                WHERE plan_id = ? AND id = ? AND status = 'running'
                """,
                (error, plan_id, task_id),
            )
            if cursor.rowcount != 1:
                raise ValueError(f"Running task '{task_id}' was not found.")

        self._write(fail)

    def find_ready_tasks(self, plan_id: str) -> list[str]:
        """Return pending task IDs whose dependencies are completed."""

        with closing(self._connect()) as connection:
            plan_row = connection.execute(
                "SELECT status FROM plans WHERE id = ?",
                (plan_id,),
            ).fetchone()
            if plan_row is None:
                raise ValueError("Plan not found.")
            if plan_row[0] != "approved":
                return []

            rows = connection.execute(
                "SELECT id, depends_on, status FROM tasks WHERE plan_id = ?",
                (plan_id,),
            ).fetchall()

        completed = {task_id for task_id, _, status in rows if status == "completed"}
        return [
            task_id
            for task_id, dependencies, status in rows
            if status == "pending" and set(json.loads(dependencies)) <= completed
        ]


PLANNER_PROMPT = """You are a software planning assistant.
Create a concise, executable DAG plan for the user's request.

Rules:
- Return only data matching the provided schema.
- Give every task a short, unique ID.
- Each task prompt must describe one concrete outcome.
- Use depends_on to express ordering.
- Keep independent tasks independent so they may run in parallel.
- Do not include statuses, results, approval fields, or circular dependencies.
- Do not perform the work; only plan it.

User request:
{request}
"""


def _planner_tool_schema() -> dict:
    """Return PlanSpec's schema without Gemini-unsupported annotations."""

    def clean(value):
        if isinstance(value, dict):
            return {
                key: clean(item)
                for key, item in value.items()
                if key not in {"$schema", "additionalProperties"}
            }
        if isinstance(value, list):
            return [clean(item) for item in value]
        return value

    return clean(PlanSpec.model_json_schema())


async def create_plan(llm, request: str) -> PlanSpec:
    """Generate and fully validate a plan, retrying malformed output once."""

    # Force tool-based structured output instead of relying on the provider's
    # native JSON-schema implementation. Some Gemini model/schema combinations
    # are advertised as supporting native structured output but reject the
    # generated request with INVALID_ARGUMENT. The schema tool is internal to
    # the response strategy; the planner still has no executable tools.
    planner_agent = create_agent(
        model=llm,
        tools=[],
        response_format=ToolStrategy(_planner_tool_schema()),
    )
    last_error: Exception | None = None
    for attempt in range(2):
        prompt = PLANNER_PROMPT.format(request=request)
        if attempt:
            prompt += (
                "\nYour previous response was invalid. Return a corrected plan that "
                f"also fixes this validation error: {last_error}"
            )
        try:
            state = await planner_agent.ainvoke(
                {"messages": [{"role": "user", "content": prompt}]}
            )
            plan = PlanSpec.model_validate(state.get("structured_response"))
            validate_dag(plan)
            return plan
        except Exception as exc:
            last_error = exc

    raise ValueError(f"The planner did not return a valid DAG: {last_error}")


def render_plan(console: Console, plan: PlanSpec) -> None:
    table = Table(
        title=f"Plan: {plan.title}",
        title_style="bold cyan",
        header_style="bold magenta",
        border_style="bright_blue",
        show_lines=True,
    )
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("Task", style="white")
    table.add_column("Depends on", style="yellow")
    table.add_column("Status", style="bright_black", no_wrap=True)

    for task in plan.tasks:
        table.add_row(
            task.id,
            task.prompt,
            ", ".join(task.depends_on) or "-",
            "pending",
        )

    console.print()
    console.print(table)



"""
Before/during the LLM call:
planner_agent = create_agent(..., tools=[], response_format=ToolStrategy(PlanSpec))
This forces schema output through a model tool call instead of Gemini's native
structured-output request format.

After receiving the response:
plan = PlanSpec.model_validate(state.get("structured_response"))
This independently checks that the returned data genuinely matches the schema.
Then:
validate_dag(plan)
checks rules that the basic schema cannot enforce:
Task IDs are unique.
Dependencies exist.
Tasks do not depend on themselves.
The graph has no cycles.
So the flow is:
Ask LLM for PlanSpec
→ receive response
→ validate it as PlanSpec
→ validate DAG relationships
→ accept or retry
The first step guides the unreliable LLM. The later validation steps prevent malformed or logically invalid plans from entering the database.

"""
