import asyncio
import sqlite3
import tempfile
import threading
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import AsyncMock, patch

from rich.console import Console

from ia_claude.long_running_tasks.planning import (
    PlanSpec,
    PlanStore,
    _planner_tool_schema,
    create_plan,
    validate_dag,
)
from ia_claude.long_running_tasks.orchestrator import handle_plan_command


VALID_PLAN = {
    "title": "Build a todo app",
    "tasks": [
        {"id": "api", "prompt": "Build the API", "depends_on": []},
        {"id": "ui", "prompt": "Build the UI", "depends_on": []},
        {
            "id": "connect",
            "prompt": "Connect the UI to the API",
            "depends_on": ["api", "ui"],
        },
    ],
}


class FakePlannerAgent:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.requests = []

    async def ainvoke(self, request):
        self.requests.append(request)
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return {"structured_response": response}


class FakeLlm:
    def __init__(self, responses):
        self.planner = FakePlannerAgent(responses)


class PlanningTests(unittest.TestCase):
    def test_rejects_unknown_fields(self):
        invalid = VALID_PLAN | {"status": "running"}
        with self.assertRaises(ValueError):
            PlanSpec.model_validate(invalid)

    def test_rejects_missing_dependency(self):
        plan_data = {
            "title": "Broken plan",
            "tasks": [
                {"id": "ui", "prompt": "Build UI", "depends_on": ["api"]}
            ],
        }
        with self.assertRaisesRegex(ValueError, "unknown dependencies"):
            validate_dag(PlanSpec.model_validate(plan_data))

    def test_rejects_cycle(self):
        plan_data = {
            "title": "Cyclic plan",
            "tasks": [
                {"id": "a", "prompt": "Task A", "depends_on": ["b"]},
                {"id": "b", "prompt": "Task B", "depends_on": ["a"]},
            ],
        }
        with self.assertRaisesRegex(ValueError, "dependency cycle"):
            validate_dag(PlanSpec.model_validate(plan_data))

    def test_create_plan_retries_once(self):
        llm = FakeLlm([RuntimeError("bad response"), VALID_PLAN])
        with patch(
            "ia_claude.long_running_tasks.planning.create_agent",
            return_value=llm.planner,
        ) as create_agent_mock:
            plan = asyncio.run(create_plan(llm, "Build a todo app"))
        self.assertEqual(plan.title, "Build a todo app")
        self.assertEqual(create_agent_mock.call_args.kwargs["tools"], [])
        response_format = create_agent_mock.call_args.kwargs["response_format"]
        self.assertEqual(response_format.schema, _planner_tool_schema())
        self.assertEqual(len(llm.planner.requests), 2)

    def test_planner_tool_schema_omits_unsupported_gemini_keywords(self):
        def keys(value):
            if isinstance(value, dict):
                yield from value
                for item in value.values():
                    yield from keys(item)
            elif isinstance(value, list):
                for item in value:
                    yield from keys(item)

        schema_keys = set(keys(_planner_tool_schema()))

        self.assertNotIn("$schema", schema_keys)
        self.assertNotIn("additionalProperties", schema_keys)

    def test_approval_saves_plan_and_tasks(self):
        with tempfile.TemporaryDirectory() as directory:
            db_path = Path(directory) / "plans.db"
            output_path = Path(directory) / "output.txt"
            store = PlanStore(db_path)
            with output_path.open("w", encoding="utf-8") as output:
                console = Console(file=output)
                with patch(
                    "ia_claude.long_running_tasks.orchestrator.Prompt.ask",
                    return_value="Y",
                ), patch(
                    "ia_claude.long_running_tasks.orchestrator.execute_plan",
                    new_callable=AsyncMock,
                ) as execute_plan_mock:
                    llm = FakeLlm([VALID_PLAN])
                    with patch(
                        "ia_claude.long_running_tasks.planning.create_agent",
                        return_value=llm.planner,
                    ):
                        decision = asyncio.run(
                            handle_plan_command(
                                llm,
                                "Build a todo app",
                                console,
                                store,
                            )
                        )

            rendered_output = output_path.read_text(encoding="utf-8")

            self.assertEqual(decision, "approved")
            with closing(sqlite3.connect(db_path)) as connection:
                plan_status = connection.execute(
                    "SELECT status FROM plans"
                ).fetchone()[0]
                task_rows = connection.execute(
                    "SELECT id, status, summary FROM tasks ORDER BY id"
                ).fetchall()

            self.assertEqual(plan_status, "approved")
            self.assertEqual(len(task_rows), 3)
            self.assertTrue(all(row[1] == "pending" for row in task_rows))
            self.assertTrue(all(row[2] is None for row in task_rows))
            self.assertIn("Plan status: approved", rendered_output)
            execute_plan_mock.assert_awaited_once()

    def test_store_uses_wal_and_retries_a_contended_write(self):
        with tempfile.TemporaryDirectory() as directory:
            db_path = Path(directory) / "plans.db"
            store = PlanStore(
                db_path,
                busy_timeout=0.01,
                lock_retry_delays=(0.01, 0.05, 0.1),
            )
            store.initialize()

            with closing(sqlite3.connect(db_path)) as connection:
                journal_mode = connection.execute(
                    "PRAGMA journal_mode"
                ).fetchone()[0]
            self.assertEqual(journal_mode, "wal")

            lock_connection = sqlite3.connect(
                db_path,
                timeout=0,
                check_same_thread=False,
            )
            lock_connection.execute("BEGIN IMMEDIATE")
            release_lock = threading.Timer(0.04, lock_connection.rollback)
            release_lock.start()
            try:
                plan_id = store.save(
                    "Build a todo app",
                    PlanSpec.model_validate(VALID_PLAN),
                    "approved",
                )
            finally:
                release_lock.cancel()
                release_lock.join()
                lock_connection.close()

            with closing(sqlite3.connect(db_path)) as connection:
                saved_plan_id = connection.execute(
                    "SELECT id FROM plans"
                ).fetchone()[0]
            self.assertEqual(saved_plan_id, plan_id)

    def test_ready_tasks_advance_only_after_dependencies_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            db_path = Path(directory) / "plans.db"
            store = PlanStore(db_path)
            store.initialize()
            plan_id = store.save(
                "Build a todo app",
                PlanSpec.model_validate(VALID_PLAN),
                "approved",
            )

            self.assertEqual(store.find_ready_tasks(plan_id), ["api", "ui"])

            with closing(sqlite3.connect(db_path)) as connection, connection:
                connection.execute(
                    "UPDATE tasks SET status = 'completed' "
                    "WHERE plan_id = ? AND id IN ('api', 'ui')",
                    (plan_id,),
                )

            self.assertEqual(store.find_ready_tasks(plan_id), ["connect"])

    def test_rejection_is_persisted(self):
        with tempfile.TemporaryDirectory() as directory:
            db_path = Path(directory) / "plans.db"
            store = PlanStore(db_path)
            console = Console(quiet=True)
            with patch(
                "ia_claude.long_running_tasks.orchestrator.Prompt.ask",
                return_value="n",
            ):
                llm = FakeLlm([VALID_PLAN])
                with patch(
                    "ia_claude.long_running_tasks.planning.create_agent",
                    return_value=llm.planner,
                ):
                    decision = asyncio.run(
                        handle_plan_command(
                            llm,
                            "Build a todo app",
                            console,
                            store,
                        )
                    )

            with closing(sqlite3.connect(db_path)) as connection:
                status = connection.execute(
                    "SELECT status FROM plans"
                ).fetchone()[0]

            self.assertEqual(decision, "rejected")
            self.assertEqual(status, "rejected")


if __name__ == "__main__":
    unittest.main()
