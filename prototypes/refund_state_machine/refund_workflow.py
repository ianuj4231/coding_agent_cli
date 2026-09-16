"""SAFE: a real LLM interprets intent; code controls payment and retry.

OPENROUTER_API_KEY is required. SQLite persists workflow facts. The payment
provider is simulated locally and no real payment is made.
"""

import sqlite3
import uuid
from pathlib import Path
from typing import Literal, TypedDict

from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

from ia_claude.llm.factory import get_llm


load_dotenv()

DB_PATH = Path(__file__).with_name("refund_demo.sqlite")
ORDERS = {"A1234": {"amount": 100, "eligible": True}}

# The fake provider remembers idempotency keys just like a payment API would.
provider_refunds: dict[str, str] = {}
provider_calls = 0
total_refunded = 0


class State(TypedDict, total=False):
    message: str
    order_id: str
    intent: str
    approved: bool
    refund_id: str
    result: str


class Intent(BaseModel):
    intent: Literal["refund", "other"]


def save(refund_id: str, order_id: str, status: str) -> None:
    """Persist the objective refund status."""

    with sqlite3.connect(DB_PATH) as db:
        db.execute(
            "CREATE TABLE IF NOT EXISTS refunds "
            "(refund_id TEXT PRIMARY KEY, order_id TEXT, status TEXT)"
        )
        db.execute(
            "INSERT OR REPLACE INTO refunds VALUES (?, ?, ?)",
            (refund_id, order_id, status),
        )


def find_refund(order_id: str) -> tuple[str, str] | None:
    """Return this order's existing (idempotency_key, status), if any."""

    with sqlite3.connect(DB_PATH) as db:
        db.execute(
            "CREATE TABLE IF NOT EXISTS refunds "
            "(refund_id TEXT PRIMARY KEY, order_id TEXT, status TEXT)"
        )
        return db.execute(
            "SELECT refund_id, status FROM refunds "
            "WHERE order_id = ? ORDER BY rowid DESC LIMIT 1",
            (order_id,),
        ).fetchone()


def understand(state: State) -> State:
    """Real LLM call. It receives no tools and can only classify intent."""

    model = get_llm().with_structured_output(Intent)
    answer = model.invoke("Classify as refund or other: " + state["message"])
    return {"intent": answer.intent}


def policy(state: State) -> State:
    order = ORDERS.get(state["order_id"])
    return {"approved": bool(state["intent"] == "refund" and order and order["eligible"])}


def route(state: State) -> Literal["refund", "deny"]:
    return "refund" if state["approved"] is True else "deny"


def fake_provider_refund(order_id: str, amount: int, idempotency_key: str) -> str:
    """Process once per key; deliberately lose the first success response."""

    global provider_calls, total_refunded
    provider_calls += 1

    if idempotency_key in provider_refunds:
        return "SUCCESS: existing transaction " + provider_refunds[idempotency_key]

    transaction_id = "TX_771"
    provider_refunds[idempotency_key] = transaction_id
    total_refunded += amount

    if provider_calls == 1:
        return "TIMEOUT: no response received; outcome unknown"
    return "SUCCESS: transaction " + transaction_id


def refund(state: State) -> State:
    """Plain code owns the irreversible action; no LLM call exists here."""

    existing = find_refund(state["order_id"])

    if existing is None:
        # First attempt: create one stable key and persist it BEFORE calling.
        refund_id = str(uuid.uuid4())
        save(refund_id, state["order_id"], "attempt_started_outcome_unknown")
    else:
        refund_id, status = existing

        if status == "succeeded":
            return {
                "refund_id": refund_id,
                "result": "Already succeeded; no provider call needed.",
            }

        if status == "failed":
            return {
                "refund_id": refund_id,
                "result": "Previously failed permanently; not retrying.",
            }

        # status == attempt_started_outcome_unknown:
        # Retry/reconcile using the SAME idempotency key.

    order = ORDERS[state["order_id"]]
    result = fake_provider_refund(state["order_id"], order["amount"], refund_id)

    if result.startswith("TIMEOUT"):
        save(refund_id, state["order_id"], "attempt_started_outcome_unknown")

        # This retry uses the SAME key. A real provider may also offer a status
        # lookup endpoint instead of requiring an immediate retry.
        result = fake_provider_refund(state["order_id"], order["amount"], refund_id)

    if result.startswith("SUCCESS"):
        save(refund_id, state["order_id"], "succeeded")
    else:
        # Only a definite, non-retryable provider rejection should be "failed".
        save(refund_id, state["order_id"], "failed")

    return {"refund_id": refund_id, "result": result}


def deny(state: State) -> State:
    return {"result": "DENIED"}


def build_graph():
    graph = StateGraph(State)
    graph.add_node("understand", understand)
    graph.add_node("policy", policy)
    graph.add_node("refund", refund)
    graph.add_node("deny", deny)
    graph.add_edge(START, "understand")
    graph.add_edge("understand", "policy")
    graph.add_conditional_edges("policy", route)
    graph.add_edge("refund", END)
    graph.add_edge("deny", END)
    return graph.compile()


if __name__ == "__main__":
    final = build_graph().invoke(
        {"message": "Refund $100 for order A1234.", "order_id": "A1234"}
    )
    print("Workflow result:", final["result"])
    print("Provider calls (initial + retry):", provider_calls)
    print("Total fake money refunded: $", total_refunded, sep="")
    print("Persisted state DB:", DB_PATH)
