"""DANGER: a real LLM controls refund calls and retries after a timeout.

OPENROUTER_API_KEY is required. The payment provider is simulated locally:
its first call moves fake money but returns TIMEOUT, reproducing the ambiguity
from the video. No real payment is made.
"""

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.tools import tool

from ia_claude.llm.factory import get_llm


load_dotenv()

ORDER_AMOUNT = 100
total_refunded = 0
provider_calls = 0


@tool
def issue_refund(order_id: str, amount: int) -> str:
    """Issue a refund. Retry if the provider times out."""

    global total_refunded, provider_calls
    provider_calls += 1

    # The provider processes every request because there is no idempotency key.
    total_refunded += amount

    if provider_calls == 1:
        # Money moved, but the success response never reached our backend.
        return "TIMEOUT: no response received; outcome unknown"

    return f"SUCCESS: refund transaction TX_{provider_calls} created"


agent = create_agent(
    model=get_llm(),
    tools=[issue_refund],  # Dangerous tool is available on every model call.
    system_prompt=(
        "You process refund requests. Use issue_refund when asked. "
        "If a tool times out, retry the refund."
    ),
)


if __name__ == "__main__":
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Refund $100 for order A1234.",
                }
            ]
        }
    )

    print("Final model answer:", result["messages"][-1].content)
    print("Provider calls:", provider_calls)
    print("Total fake money refunded: $", total_refunded, sep="")

