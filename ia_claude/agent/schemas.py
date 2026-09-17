"""Structured response schemas for the coding agent."""

from pydantic import BaseModel, Field


class AgentResponse(BaseModel):
    """Final response returned by the coding agent."""

    answer: str = Field(
        description="The complete final answer to show to the user."
    )
