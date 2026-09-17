"""Pydantic schemas used by the long-running plan workflow."""

from pydantic import BaseModel, ConfigDict, Field, field_validator


MAX_TASKS = 20


class TaskSpec(BaseModel):
    """One planner-created unit of work."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, max_length=50)
    prompt: str = Field(min_length=1, max_length=2_000)
    depends_on: list[str] = Field(default_factory=list, max_length=MAX_TASKS)

    @field_validator("id", "prompt")
    @classmethod
    def strip_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value

    @field_validator("depends_on")
    @classmethod
    def clean_dependencies(cls, values: list[str]) -> list[str]:
        cleaned = [value.strip() for value in values]
        if any(not value for value in cleaned):
            raise ValueError("dependency IDs must not be blank")
        if len(cleaned) != len(set(cleaned)):
            raise ValueError("dependency IDs must be unique")
        return cleaned


class PlanSpec(BaseModel):
    """Strict structured output expected from the planner model."""

    model_config = ConfigDict(extra="forbid")

    title: str = Field(min_length=1, max_length=120)
    tasks: list[TaskSpec] = Field(min_length=1, max_length=MAX_TASKS)

    @field_validator("title")
    @classmethod
    def strip_title(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value


class SubtaskResult(BaseModel):
    """Validated final output from one specialist agent."""

    model_config = ConfigDict(extra="forbid")

    summary: str = Field(min_length=1, max_length=4_000)


class JudgeVerdict(BaseModel):
    """Structured decision from the independent task judge."""

    model_config = ConfigDict(extra="forbid")

    passed: bool
    reason: str = Field(min_length=1, max_length=1_000)
