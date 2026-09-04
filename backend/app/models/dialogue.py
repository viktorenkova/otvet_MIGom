"""Internal dialogue types. Routing scores and trusted tokens never live here."""
from typing import Literal
from pydantic import BaseModel, Field


class QueryUnderstanding(BaseModel):
    goal: str = ""
    intent: str = "unknown"
    objects: list[str] = Field(default_factory=list)
    operations: list[str] = Field(default_factory=list)
    states: list[str] = Field(default_factory=list)
    stage: str = ""
    actor: str | None = None
    entities: dict[str, str] = Field(default_factory=dict)
    background_events: list[str] = Field(default_factory=list)
    secondary_requests: list[str] = Field(default_factory=list)
    negations: list[str] = Field(default_factory=list)
    unknown_fields: list[str] = Field(default_factory=list)


class DialogueState(BaseModel):
    schema_version: Literal[1] = 1
    version: int = 0
    subject: str = "guest"
    active: QueryUnderstanding = Field(default_factory=QueryUnderstanding)
    active_scenario_id: str | None = None
    pending_requests: list[str] = Field(default_factory=list)
    expected_candidates: list[dict[str, str]] = Field(default_factory=list)
    expected_field: str | None = None
    pending_action_id: str | None = None
    previous_action: str | None = None
    previous_message_id: str | None = None
    clarification_attempts: int = 0
    status: Literal["idle", "active", "clarifying", "answered", "manual_help"] = "idle"


class DialogueTurn(BaseModel):
    understanding: QueryUnderstanding
    state: DialogueState
    transition: Literal["new", "continue", "correct", "switch", "next", "repeat"]
    search_message: str
    selected_scenario_id: str | None = None
    category_choice: dict[str, str] | None = None
    resume_action_id: str | None = None
    service_reply: Literal["clarify_entity", "manual_help"] | None = None
    used_context: list[str] = Field(default_factory=list)
