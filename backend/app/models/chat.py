from typing import Any, Literal
from uuid import uuid4

from pydantic import AliasChoices, BaseModel, Field

from .user_context import UserContext, UserRole


class LegacyChatRequest(BaseModel):
    message: str


class LegacyChatResponse(BaseModel):
    answer: str
    needs_escalation: bool


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    session_id: str = Field(default_factory=lambda: str(uuid4()))
    context: UserContext = Field(default_factory=UserContext, validation_alias=AliasChoices("context", "user_context"))
    contact: str | None = None
    attachments: list[str] = Field(default_factory=list)
    consent_to_ticket: bool = False
    selected_action_id: str | None = None
    conversation_turn_id: str | None = None
    trusted_context_token: str | None = None

    model_config = {"populate_by_name": True}


class TemplateLink(BaseModel):
    label: str
    url: str


ChatActionType = Literal[
    "answer",
    "clarify",
    "navigate",
    "fetch_status",
    "provide_document",
    "open_ticket",
    "request_callback",
    "handoff",
]


class ChatAction(BaseModel):
    id: str
    type: ChatActionType
    label: str
    scenario_id: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    requires_auth: bool = False
    requires_confirmation: bool = False


Resolution = Literal[
    "answered",
    "clarified",
    "status",
    "escalated",
    "out_of_scope",
]


class ChatResponse(BaseModel):
    session_id: str | None = None
    message_id: str | None = None
    answer: str
    intent: str
    scenario_id: str | None = None
    resolution: Resolution = "answered"
    role: UserRole
    needs_ticket: bool
    ticket_id: str | None = None
    safety_categories: list[str] = Field(default_factory=list)
    safety_flags: list[str] = Field(default_factory=list)
    suggested_fields: list[str] = Field(default_factory=list)
    ticket_required_fields: list[str] = Field(default_factory=list)
    attachments: list[str] = Field(default_factory=list)
    template_links: list[TemplateLink] = Field(default_factory=list)
    model_used: str = "mock"
    confidence_level: str = "high"
    clarifying_options: list[str] = Field(default_factory=list)
    actions: list[ChatAction] = Field(default_factory=list)
    used_context: list[str] = Field(default_factory=list)
    data_freshness: str | None = None
    action: str = "answer"


class ChatFeedbackRequest(BaseModel):
    session_id: str
    rating: int | None = Field(default=None, ge=1, le=5)
    comment: str | None = None
    message_id: str | None = None
