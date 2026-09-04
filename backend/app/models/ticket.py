from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field

from .user_context import UserContext, UserRole


TicketStatus = Literal[
    "new",
    "pending",
    "sent_email",
    "sent_telegram",
    "sent_bitrix",
    "delivery_failed",
    "failed",
    "closed",
]


class Ticket(BaseModel):
    idempotency_key: str | None = None
    id: str = ""
    status: TicketStatus = "new"
    topic: str
    description: str
    contact: str | None = None
    role: UserRole = "guest"
    user_id: str | None = None
    lot_id: str | None = None
    payment_id: str | None = None
    session_id: str
    page_type: str | None = None
    dialog_history: list[dict] = Field(default_factory=list)
    attachments: list[str] = Field(default_factory=list)
    category: str | None = None
    priority: str = "normal"
    scenario_id: str | None = None
    source_message_id: str | None = None
    collected_fields: dict[str, str] = Field(default_factory=dict)
    delivery_attempts: int = 0
    next_delivery_attempt_at: datetime | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class TicketCreateRequest(BaseModel):
    idempotency_key: str | None = Field(default=None, min_length=1, max_length=128)
    topic: str = Field(min_length=1)
    description: str = Field(min_length=1)
    contact: str | None = None
    session_id: str
    context: UserContext = Field(default_factory=UserContext)
    lot_id: str | None = None
    payment_id: str | None = None
    attachments: list[str] = Field(default_factory=list)
    category: str | None = None
    priority: str = "normal"
    scenario_id: str | None = None
    source_message_id: str | None = None
    request_callback: bool = False
    preferred_callback_time: str | None = None
    trusted_context_token: str | None = None
