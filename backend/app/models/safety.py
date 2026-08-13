from datetime import datetime, timezone

from pydantic import BaseModel, Field


class SafetyCheckResult(BaseModel):
    allowed: bool = True
    categories: list[str] = Field(default_factory=list)
    answer_override: str | None = None
    answer_prefix: str | None = None
    needs_review: bool = False


class SafetyEvent(BaseModel):
    session_id: str
    user_id: str | None = None
    message: str
    category: str
    answer: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    needs_review: bool = False
    ticket_created: bool = False
