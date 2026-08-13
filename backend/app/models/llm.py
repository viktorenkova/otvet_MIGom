from pydantic import BaseModel, Field

from .user_context import UserRole


class LLMRequest(BaseModel):
    prompt: str
    fallback_text: str
    provider: str
    model: str
    fallback_model: str | None = None
    task_type: str
    session_id: str
    user_role: UserRole
    escalation_required: bool = False
    safety_flags: list[str] = Field(default_factory=list)


class LLMResult(BaseModel):
    text: str
    provider: str
    model: str
    task_type: str
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0
    latency_ms: int = 0
    success: bool = True
    error: str | None = None
