from functools import lru_cache
import os
from pathlib import Path
from urllib.parse import urlparse

from pydantic import BaseModel
from typing import Literal


class Settings(BaseModel):
    app_name: str = "migtorg-chatbot"
    app_environment: str = "development"
    deploy_version: str = "local"
    debug: bool = False
    cors_allowed_origins: list[str] = ["*"]
    database_path: str = "migtorg_chatbot.sqlite3"
    dialogue_state_enabled: bool = False
    answer_assembly_enabled: bool = False
    chat_max_concurrency: int = 8
    routing_architecture: Literal["control", "local"] = "control"
    llm_understanding_enabled: bool = False
    architecture_experiment: bool = False
    llm_enabled: bool = False
    llm_provider: str = "mock"
    llm_environment: str = "dev"
    llm_primary_model: str = "mock/safe-rules"
    llm_fallback_model: str = "mock/safe-rules"
    llm_reasoning_effort: str = "none"
    llm_request_timeout_seconds: int = 30
    llm_total_timeout_seconds: int = 8
    llm_max_output_tokens: int = 300
    llm_max_concurrency: int = 8
    llm_circuit_failure_threshold: int = 3
    llm_circuit_cooldown_seconds: int = 60
    llm_dev_budget_usd: float = 1.0
    llm_production_budget_usd: float = 25.0
    llm_daily_budget_usd: float = 5.0
    llm_budget_warning_pct: float = 80.0
    llm_rollout_percentage: int = 0
    llm_input_cost_per_million_usd: float = 0.0
    llm_output_cost_per_million_usd: float = 0.0
    litellm_proxy_url: str = ""
    litellm_api_key: str = ""
    qwen_base_url: str = ""
    qwen_api_key: str = ""
    langfuse_enabled: bool = False
    langfuse_host: str = ""
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    quality_report_token: str = ""
    ticket_email_enabled: bool = False
    ticket_email_to: str = "support@migtorg.example"
    ticket_email_from: str = "bot@migtorg.example"
    smtp_host: str = ""
    smtp_port: int = 587
    smtp_user: str = ""
    smtp_password: str = ""
    knowledge_v2_enabled: bool = True
    knowledge_v2_shadow_mode: bool = False
    trusted_context_secret: str = ""
    trusted_context_issuer: str = "migtorg-site"
    internal_status_api_enabled: bool = False
    internal_status_api_url: str = ""
    internal_status_timeout_seconds: int = 5

    @property
    def knowledge_root(self) -> Path:
        return Path(__file__).resolve().parents[2] / "knowledge"

    @property
    def static_root(self) -> Path:
        return Path(__file__).resolve().parents[1] / "static"

    @property
    def widget_root(self) -> Path:
        frontend_root = Path(__file__).resolve().parents[2] / "frontend"
        nested_widget_root = frontend_root / "chat-widget"

        # Keep compatibility with both repository layouts:
        # frontend/chat-widget/* (original) and frontend/* (flat).
        if (nested_widget_root / "index.html").is_file():
            return nested_widget_root
        return frontend_root

    @property
    def active_llm_budget_usd(self) -> float:
        if self.llm_environment == "production":
            return self.llm_production_budget_usd
        return self.llm_dev_budget_usd

    @property
    def active_llm_monthly_budget_usd(self) -> float:
        return self.active_llm_budget_usd


def _is_https_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme == "https" and bool(parsed.netloc)


def validate_llm_runtime(settings: Settings) -> None:
    if not (settings.llm_enabled or settings.llm_understanding_enabled):
        return
    if settings.llm_provider not in {"qwen", "litellm"}:
        raise ValueError("LLM_ENABLED requires provider qwen or litellm")
    endpoint = settings.qwen_base_url if settings.llm_provider == "qwen" else settings.litellm_proxy_url
    api_key = settings.qwen_api_key if settings.llm_provider == "qwen" else settings.litellm_api_key
    if not _is_https_url(endpoint) or not api_key:
        raise ValueError("Enabled LLM requires an HTTPS endpoint and backend-only API key")
    if settings.llm_input_cost_per_million_usd <= 0 or settings.llm_output_cost_per_million_usd <= 0:
        raise ValueError("Enabled LLM requires positive token pricing for enforceable budgets")
    if settings.llm_daily_budget_usd <= 0 or settings.active_llm_monthly_budget_usd <= 0:
        raise ValueError("Enabled LLM requires positive daily and monthly budgets")
    if not 0 < settings.llm_total_timeout_seconds <= settings.llm_request_timeout_seconds * 2:
        raise ValueError("LLM total timeout must be positive and bounded")
    if settings.llm_max_concurrency <= 0 or settings.llm_circuit_failure_threshold <= 0:
        raise ValueError("LLM concurrency and circuit threshold must be positive")
    if settings.llm_rollout_percentage not in {0, 5, 25, 50, 100}:
        raise ValueError("LLM rollout percentage must be one of 0, 5, 25, 50, 100")


def _bool_from_env(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


@lru_cache
def get_settings() -> Settings:
    cors_allowed_origins = [
        origin.strip()
        for origin in os.getenv("CORS_ALLOWED_ORIGINS", "*").split(",")
        if origin.strip()
    ]
    settings = Settings(
        app_name=os.getenv("APP_NAME", "migtorg-chatbot"),
        app_environment=os.getenv("APP_ENVIRONMENT", "development"),
        deploy_version=os.getenv("DEPLOY_VERSION", "local"),
        debug=_bool_from_env(os.getenv("DEBUG"), False),
        cors_allowed_origins=cors_allowed_origins or ["*"],
        dialogue_state_enabled=_bool_from_env(os.getenv("DIALOGUE_STATE_ENABLED")),
        answer_assembly_enabled=_bool_from_env(os.getenv("ANSWER_ASSEMBLY_ENABLED")),
        chat_max_concurrency=max(1, int(os.getenv("CHAT_MAX_CONCURRENCY", "8"))),
        routing_architecture=os.getenv("ROUTING_ARCHITECTURE", "control"),
        llm_understanding_enabled=os.getenv("LLM_UNDERSTANDING_ENABLED", "false").lower() == "true",
        architecture_experiment=os.getenv("ARCHITECTURE_EXPERIMENT", "true").lower() == "true",
        llm_enabled=_bool_from_env(os.getenv("LLM_ENABLED"), False),
        llm_provider=os.getenv("LLM_PROVIDER", "mock"),
        llm_environment=os.getenv("LLM_ENVIRONMENT", "dev"),
        llm_primary_model=os.getenv("LLM_PRIMARY_MODEL", "mock/safe-rules"),
        llm_fallback_model=os.getenv("LLM_FALLBACK_MODEL", "mock/safe-rules"),
        llm_reasoning_effort=os.getenv("LLM_REASONING_EFFORT", "none"),
        llm_request_timeout_seconds=int(os.getenv("LLM_REQUEST_TIMEOUT_SECONDS", "30")),
        llm_total_timeout_seconds=int(os.getenv("LLM_TOTAL_TIMEOUT_SECONDS", "8")),
        llm_max_output_tokens=int(os.getenv("LLM_MAX_OUTPUT_TOKENS", "300")),
        llm_max_concurrency=int(os.getenv("LLM_MAX_CONCURRENCY", "8")),
        llm_circuit_failure_threshold=int(os.getenv("LLM_CIRCUIT_FAILURE_THRESHOLD", "3")),
        llm_circuit_cooldown_seconds=int(os.getenv("LLM_CIRCUIT_COOLDOWN_SECONDS", "60")),
        llm_dev_budget_usd=float(os.getenv("LLM_DEV_BUDGET_USD", "1.0")),
        llm_production_budget_usd=float(os.getenv("LLM_PRODUCTION_BUDGET_USD", "25.0")),
        llm_daily_budget_usd=float(os.getenv("LLM_DAILY_BUDGET_USD", "5.0")),
        llm_budget_warning_pct=float(os.getenv("LLM_BUDGET_WARNING_PCT", "80.0")),
        llm_rollout_percentage=int(os.getenv("LLM_ROLLOUT_PERCENTAGE", "0")),
        llm_input_cost_per_million_usd=float(os.getenv("LLM_INPUT_COST_PER_MILLION_USD", "0")),
        llm_output_cost_per_million_usd=float(os.getenv("LLM_OUTPUT_COST_PER_MILLION_USD", "0")),
        litellm_proxy_url=os.getenv("LITELLM_PROXY_URL", ""),
        litellm_api_key=os.getenv("LITELLM_API_KEY", ""),
        qwen_base_url=os.getenv("QWEN_BASE_URL", ""),
        qwen_api_key=os.getenv("QWEN_API_KEY", os.getenv("DASHSCOPE_API_KEY", "")),
        langfuse_enabled=_bool_from_env(os.getenv("LANGFUSE_ENABLED"), False),
        langfuse_host=os.getenv("LANGFUSE_HOST", ""),
        langfuse_public_key=os.getenv("LANGFUSE_PUBLIC_KEY", ""),
        langfuse_secret_key=os.getenv("LANGFUSE_SECRET_KEY", ""),
        quality_report_token=os.getenv("QUALITY_REPORT_TOKEN", ""),
        database_path=os.getenv("DATABASE_PATH", "migtorg_chatbot.sqlite3"),
        ticket_email_enabled=_bool_from_env(os.getenv("TICKET_EMAIL_ENABLED"), False),
        ticket_email_to=os.getenv("TICKET_EMAIL_TO", "support@migtorg.example"),
        ticket_email_from=os.getenv("TICKET_EMAIL_FROM", "bot@migtorg.example"),
        smtp_host=os.getenv("SMTP_HOST", ""),
        smtp_port=int(os.getenv("SMTP_PORT", "587")),
        smtp_user=os.getenv("SMTP_USER", ""),
        smtp_password=os.getenv("SMTP_PASSWORD", ""),
        knowledge_v2_enabled=_bool_from_env(os.getenv("KNOWLEDGE_V2_ENABLED"), True),
        knowledge_v2_shadow_mode=_bool_from_env(os.getenv("KNOWLEDGE_V2_SHADOW_MODE"), False),
        trusted_context_secret=os.getenv("TRUSTED_CONTEXT_SECRET", ""),
        trusted_context_issuer=os.getenv("TRUSTED_CONTEXT_ISSUER", "migtorg-site"),
        internal_status_api_enabled=_bool_from_env(os.getenv("INTERNAL_STATUS_API_ENABLED"), False),
        internal_status_api_url=os.getenv("INTERNAL_STATUS_API_URL", ""),
        internal_status_timeout_seconds=int(os.getenv("INTERNAL_STATUS_TIMEOUT_SECONDS", "5")),
    )
    validate_llm_runtime(settings)
    return settings
