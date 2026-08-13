from backend.app.config import Settings
from backend.app.models.llm import LLMResult


class LangfuseClient:
    """Langfuse-ready boundary.

    MVP keeps the dependency optional. When Langfuse credentials are configured,
    this class is the single place to add the real ingestion call.
    """

    def __init__(self, settings: Settings):
        self.settings = settings

    def capture_llm_result(
        self,
        result: LLMResult,
        session_id: str,
        user_role: str,
        escalation_required: bool,
        safety_flags: list[str],
    ) -> None:
        if not self.settings.langfuse_enabled:
            return
        if not (self.settings.langfuse_host and self.settings.langfuse_public_key and self.settings.langfuse_secret_key):
            return
        # Real Langfuse ingestion should be added here once deployment credentials are issued.
        return
