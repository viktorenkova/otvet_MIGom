from backend.app.bot.knowledge_search import KnowledgeSearchResult, get_article_by_id
from backend.app.config import Settings
from backend.app.main import _in_llm_rollout
from backend.app.models.llm import LLMResult
from backend.tools.run_llm_shadow import run_llm_shadow, summarize


def test_llm_rollout_is_stable_and_honors_boundaries() -> None:
    assert _in_llm_rollout("session", 0) is False
    assert _in_llm_rollout("session", 100) is True
    assert _in_llm_rollout("stable-session", 25) == _in_llm_rollout("stable-session", 25)


def test_llm_shadow_creates_redacted_expert_review_record(monkeypatch) -> None:
    class FakeProvider:
        def generate(self, request):
            return LLMResult(
                text=request.fallback_text,
                provider="fake",
                model="fake-model",
                task_type=request.task_type,
                input_tokens=10,
                output_tokens=5,
                total_tokens=15,
                estimated_cost_usd=0.01,
                latency_ms=20,
                environment="dev",
            )

    monkeypatch.setattr("backend.app.bot.answer_generator.build_llm_provider", lambda _settings: FakeProvider())
    article = get_article_by_id("buyer.get_started", "guest")
    assert article is not None

    def router(_message: str, _intent: str, _role: str) -> KnowledgeSearchResult:
        return KnowledgeSearchResult(article, 200, "high")

    results = run_llm_shadow(
        [{
            "id": "one",
            "event_id": "llm-shadow-0001",
            "text": "Клиент: Иван Иванов, как участвовать? Телефон +7 999 123-45-67",
            "role": "guest",
            "source": "input.json",
        }],
        Settings(llm_enabled=True, llm_provider="fake", llm_primary_model="fake-model"),
        router=router,
    )
    assert len(results) == 1
    assert "Иванов" not in results[0]["message_redacted"]
    assert "123-45-67" not in results[0]["message_redacted"]
    assert results[0]["expert_review"]["correctness"] is None
    report = summarize(results)
    assert report["llm"]["invoked"] == 1
    assert report["llm"]["verifier_accepted"] == 1
    assert report["privacy"]["records_with_detected_pii"] == 0
    assert report["expert_review"]["gate_passed"] is False
