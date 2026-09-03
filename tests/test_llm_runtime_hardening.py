from backend.app.bot.dialog_logger import DialogLogger
from backend.app.config import Settings, validate_llm_runtime
from backend.app.models.llm import LLMResult


def test_llm_telemetry_tracks_environment_cost_verifier_and_fallback(tmp_path) -> None:
    logger = DialogLogger(str(tmp_path / "runtime.sqlite3"))
    logger.log_llm_request(
        LLMResult(
            text="Безопасный ответ",
            provider="qwen",
            model="qwen-plus",
            task_type="answer_generation",
            estimated_cost_usd=0.25,
            latency_ms=120,
            success=True,
            environment="production",
            verification_accepted=False,
            verification_reason="semantic_marker_changed",
            fallback_used=True,
            correlation_id="hashed-correlation",
        ),
        "raw-session-id",
        "guest",
        False,
        [],
    )
    assert logger.get_llm_spend("production", days=1) == 0.25
    assert logger.get_llm_spend("dev", days=1) == 0.0
    metrics = logger.get_llm_metrics(days=1)
    assert metrics["requests"] == 1
    assert metrics["accepted"] == 0
    assert metrics["rejected"] == 1
    assert metrics["fallback_used"] == 1
    assert metrics["verification_reasons"] == {"semantic_marker_changed": 1}


def test_enabled_llm_runtime_validation_is_fail_fast() -> None:
    valid = Settings(
        llm_enabled=True,
        llm_provider="qwen",
        qwen_base_url="https://qwen.example/v1",
        qwen_api_key="test-only",
        llm_input_cost_per_million_usd=1.0,
        llm_output_cost_per_million_usd=2.0,
    )
    validate_llm_runtime(valid)

    invalid = valid.model_copy(update={"qwen_api_key": ""})
    try:
        validate_llm_runtime(invalid)
    except ValueError as exc:
        assert "API key" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Invalid enabled LLM configuration was accepted")
