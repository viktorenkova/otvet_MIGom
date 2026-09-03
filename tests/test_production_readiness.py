from pathlib import Path

from backend.app.config import Settings
from backend.app.main import health
from backend.tools.check_production_readiness import run_checks


def _production_settings(tmp_path: Path) -> Settings:
    return Settings(
        app_environment="production",
        deploy_version="2026.08.12-abc1234",
        debug=False,
        cors_allowed_origins=["https://migtorg.com", "https://www.migtorg.com"],
        database_path=str(tmp_path / "migtorg.sqlite3"),
        ticket_email_enabled=True,
        ticket_email_to="support@migtorg.com",
        ticket_email_from="bot@migtorg.com",
        smtp_host="smtp.migtorg.com",
        quality_report_token="q" * 32,
    )


def _statuses(result: dict) -> dict[str, str]:
    return {item["name"]: item["status"] for item in result["checks"]}


def test_production_readiness_passes_required_checks(tmp_path):
    result = run_checks(_production_settings(tmp_path), {"production_release_ready": True})

    assert result["production_ready"] is True
    assert result["failures"] == 0
    assert _statuses(result)["trusted_status_integration"] == "warn"


def test_production_readiness_rejects_unsafe_runtime_config(tmp_path):
    settings = _production_settings(tmp_path).model_copy(
        update={
            "app_environment": "development",
            "deploy_version": "local",
            "debug": True,
            "cors_allowed_origins": ["*"],
            "ticket_email_enabled": False,
            "knowledge_v2_shadow_mode": True,
        }
    )

    result = run_checks(settings, {"production_release_ready": False})
    statuses = _statuses(result)

    assert result["production_ready"] is False
    assert statuses["app_environment"] == "fail"
    assert statuses["deploy_version"] == "fail"
    assert statuses["debug_disabled"] == "fail"
    assert statuses["cors_origins"] == "fail"
    assert statuses["ticket_delivery"] == "fail"
    assert statuses["knowledge_v2_active"] == "fail"
    assert statuses["knowledge_production_gate"] == "fail"


def test_enabled_integrations_require_secure_configuration(tmp_path):
    settings = _production_settings(tmp_path).model_copy(
        update={
            "internal_status_api_enabled": True,
            "internal_status_api_url": "http://internal.example/status",
            "trusted_context_secret": "short",
            "llm_enabled": True,
            "llm_environment": "dev",
            "llm_provider": "qwen",
            "qwen_base_url": "http://qwen.example/v1",
            "qwen_api_key": "",
        }
    )

    statuses = _statuses(run_checks(settings, {"production_release_ready": True}))

    assert statuses["trusted_status_integration"] == "fail"
    assert statuses["llm_configuration"] == "fail"


def test_enabled_llm_readiness_requires_pricing_and_runtime_limits(tmp_path):
    base = _production_settings(tmp_path).model_copy(
        update={
            "llm_enabled": True,
            "llm_environment": "production",
            "llm_provider": "qwen",
            "qwen_base_url": "https://qwen.example/v1",
            "qwen_api_key": "test-only",
            "llm_input_cost_per_million_usd": 1.0,
            "llm_output_cost_per_million_usd": 2.0,
        }
    )
    assert _statuses(run_checks(base, {"production_release_ready": True}))["llm_configuration"] == "pass"
    invalid = base.model_copy(update={"llm_input_cost_per_million_usd": 0.0})
    assert _statuses(run_checks(invalid, {"production_release_ready": True}))["llm_configuration"] == "fail"


def test_public_health_is_release_focused_and_does_not_expose_local_paths():
    result = health()

    assert result["status"] == "ok"
    assert {"app_name", "environment", "deploy_version", "widget_ready", "knowledge_mode"} <= result.keys()
    assert {"app_file", "widget_root", "routes"}.isdisjoint(result)
