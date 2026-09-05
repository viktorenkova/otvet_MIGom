from __future__ import annotations

from backend.app.build_manifest import build_runtime_manifest
from backend.app.config import get_settings
from backend.app.main import app
from backend.tools.evaluate_live_queries import _health_endpoint_candidates


def test_runtime_manifest_contains_release_comparability_fingerprints() -> None:
    manifest = build_runtime_manifest(get_settings())
    required = {
        "manifest_sha256",
        "git_sha",
        "knowledge_sha256",
        "scenarios_sha256",
        "matching_config_sha256",
        "retrieval_taxonomy_sha256",
        "application_bundle_sha256",
        "routing_bundle_sha256",
        "prompt_bundle_sha256",
        "widget_bundle_sha256",
        "llm_provider",
        "llm_primary_model",
        "llm_policy_sha256",
    }
    assert required.issubset(manifest)
    for key in required - {"git_sha", "llm_provider", "llm_primary_model"}:
        assert len(manifest[key]) == 64
    assert manifest["knowledge_mode"] == "v3_1"


def test_health_manifest_is_exposed_both_directly_and_through_api_proxy_path() -> None:
    paths = {route.path for route in app.routes}

    assert "/health" in paths
    assert "/api/health" in paths


def test_remote_audit_prefers_api_health_for_proxied_deployment() -> None:
    assert _health_endpoint_candidates("https://chat.migtorg.com/api/chat/message") == [
        "https://chat.migtorg.com/api/health",
        "https://chat.migtorg.com/health",
    ]
    assert _health_endpoint_candidates(
        "https://chat.migtorg.com/api/chat/message",
        "https://dev.example.test/manifest",
    ) == ["https://dev.example.test/manifest"]


def test_manifest_identifies_effective_thresholds_and_action_channels():
    from backend.app.config import Settings
    control = build_runtime_manifest(Settings(routing_architecture="control"))
    local = build_runtime_manifest(Settings(routing_architecture="local", ticket_email_enabled=True))
    assert control["effective_scorer_config"]["path"] == "configs/reranker_config.json"
    assert local["effective_scorer_config"]["path"] == "configs/architecture_reranker_config.json"
    assert control["effective_scorer_config"]["sha256"] != local["effective_scorer_config"]["sha256"]
    assert control["action_settings"]["ticket_email_enabled"] is False
    assert local["action_settings"]["ticket_email_enabled"] is True
    assert control["manifest_sha256"] != local["manifest_sha256"]


def test_manifest_never_exports_secrets():
    import json
    from backend.app.config import Settings
    secret = "test-only-do-not-export-value"
    manifest = build_runtime_manifest(Settings(qwen_api_key=secret, litellm_api_key=secret,
        smtp_password=secret, trusted_context_secret=secret))
    assert secret not in json.dumps(manifest)
