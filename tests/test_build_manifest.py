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
