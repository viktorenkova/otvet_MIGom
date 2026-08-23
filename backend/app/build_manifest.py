from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes()) if path.is_file() else "missing"


def _sha256_files(paths: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    existing = sorted(
        (path for path in paths if path.is_file()),
        key=lambda item: item.relative_to(ROOT).as_posix(),
    )
    for path in existing:
        relative = path.relative_to(ROOT).as_posix().encode("utf-8")
        payload = path.read_bytes()
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _tree_files(relative_root: str, suffixes: set[str] | None = None) -> list[Path]:
    root = ROOT / relative_root
    if not root.is_dir():
        return []
    return [
        path
        for path in root.rglob("*")
        if path.is_file() and (suffixes is None or path.suffix.casefold() in suffixes)
    ]


def _git_revision() -> tuple[str, bool | None]:
    configured = os.getenv("GIT_COMMIT_SHA", "").strip()
    if not configured:
        deploy_version = os.getenv("DEPLOY_VERSION", "").strip()
        if deploy_version and deploy_version != "local":
            configured = deploy_version
    if configured:
        return configured, None
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain", "--untracked-files=no"],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
                timeout=2,
            ).stdout.strip()
        )
        return revision, dirty
    except (OSError, subprocess.SubprocessError):
        return "unknown", None


def build_runtime_manifest(settings: Any) -> dict[str, Any]:
    """Return non-secret build fingerprints used to prove audit comparability."""

    git_sha, working_tree_dirty = _git_revision()
    scenarios = ROOT / "knowledge" / "v2" / "scenarios.json"
    matching_config = ROOT / "configs" / "no_llm_matching_config.json"
    retrieval_taxonomy = ROOT / "configs" / "retrieval_taxonomy_terms.json"
    routing_files = [
        ROOT / "backend" / "app" / "bot" / "routing_v3.py",
        ROOT / "backend" / "app" / "bot" / "scenario_engine.py",
        ROOT / "backend" / "app" / "bot" / "knowledge_search.py",
        ROOT / "backend" / "app" / "bot" / "intent_classifier.py",
        ROOT / "backend" / "app" / "bot" / "text_processing.py",
        matching_config,
        retrieval_taxonomy,
        ROOT / "configs" / "intent_patterns.json",
        ROOT / "configs" / "synonym_groups.json",
        ROOT / "configs" / "typo_corrections.json",
    ]
    prompt_files = [
        ROOT / "backend" / "app" / "bot" / "answer_generator.py",
        ROOT / "backend" / "app" / "integrations" / "llm_provider.py",
    ]
    payload: dict[str, Any] = {
        "manifest_schema": 1,
        "deploy_version": str(settings.deploy_version),
        "git_sha": git_sha,
        "working_tree_dirty": working_tree_dirty,
        "knowledge_sha256": _sha256_files(_tree_files("knowledge", {".json", ".md"})),
        "scenarios_sha256": _sha256_file(scenarios),
        "matching_config_sha256": _sha256_file(matching_config),
        "retrieval_taxonomy_sha256": _sha256_file(retrieval_taxonomy),
        "application_bundle_sha256": _sha256_files(_tree_files("backend/app", {".py"})),
        "routing_bundle_sha256": _sha256_files(routing_files),
        "prompt_bundle_sha256": _sha256_files(prompt_files),
        "widget_bundle_sha256": _sha256_files(_tree_files("frontend/chat-widget")),
        "knowledge_mode": (
            "v3_1"
            if settings.knowledge_v2_enabled
            and not settings.knowledge_v2_shadow_mode
            and (ROOT / "knowledge" / "v3_1" / "scenarios.json").is_file()
            else "v2"
            if settings.knowledge_v2_enabled and not settings.knowledge_v2_shadow_mode
            else "legacy_or_shadow"
        ),
        "llm_enabled": bool(settings.llm_enabled),
        "llm_provider": str(settings.llm_provider),
        "llm_primary_model": str(settings.llm_primary_model),
        "llm_fallback_model": str(settings.llm_fallback_model),
    }
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    payload["manifest_sha256"] = _sha256_bytes(canonical.encode("utf-8"))
    return payload
