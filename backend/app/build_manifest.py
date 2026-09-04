from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]


def _embedding_fingerprint(index) -> dict[str, Any]:
    if index.model is None:
        return {"weights_sha256": None, "revision": None}
    cached = getattr(index, "_manifest_embedding_fingerprint", None)
    if cached is not None:
        return cached
    digest = hashlib.sha256()
    for name, tensor in sorted(index.model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    config = index.model[0].auto_model.config
    result = {"weights_sha256": digest.hexdigest(), "revision": getattr(config, "_commit_hash", None)}
    index._manifest_embedding_fingerprint = result
    return result


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
    llm_policy = {
        "environment": settings.llm_environment,
        "reasoning_effort": settings.llm_reasoning_effort,
        "request_timeout_seconds": settings.llm_request_timeout_seconds,
        "total_timeout_seconds": settings.llm_total_timeout_seconds,
        "max_output_tokens": settings.llm_max_output_tokens,
        "max_concurrency": settings.llm_max_concurrency,
        "circuit_failure_threshold": settings.llm_circuit_failure_threshold,
        "circuit_cooldown_seconds": settings.llm_circuit_cooldown_seconds,
        "rollout_percentage": settings.llm_rollout_percentage,
    }
    llm_policy_raw = json.dumps(llm_policy, sort_keys=True, separators=(",", ":")).encode("utf-8")
    payload: dict[str, Any] = {
        "manifest_schema": 2,
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
        "widget_bundle_sha256": _sha256_files(_tree_files("frontend", {".js", ".css", ".html"})),
        "scorer_artifact_sha256": _sha256_file(ROOT / "artifacts/stage3-pairwise-reranker.joblib"),
        "scorer_config_sha256": _sha256_file(ROOT / "configs/reranker_config.json"),
        "candidate_scorer_config_sha256": _sha256_file(ROOT / "configs/architecture_reranker_config.json"),
        "feature_schema_sha256": _sha256_file(ROOT / "backend/app/bot/pairwise_reranker.py"),
        "evaluator_sha256": _sha256_files(_tree_files("backend/tools", {".py"})),
        "policy_bundle_sha256": _sha256_files(_tree_files("configs", {".json"})),
        "routing_architecture": settings.routing_architecture,
        "dialogue_state_enabled": settings.dialogue_state_enabled,
        "answer_assembly_enabled": settings.answer_assembly_enabled,
        "llm_understanding_enabled": settings.llm_understanding_enabled,
        "architecture_experiment": settings.architecture_experiment,
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
        "llm_policy_sha256": _sha256_bytes(llm_policy_raw),
    }
    from backend.app.bot.knowledge_search import _semantic_index
    semantic = json.loads(matching_config.read_text(encoding="utf-8")).get("semantic_matching", {})
    payload["retrieval_settings"] = semantic
    payload["dense_runtime"] = (
        {"initialized": True, "available": _semantic_index().model is not None,
         "error": _semantic_index().dense_error,
         "configured_model": _semantic_index().config.get("dense_model"),
         **_embedding_fingerprint(_semantic_index())}
        if _semantic_index.cache_info().currsize else {"initialized": False, "available": None}
    )
    from importlib.metadata import version, PackageNotFoundError
    payload["dependencies"] = {}
    for name in ("numpy", "scikit-learn", "sentence-transformers", "torch", "fastapi", "pydantic"):
        try:
            payload["dependencies"][name] = version(name)
        except PackageNotFoundError:
            payload["dependencies"][name] = "unavailable"
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    payload["manifest_sha256"] = _sha256_bytes(canonical.encode("utf-8"))
    return payload
