"""Validate runtime configuration and knowledge gates before production deploy."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from urllib.parse import urlsplit

from backend.app.config import Settings, get_settings
from backend.tools.audit_knowledge import audit as audit_knowledge


def _check(name: str, status: str, detail: str) -> dict[str, str]:
    return {"name": name, "status": status, "detail": detail}


def _is_https_origin(value: str) -> bool:
    parsed = urlsplit(value)
    return (
        parsed.scheme == "https"
        and bool(parsed.netloc)
        and parsed.path in {"", "/"}
        and not parsed.query
        and not parsed.fragment
    )


def _is_https_url(value: str) -> bool:
    parsed = urlsplit(value)
    return parsed.scheme == "https" and bool(parsed.netloc)


def run_checks(settings: Settings, knowledge_report: dict | None = None) -> dict:
    checks: list[dict[str, str]] = []

    checks.append(
        _check(
            "app_environment",
            "pass" if settings.app_environment == "production" else "fail",
            f"APP_ENVIRONMENT={settings.app_environment!r}; expected 'production'.",
        )
    )
    version = settings.deploy_version.strip()
    checks.append(
        _check(
            "deploy_version",
            "pass" if version and version.lower() not in {"local", "unknown", "dev"} else "fail",
            "DEPLOY_VERSION identifies the artifact."
            if version and version.lower() not in {"local", "unknown", "dev"}
            else "Set DEPLOY_VERSION to an image tag or commit SHA.",
        )
    )
    checks.append(
        _check(
            "debug_disabled",
            "pass" if not settings.debug else "fail",
            "DEBUG must be false in production.",
        )
    )

    origins = settings.cors_allowed_origins
    cors_valid = bool(origins) and "*" not in origins and all(_is_https_origin(item) for item in origins)
    checks.append(
        _check(
            "cors_origins",
            "pass" if cors_valid else "fail",
            "CORS origins must be explicit HTTPS origins without paths, query strings, or fragments.",
        )
    )

    widget_files = [settings.widget_root / name for name in ("index.html", "widget.js", "style.css")]
    missing_widget_files = [path.name for path in widget_files if not path.is_file()]
    checks.append(
        _check(
            "widget_assets",
            "fail" if missing_widget_files else "pass",
            "Missing: " + ", ".join(missing_widget_files) if missing_widget_files else "Widget assets are present.",
        )
    )

    knowledge_active = settings.knowledge_v2_enabled and not settings.knowledge_v2_shadow_mode
    checks.append(
        _check(
            "knowledge_v2_active",
            "pass" if knowledge_active else "fail",
            "KNOWLEDGE_V2_ENABLED must be true and KNOWLEDGE_V2_SHADOW_MODE must be false.",
        )
    )
    report = knowledge_report if knowledge_report is not None else audit_knowledge()
    checks.append(
        _check(
            "knowledge_production_gate",
            "pass" if report.get("production_release_ready") is True else "fail",
            "Knowledge audit production_release_ready must be true.",
        )
    )

    database_path = Path(settings.database_path)
    checks.append(
        _check(
            "database_persistence",
            "pass" if database_path.is_absolute() else "warn",
            "Use an absolute path on a persistent volume; relative paths depend on the process working directory.",
        )
    )

    email_fields_ready = all(
        (
            settings.smtp_host.strip(),
            settings.ticket_email_to.strip(),
            settings.ticket_email_from.strip(),
        )
    )
    email_addresses_real = all(
        not address.lower().endswith(".example")
        for address in (settings.ticket_email_to.strip(), settings.ticket_email_from.strip())
    )
    checks.append(
        _check(
            "ticket_delivery",
            "pass" if settings.ticket_email_enabled and email_fields_ready and email_addresses_real else "fail",
            "Enable email delivery and configure SMTP plus non-placeholder sender and recipient addresses.",
        )
    )

    checks.append(
        _check(
            "operations_token",
            "pass" if len(settings.quality_report_token) >= 32 else "warn",
            "Set a random QUALITY_REPORT_TOKEN of at least 32 characters for reports and retry operations.",
        )
    )

    if settings.internal_status_api_enabled:
        status_ready = (
            _is_https_url(settings.internal_status_api_url)
            and len(settings.trusted_context_secret) >= 32
            and bool(settings.trusted_context_issuer.strip())
        )
        checks.append(
            _check(
                "trusted_status_integration",
                "pass" if status_ready else "fail",
                "Enabled status API requires HTTPS URL, issuer, and a secret of at least 32 characters.",
            )
        )
    else:
        checks.append(
            _check(
                "trusted_status_integration",
                "warn",
                "Personal status integration is disabled; the bot will safely redirect these requests.",
            )
        )

    if settings.llm_enabled:
        provider_ready = settings.llm_provider in {"litellm", "qwen"}
        if settings.llm_provider == "litellm":
            provider_ready = provider_ready and _is_https_url(settings.litellm_proxy_url)
        elif settings.llm_provider == "qwen":
            provider_ready = (
                provider_ready and _is_https_url(settings.qwen_base_url) and bool(settings.qwen_api_key)
            )
        provider_ready = (
            provider_ready
            and settings.llm_environment == "production"
            and settings.llm_input_cost_per_million_usd > 0
            and settings.llm_output_cost_per_million_usd > 0
            and settings.llm_daily_budget_usd > 0
            and settings.active_llm_monthly_budget_usd > 0
            and 0 < settings.llm_total_timeout_seconds <= settings.llm_request_timeout_seconds * 2
            and settings.llm_max_concurrency > 0
            and settings.llm_circuit_failure_threshold > 0
            and settings.llm_rollout_percentage in {0, 5, 25, 50, 100}
        )
        checks.append(
            _check(
                "llm_configuration",
                "pass" if provider_ready else "fail",
                "Enabled LLM requires production environment, HTTPS provider credentials, token pricing, budgets, timeout and circuit limits.",
            )
        )
    else:
        checks.append(_check("llm_configuration", "pass", "LLM is disabled; deterministic rules remain active."))

    failures = sum(item["status"] == "fail" for item in checks)
    warnings = sum(item["status"] == "warn" for item in checks)
    return {
        "production_ready": failures == 0,
        "failures": failures,
        "warnings": warnings,
        "checks": checks,
    }


def _load_env_file(path: Path) -> None:
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.removeprefix("export ").split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        os.environ.setdefault(key, value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Check MIGTORG chatbot production readiness.")
    parser.add_argument("--env-file", type=Path, help="Load configuration without overriding existing variables.")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--strict", action="store_true", help="Exit with code 1 when a required check fails.")
    args = parser.parse_args()
    if args.env_file:
        _load_env_file(args.env_file)
        get_settings.cache_clear()
    result = run_checks(get_settings())
    serialized = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    print(serialized)
    if args.strict and not result["production_ready"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
