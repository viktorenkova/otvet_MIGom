from __future__ import annotations

from collections import defaultdict
from datetime import date
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
LEGACY_PATH = ROOT / "knowledge" / "normalized" / "migtorg_knowledge_base.json"
SCENARIOS_PATH = ROOT / "knowledge" / "v2" / "scenarios.json"
OVERRIDES_PATH = ROOT / "knowledge" / "normalized" / "scenario_overrides.json"
OUTPUT_PATH = ROOT / "knowledge" / "v2" / "legacy_inventory.json"

CRITICAL_INTENTS = {"payment", "refund", "penalty", "refusal", "tariffs"}
HIGH_INTENTS = {"transfer", "inspection", "pickup", "registration"}


def _priority(intent: str) -> str:
    if intent in CRITICAL_INTENTS:
        return "critical"
    if intent in HIGH_INTENTS:
        return "high"
    return "normal"


def build_inventory() -> dict:
    legacy = json.loads(LEGACY_PATH.read_text(encoding="utf-8"))
    scenarios = json.loads(SCENARIOS_PATH.read_text(encoding="utf-8"))
    overrides = json.loads(OVERRIDES_PATH.read_text(encoding="utf-8"))

    scenario_targets: dict[str, list[str]] = defaultdict(list)
    primary_legacy_ids: set[str] = set()
    for scenario in scenarios.get("records", []):
        if str(scenario.get("status") or "active") != "active":
            continue
        legacy_ids = [str(item) for item in scenario.get("legacy_ids", [])]
        if legacy_ids:
            primary_legacy_ids.add(legacy_ids[0])
        for legacy_id in legacy_ids:
            scenario_targets[legacy_id].append(str(scenario["scenario_id"]))

    disabled_overrides = {
        str(legacy_id)
        for legacy_id, override in overrides.get("articles", {}).items()
        if isinstance(override, dict) and override.get("fallback_allowed") is False
    }

    records = []
    for record in legacy.get("records", []):
        if str(record.get("status") or "active") != "active":
            continue
        legacy_id = str(record["id"])
        targets = sorted(set(scenario_targets.get(legacy_id, [])))
        source = str(record.get("source") or "")
        if targets:
            status = "migrated_to_v2" if legacy_id in primary_legacy_ids else "merged_into_v2"
            rationale = "Content is represented by active v2 scenario(s); legacy search entry must be suppressed."
        elif legacy_id in disabled_overrides:
            status = "deactivated"
            rationale = "Legacy answer is unsafe, ambiguous, outdated, or superseded by guarded routing."
        elif "disputed-site-document-points.md" in source:
            status = "expert_review_required"
            rationale = "Source set contains an explicitly disputed provision; publication requires expert resolution."
        else:
            status = "retained_confirmed"
            rationale = "Reviewed legacy answer remains temporarily active until its confirmed block is migrated to v2."

        records.append(
            {
                "legacy_id": legacy_id,
                "title": str(record.get("title") or record.get("topic") or ""),
                "intent": str(record.get("intent") or "unknown"),
                "priority": _priority(str(record.get("intent") or "unknown")),
                "status": status,
                "target_scenario_ids": targets,
                "source": source,
                "reviewed_at": str(record.get("reviewed_at") or ""),
                "review_owner": str(record.get("review_owner") or ""),
                "rationale": rationale,
                "blocks_production": status in {"retained_confirmed", "expert_review_required"},
            }
        )

    return {
        "schema_version": 1,
        "generated_at": date.today().isoformat(),
        "policy": "Each active legacy record has exactly one disposition. Retained confirmed and expert-review records block migration completion.",
        "allowed_statuses": [
            "migrated_to_v2",
            "merged_into_v2",
            "retained_confirmed",
            "expert_review_required",
            "deactivated",
        ],
        "records": records,
    }


def main() -> None:
    OUTPUT_PATH.write_text(
        json.dumps(build_inventory(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
