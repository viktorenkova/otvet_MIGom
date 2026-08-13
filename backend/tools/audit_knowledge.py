from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import date, datetime, timedelta
import json
from pathlib import Path

from backend.app.bot.scenario_engine import load_scenarios
from backend.app.config import get_settings
from backend.tools.evaluate_scenarios import evaluate


def audit() -> dict:
    scenarios = load_scenarios()
    errors: list[dict] = []
    example_owners: dict[str, list[str]] = defaultdict(list)
    mapped_legacy: set[str] = set()
    today = date.today()
    for item in scenarios:
        for field, value in (
            ("facts", item.facts),
            ("positive_examples", item.positive_examples),
            ("short_answer", item.short_answer),
            ("source", item.source),
            ("review_owner", item.review_owner),
            ("expert", item.expert),
        ):
            if not value:
                errors.append({"scenario_id": item.scenario_id, "issue": f"missing_{field}"})
        for example in item.positive_examples:
            example_owners[" ".join(example.casefold().split())].append(item.scenario_id)
        mapped_legacy.update(item.legacy_ids)
        try:
            reviewed = datetime.fromisoformat(item.reviewed_at).date()
            if reviewed + timedelta(days=item.review_interval_days) < today:
                errors.append({"scenario_id": item.scenario_id, "issue": "review_expired"})
        except ValueError:
            errors.append({"scenario_id": item.scenario_id, "issue": "invalid_reviewed_at"})
        answer = item.answer.casefold()
        if any(marker in answer for marker in ("если пользователь", "бот должен", "первая линия")):
            errors.append({"scenario_id": item.scenario_id, "issue": "internal_instruction_in_answer"})

    collisions = [
        {"example": example, "scenario_ids": owners}
        for example, owners in example_owners.items()
        if len(set(owners)) > 1
    ]
    legacy_path = get_settings().knowledge_root / "normalized" / "migtorg_knowledge_base.json"
    legacy_records = json.loads(legacy_path.read_text(encoding="utf-8")).get("records", []) if legacy_path.exists() else []
    active_legacy_ids = {
        str(item.get("id"))
        for item in legacy_records
        if str(item.get("status") or "active") == "active"
    }
    unmapped = sorted(active_legacy_ids - mapped_legacy)
    inventory_path = get_settings().knowledge_root / "v2" / "legacy_inventory.json"
    inventory_errors: list[dict] = []
    inventory_records: list[dict] = []
    allowed_inventory_statuses = {
        "migrated_to_v2",
        "merged_into_v2",
        "retained_confirmed",
        "expert_review_required",
        "deactivated",
    }
    if inventory_path.exists():
        inventory_raw = json.loads(inventory_path.read_text(encoding="utf-8"))
        inventory_records = [item for item in inventory_raw.get("records", []) if isinstance(item, dict)]
    else:
        inventory_errors.append({"issue": "missing_legacy_inventory"})
    inventory_ids = [str(item.get("legacy_id") or "") for item in inventory_records]
    inventory_id_set = set(inventory_ids)
    for legacy_id in sorted(active_legacy_ids - inventory_id_set):
        inventory_errors.append({"legacy_id": legacy_id, "issue": "missing_inventory_record"})
    for legacy_id in sorted(inventory_id_set - active_legacy_ids):
        inventory_errors.append({"legacy_id": legacy_id, "issue": "inventory_record_not_active_legacy"})
    for legacy_id, count in sorted((item, inventory_ids.count(item)) for item in inventory_id_set):
        if not legacy_id or count != 1:
            inventory_errors.append({"legacy_id": legacy_id, "issue": "missing_or_duplicate_inventory_record"})
    active_scenario_ids = {item.scenario_id for item in scenarios}
    status_counts: dict[str, int] = defaultdict(int)
    source_competition_ids: list[str] = []
    blocking_inventory_ids: list[str] = []
    for item in inventory_records:
        legacy_id = str(item.get("legacy_id") or "")
        status = str(item.get("status") or "")
        status_counts[status] += 1
        if status not in allowed_inventory_statuses:
            inventory_errors.append({"legacy_id": legacy_id, "issue": "invalid_inventory_status"})
        targets = [str(value) for value in item.get("target_scenario_ids", [])]
        if status in {"migrated_to_v2", "merged_into_v2"} and not targets:
            inventory_errors.append({"legacy_id": legacy_id, "issue": "missing_target_scenario"})
        for target in targets:
            if target not in active_scenario_ids:
                inventory_errors.append({"legacy_id": legacy_id, "issue": "inactive_target_scenario", "target": target})
        if status in {"retained_confirmed", "expert_review_required"}:
            blocking_inventory_ids.append(legacy_id)
        if legacy_id in mapped_legacy and status not in {"migrated_to_v2", "merged_into_v2", "deactivated"}:
            source_competition_ids.append(legacy_id)
    curated_aliases = {
        legacy_id: scenario.scenario_id
        for scenario in scenarios
        for legacy_id in scenario.legacy_ids
    }
    compatibility_map = {
        legacy_id: curated_aliases.get(legacy_id, legacy_id)
        for legacy_id in sorted(active_legacy_ids)
    }
    engine_release_ready = not errors and not collisions
    inventory_complete = not inventory_errors and inventory_id_set == active_legacy_ids
    content_migration_complete = inventory_complete and not blocking_inventory_ids
    review_queue_path = get_settings().knowledge_root / "v2" / "review_queue.json"
    review_queue_records = []
    review_queue_errors: list[dict] = []
    if review_queue_path.exists():
        review_raw = json.loads(review_queue_path.read_text(encoding="utf-8"))
        review_queue_records = [
            item for item in review_raw.get("records", []) if isinstance(item, dict)
        ]
        active_scenario_ids = {item.scenario_id for item in scenarios}
        candidate_ids: set[str] = set()
        for item in review_queue_records:
            candidate_id = str(item.get("candidate_id") or "")
            proposed_id = str(item.get("proposed_scenario_id") or "")
            if not candidate_id or candidate_id in candidate_ids:
                review_queue_errors.append({"candidate_id": candidate_id, "issue": "missing_or_duplicate_candidate_id"})
            candidate_ids.add(candidate_id)
            for field in ("proposed_scenario_id", "risk", "support_conversations", "questions_for_expert", "publication_blockers", "owner", "expert_role"):
                if not item.get(field):
                    review_queue_errors.append({"candidate_id": candidate_id, "issue": f"missing_{field}"})
            if str(item.get("status") or "") != "expert_review_required":
                review_queue_errors.append({"candidate_id": candidate_id, "issue": "invalid_review_status"})
            if proposed_id in active_scenario_ids:
                review_queue_errors.append({"candidate_id": candidate_id, "issue": "draft_already_active"})
    gold_path = Path("tests/data/scenario_gold.jsonl")
    gold_report = evaluate(gold_path) if gold_path.exists() else {"total": 0, "production_gate_passed": False}
    manual_review_path = get_settings().knowledge_root / "v2" / "manual_dialogue_review.json"
    manual_review_passed = False
    if manual_review_path.exists():
        manual_review = json.loads(manual_review_path.read_text(encoding="utf-8"))
        manual_review_passed = bool(
            manual_review.get("completed")
            and manual_review.get("systematic_false_answers_found") is False
            and manual_review.get("review_owner")
            and manual_review.get("reviewed_at")
        )
    return {
        "schema_version": 2,
        "scenarios": len(scenarios),
        "errors": errors,
        "example_collisions": collisions,
        "legacy_records": len(active_legacy_ids),
        "mapped_legacy_records": len(active_legacy_ids & mapped_legacy),
        "unmapped_legacy_records": unmapped,
        "legacy_compatibility_map": compatibility_map,
        "engine_release_ready": engine_release_ready,
        "content_migration_complete": content_migration_complete,
        "legacy_inventory_records": len(inventory_records),
        "legacy_inventory_status_counts": dict(sorted(status_counts.items())),
        "legacy_inventory_errors": inventory_errors,
        "source_competition_ids": sorted(source_competition_ids),
        "blocking_legacy_records": len(blocking_inventory_ids),
        "gold_evaluation": gold_report,
        "manual_dialogue_review_passed": manual_review_passed,
        "production_release_ready": (
            engine_release_ready
            and content_migration_complete
            and not source_competition_ids
            and bool(gold_report.get("production_gate_passed"))
            and manual_review_passed
        ),
        "review_queue_records": len(review_queue_records),
        "review_queue_errors": review_queue_errors,
        # Backward-compatible gate: a staged v2 rollout may keep serving unmigrated
        # records through the legacy compatibility layer.
        "release_ready": engine_release_ready,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit scenario knowledge before release.")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    report = audit()
    serialized = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    else:
        print(serialized)
    if args.strict and not report["production_release_ready"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
