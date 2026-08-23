from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from backend.app.bot.routing_v3 import get_routing_v3


DEFAULT_DATASET = Path("tests/data/routing_v3_closed_control_270.json")
DEFAULT_REPORT = Path("reports/routing-v3-closed-control-270.json")
DEFAULT_OUTPUT = Path("reports/routing-label-adjudication-110.json")


def _priority(case: dict[str, Any], result: dict[str, Any]) -> tuple[str, list[str]]:
    labels = case.get("expected", {}).get("expected_scenario_ids", [])
    reasons: list[str] = []
    if not result.get("checks", {}).get("quality_pass"):
        reasons.append("current_quality_failure")
    if len(labels) != 1 or any(item is None for item in labels):
        reasons.append("multiple_or_null_acceptable_labels")
    if case.get("group") in {"payments", "refunds_penalties", "transfer_docs"}:
        reasons.append("financial_or_transactional_risk")
    priority = "high" if reasons[:2] else "medium" if reasons else "low"
    return priority, reasons


def build(dataset_path: Path, report_path: Path) -> dict[str, Any]:
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    widget_cases = [item for item in dataset["cases"] if item.get("source") == "live_widget_audit_110"]
    results = {item["id"]: item for item in report["single_turn_results"]}
    records: list[dict[str, Any]] = []
    for case in widget_cases:
        result = results[case["id"]]
        priority, reasons = _priority(case, result)
        candidates = get_routing_v3().rank(case["text"], role="guest", top_k=5)
        records.append({
            "case_id": case["id"],
            "text": case["text"],
            "group": case["group"],
            "current_expected_scenario_ids": case["expected"].get("expected_scenario_ids", []),
            "current_expected_intents": case["expected"].get("expected_intents", []),
            "current_actual_scenario_id": (result.get("response") or {}).get("scenario_id"),
            "current_quality_pass": bool(result["checks"]["quality_pass"]),
            "top5_candidates": [
                {"scenario_id": item.scenario.scenario_id, "score": round(item.score, 6)}
                for item in candidates
            ],
            "review_priority": priority,
            "review_reasons": reasons,
            "review_status": "pending_domain_expert",
            "reviewer": None,
            "reviewed_at": None,
            "adjudicated_scenario_ids": None,
            "adjudication_note": None,
        })
    if len(records) != 110:
        raise ValueError(f"Expected 110 widget labels, got {len(records)}")
    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "Domain-expert adjudication queue for the retrospectively labelled 110 live-widget questions.",
        "policy": "Labels are not release-authoritative until review_status=approved and reviewer/reviewed_at are set.",
        "record_count": len(records),
        "pending_count": len(records),
        "records": records,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the 110-case domain label adjudication queue.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = build(args.dataset, args.report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    priorities: dict[str, int] = {}
    for record in payload["records"]:
        priorities[record["review_priority"]] = priorities.get(record["review_priority"], 0) + 1
    print(json.dumps({"output": str(args.output), "records": payload["record_count"], "priorities": priorities}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
