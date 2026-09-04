from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_QUALITY = Path("reports/quality-stage4-live-160.json")
DEFAULT_CONTRACTS = Path("knowledge/v3_1/answer_contracts.json")
DEFAULT_OUTPUT = Path("reports/stage4-answer-layer-evaluation.json")


def _rate(passed: int, total: int) -> float:
    return round(passed / total * 100, 2) if total else 0.0


def evaluate(quality: dict[str, Any], contracts: dict[str, Any]) -> dict[str, Any]:
    from backend.app.bot.scenario_engine import load_scenarios
    expected_ids = {s.scenario_id for s in load_scenarios()}
    contract_ids = [row["scenario_id"] for row in contracts["records"]]
    routed = [row for row in quality["single_turn_results"] if row["checks"]["route_hit"]]
    criteria = {
        "theme": lambda checks: checks["scenario_ok"] and checks["intent_ok"],
        "completeness": lambda checks: checks["required_facts_ok"],
        "wording_relevance": lambda checks: checks["resolution_ok"] and checks["direct_ok"],
        "clarity": lambda checks: checks["no_duplicate_sentences"] and checks["concise_ok"],
        "no_extra": lambda checks: checks["forbidden_content_ok"],
    }
    by_criterion = {}
    for name, predicate in criteria.items():
        passed = sum(bool(predicate(row["checks"])) for row in routed)
        by_criterion[name] = {"total": len(routed), "passed": passed, "rate_pct": _rate(passed, len(routed))}
    all_passed = sum(all(predicate(row["checks"]) for predicate in criteria.values()) for row in routed)
    unsupported = [
        {"id": row["id"], "hits": row["diagnostics"]["forbidden_hits"]}
        for row in routed if row["diagnostics"]["forbidden_hits"]
    ]
    irrelevant = [
        row["id"] for row in routed
        if not row["checks"]["direct_ok"] or row["diagnostics"]["duplicate_sentences"]
    ]
    all_criteria_pct = _rate(all_passed, len(routed))
    irrelevant_pct = _rate(len(irrelevant), len(routed))
    checks = {
        "all_criteria_gte_93": all_criteria_pct >= 93.0,
        "critical_unsupported_zero": len(unsupported) == 0,
        "irrelevant_blocks_lte_2": irrelevant_pct <= 2.0,
        "contract_coverage_complete": set(contract_ids) == expected_ids and len(contract_ids) == len(set(contract_ids)),
    }
    return {
        "schema_version": 1,
        "methodology": {
            "assessment": "automated marker proxy; not independent expert semantic review",
            "runtime_freshness": "not established by a saved report",
            "population": "live cases with correct scenario and intent route",
            "theme": "accepted scenario_id and intent",
            "completeness": "all required answer marker groups are present",
            "wording_relevance": "accepted resolution and direct answer when required",
            "clarity": "no repeated sentence and answer <=900 characters",
            "no_extra": "no globally or case-forbidden answer fragment",
        },
        "correct_route_total": len(routed),
        "all_criteria": {"passed": all_passed, "rate_pct": all_criteria_pct},
        "by_criterion": by_criterion,
        "critical_unsupported": {"count": len(unsupported), "cases": unsupported},
        "irrelevant_blocks": {"count": len(irrelevant), "rate_pct": irrelevant_pct, "case_ids": irrelevant},
        "answer_contracts": {
            "scenario_count": contracts["record_count"],
            "template_kinds": sorted(contracts["template_kinds"]),
        },
        "checks": checks,
        "passed": all(checks.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quality-report", type=Path, default=DEFAULT_QUALITY)
    parser.add_argument("--contracts", type=Path, default=DEFAULT_CONTRACTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = evaluate(
        json.loads(args.quality_report.read_text(encoding="utf-8")),
        json.loads(args.contracts.read_text(encoding="utf-8")),
    )
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
