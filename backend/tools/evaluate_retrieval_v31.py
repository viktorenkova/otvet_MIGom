from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from backend.app.bot.routing_v3 import get_routing_v3


DEFAULT_DATASET = Path("tests/data/retrieval_v31_development_validation.json")
DEFAULT_OUTPUT = Path("reports/retrieval-v31-development-validation.json")


def _metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    return {
        "total": total,
        "recall_at_1_pct": round(sum(item["hit_at_1"] for item in rows) / total * 100, 2) if total else 0.0,
        "recall_at_5_pct": round(sum(item["hit_at_5"] for item in rows) / total * 100, 2) if total else 0.0,
        "recall_at_10_pct": round(sum(item["hit_at_10"] for item in rows) / total * 100, 2) if total else 0.0,
    }


def evaluate(dataset: dict[str, Any]) -> dict[str, Any]:
    router = get_routing_v3()
    rows: list[dict[str, Any]] = []
    for case in dataset["cases"]:
        candidates = router.rank(case["text"], case["role"], top_k=10)
        candidate_ids = [item.scenario.scenario_id for item in candidates]
        expected = case["expected_scenario_id"]
        rows.append({
            "id": case["id"],
            "split": case["split"],
            "variant": case["variant"],
            "expected_scenario_id": expected,
            "candidate_scenario_ids": candidate_ids,
            "hit_at_1": expected in candidate_ids[:1],
            "hit_at_5": expected in candidate_ids[:5],
            "hit_at_10": expected in candidate_ids[:10],
        })
    development = [item for item in rows if item["split"] == "development"]
    validation = [item for item in rows if item["split"] == "validation"]
    by_variant = {
        variant: _metrics([item for item in rows if item["variant"] == variant])
        for variant in sorted({item["variant"] for item in rows})
    }
    dev_metrics = _metrics(development)
    validation_metrics = _metrics(validation)
    checks = {
        "no_exact_frozen_overlap": dataset["exact_frozen_overlap_count"] == 0,
        "development_recall_at_10_gte_97": dev_metrics["recall_at_10_pct"] >= 97.0,
        "validation_recall_at_10_gte_97": validation_metrics["recall_at_10_pct"] >= 97.0,
        "each_variant_recall_at_10_gte_90": all(item["recall_at_10_pct"] >= 90.0 for item in by_variant.values()),
    }
    return {
        "schema_version": 1,
        "dataset": {
            "version": dataset["version"],
            "case_count": dataset["case_count"],
            "cases_sha256": dataset["cases_sha256"],
            "knowledge_sha256": dataset["knowledge_sha256"],
        },
        "development": dev_metrics,
        "validation": validation_metrics,
        "by_variant": by_variant,
        "checks": checks,
        "passed": all(checks.values()),
        "failures": [item for item in rows if not item["hit_at_10"]],
        "results": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate Top-K retrieval on KB v3.1 development/validation cases.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = evaluate(json.loads(args.dataset.read_text(encoding="utf-8")))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("development", "validation", "by_variant", "checks", "passed")}, ensure_ascii=False, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
