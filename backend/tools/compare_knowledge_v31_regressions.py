from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT = Path("reports/knowledge-v31-regression-comparison.json")
MAX_DEGRADATION_PP = 2.0


CORPORA = (
    (
        "gold_312",
        Path("reports/scenario-gold-production-gate-2026-08-13.json"),
        Path("reports/knowledge-v31-regression-gold-312.json"),
        "gold",
    ),
    (
        "independent_116",
        Path("reports/routing-v3-independent-acceptance.json"),
        Path("reports/knowledge-v31-regression-independent-116.json"),
        "live",
    ),
    (
        "live_160",
        Path("reports/routing-v3-live-query-final.json"),
        Path("reports/knowledge-v31-regression-live-160.json"),
        "live",
    ),
    (
        "closed_270_adjudicated",
        Path("reports/quality-stage0-adjudicated-local.json"),
        Path("reports/knowledge-v31-regression-closed-270.json"),
        "live",
    ),
)


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _live_metrics(payload: dict[str, Any]) -> dict[str, float | int]:
    summary = payload["single_turn_summary"]
    overall = summary.get("overall", summary)
    return {
        "total": int(overall["total"]),
        "route_pct": float(overall["route_hit"]["rate_pct"]),
        "quality_pct": float(overall["quality_pass"]["rate_pct"]),
        "transport_pct": float(overall["transport_ok"]["rate_pct"]),
    }


def _gold_metrics(payload: dict[str, Any]) -> dict[str, float | int]:
    return {
        "total": int(payload["total"]),
        "route_pct": round(float(payload["scenario_accuracy"]) * 100, 2),
        "quality_pct": round(float(payload["scenario_accuracy"]) * 100, 2),
        "transport_pct": 100.0 if not payload.get("errors") else 0.0,
    }


def compare() -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    errors: list[str] = []
    for name, baseline_path, current_path, report_type in CORPORA:
        extractor = _gold_metrics if report_type == "gold" else _live_metrics
        baseline = extractor(_read(baseline_path))
        current = extractor(_read(current_path))
        route_delta = round(float(current["route_pct"]) - float(baseline["route_pct"]), 2)
        quality_delta = round(float(current["quality_pct"]) - float(baseline["quality_pct"]), 2)
        checks = {
            "same_case_count": current["total"] == baseline["total"],
            "transport_100_pct": current["transport_pct"] == 100.0,
            "route_degradation_within_2pp": route_delta >= -MAX_DEGRADATION_PP,
            "quality_degradation_within_2pp": quality_delta >= -MAX_DEGRADATION_PP,
        }
        failed = [key for key, passed in checks.items() if not passed]
        if failed:
            errors.append(f"{name}: {', '.join(failed)}")
        records.append({
            "corpus": name,
            "baseline_report": str(baseline_path),
            "current_report": str(current_path),
            "baseline": baseline,
            "current": current,
            "delta_pp": {"route": route_delta, "quality": quality_delta},
            "checks": checks,
            "passed": not failed,
        })
    return {
        "schema_version": 1,
        "knowledge_version": "2026.08.23.1",
        "gate": {"maximum_allowed_degradation_pp": MAX_DEGRADATION_PP},
        "passed": not errors,
        "errors": errors,
        "corpora": records,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare KB v3.1 regressions with frozen baselines.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = compare()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
