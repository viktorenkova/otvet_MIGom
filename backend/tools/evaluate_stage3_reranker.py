from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import time
from typing import Any

from backend.app.bot.scenario_reranker import (
    ConstrainedFeatureClassifier,
    CrossEncoderScenarioReranker,
    scenario_family,
)


DEFAULT_DATASET = Path("tests/data/retrieval_v31_development_validation.json")
DEFAULT_CANDIDATES = Path("reports/semantic-retrieval-v31-development-validation.json")
DEFAULT_HARD_NEGATIVES = Path("tests/data/stage3_hard_negatives.json")
DEFAULT_OUTPUT = Path("reports/stage3-reranker-evaluation.json")
REQUIRED_FAMILIES = {"onboarding", "payments", "registration", "bidding", "transfer", "search"}


def _candidate_rows(ids: list[str]) -> list[dict[str, Any]]:
    return [
        {"scenario_id": scenario_id, "score": max(0.0, 1.0 - index * 0.08)}
        for index, scenario_id in enumerate(ids)
    ]


def _metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    correct = sum(row["correct"] for row in rows)
    return {"total": total, "correct": correct, "top1_pct": round(correct / total * 100, 2) if total else 0.0}


def _calibrate(rows: list[dict[str, Any]]) -> dict[str, float]:
    thresholds: dict[str, float] = {}
    families = sorted({row["family"] for row in rows} | {"default"})
    for family in families:
        family_rows = [row for row in rows if row["family"] == family] or rows
        values = sorted({round(float(row["probability"]), 6) for row in family_rows})
        chosen = 1.01
        for threshold in values:
            high = [row for row in family_rows if row["probability"] >= threshold and row["margin"] >= 0.01]
            if not high:
                continue
            wrong_rate = sum(not row["correct"] for row in high) / len(high)
            coverage = len(high) / len(family_rows)
            if wrong_rate <= 0.02 and coverage >= 0.50:
                chosen = threshold
                break
        thresholds[family] = round(chosen, 6)
    return thresholds


def _confidence_metrics(rows: list[dict[str, Any]], thresholds: dict[str, float]) -> dict[str, Any]:
    high = [
        row for row in rows
        if row["probability"] >= thresholds.get(row["family"], thresholds["default"])
        and row["margin"] >= 0.01
    ]
    wrong = sum(not row["correct"] for row in high)
    return {
        "high_count": len(high),
        "high_coverage_pct": round(len(high) / len(rows) * 100, 2) if rows else 0.0,
        "confident_wrong_count": wrong,
        "confident_wrong_pct": round(wrong / len(high) * 100, 2) if high else 0.0,
    }


def _run_provider(provider: Any, cases: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], float]:
    started = time.perf_counter()
    rows = []
    for case in cases:
        ranked = provider.rerank(case["text"], case["candidates"])
        top_id = ranked[0].scenario_id if ranked else None
        probability = ranked[0].probability if ranked else 0.0
        second = ranked[1].probability if len(ranked) > 1 else 0.0
        rows.append({
            "id": case["id"], "split": case["split"], "family": case["family"],
            "expected_scenario_ids": case["expected_scenario_ids"], "top_scenario_id": top_id,
            "correct": top_id in case["expected_scenario_ids"], "probability": probability,
            "margin": probability - second,
        })
    elapsed_ms = (time.perf_counter() - started) * 1000
    return rows, elapsed_ms


def evaluate(dataset: dict, candidate_report: dict, hard_negatives: dict) -> dict:
    candidate_by_id = {row["id"]: row for row in candidate_report["results"]}
    cases = []
    for case in dataset["cases"]:
        source = candidate_by_id[case["id"]]
        candidates = [
            {"scenario_id": item["scenario_id"], "score": item["score"]}
            for item in source["candidates"]
        ]
        cases.append({
            "id": case["id"], "split": case["split"], "text": case["text"],
            "expected_scenario_ids": [case["expected_scenario_id"]],
            "family": scenario_family(case["expected_scenario_id"]), "candidates": candidates,
        })
    hard_cases = [
        {
            "id": case["id"], "split": case["split"], "text": case["text"],
            "expected_scenario_ids": case["expected_scenario_ids"],
            "family": scenario_family(case["expected_scenario_ids"][0]),
            "candidates": _candidate_rows(case["historical_top10_candidate_ids"]),
        }
        for case in hard_negatives["cases"]
    ]

    baseline_rows = []
    for case in cases:
        top_id = case["candidates"][0]["scenario_id"]
        baseline_rows.append({**case, "correct": top_id in case["expected_scenario_ids"]})

    feature_rows, feature_ms = _run_provider(ConstrainedFeatureClassifier(), cases)
    cross_encoder = CrossEncoderScenarioReranker()
    cross_rows: list[dict[str, Any]] = []
    cross_hard_rows: list[dict[str, Any]] = []
    cross_ms = 0.0
    if cross_encoder.available:
        cross_rows, cross_ms = _run_provider(cross_encoder, cases)
        cross_hard_rows, hard_ms = _run_provider(cross_encoder, hard_cases)
        cross_ms += hard_ms

    development = [row for row in cross_rows if row["split"] == "development"]
    validation = [row for row in cross_rows if row["split"] == "validation"]
    thresholds = _calibrate(development) if development else {"default": 1.01}
    by_family = {
        family: _metrics([row for row in validation if row["family"] == family])
        for family in sorted(REQUIRED_FAMILIES)
    }
    validation_confidence = _confidence_metrics(validation, thresholds) if validation else {}
    checks = {
        "cross_encoder_available": cross_encoder.available,
        "validation_top1_gte_90": _metrics(validation)["top1_pct"] >= 90.0 if validation else False,
        "each_required_family_gte_80": all(item["total"] > 0 and item["top1_pct"] >= 80.0 for item in by_family.values()),
        "confident_wrong_lte_2": validation_confidence.get("confident_wrong_pct", 100.0) <= 2.0,
        "high_confidence_coverage_gte_50": validation_confidence.get("high_coverage_pct", 0.0) >= 50.0,
    }
    return {
        "schema_version": 1,
        "same_top10_contract": True,
        "datasets": {
            "development_validation": {"version": dataset["version"], "cases_sha256": dataset["cases_sha256"]},
            "hard_negatives": {"version": hard_negatives["version"], "cases_sha256": hard_negatives["cases_sha256"]},
        },
        "comparison": {
            "candidate_order_baseline": {
                "development": _metrics([row for row in baseline_rows if row["split"] == "development"]),
                "validation": _metrics([row for row in baseline_rows if row["split"] == "validation"]),
            },
            "constrained_feature_classifier": {
                "implementation": "deterministic constrained candidate-ID classifier",
                "development": _metrics([row for row in feature_rows if row["split"] == "development"]),
                "validation": _metrics([row for row in feature_rows if row["split"] == "validation"]),
                "latency_total_ms": round(feature_ms, 2),
            },
            "constrained_llm_classifier": {
                "available": False,
                "reason": "LLM is disabled in the approved local evaluation contour; no result is fabricated.",
            },
            "cross_encoder": {
                "available": cross_encoder.available, "error": cross_encoder.error,
                "model": cross_encoder.model_name, "revision": "main", "device": "cpu",
                "development": _metrics(development), "validation": _metrics(validation),
                "hard_negatives": _metrics(cross_hard_rows),
                "latency_total_ms": round(cross_ms, 2),
                "estimated_usd_per_1000_queries": 0.0,
            },
        },
        "calibration": {
            "method": "lowest per-family softmax threshold with <=2% high-confidence error and >=50% development coverage; margin >=0.01",
            "thresholds": thresholds,
            "validation": validation_confidence,
        },
        "validation_by_required_family": by_family,
        "selected": {
            "provider": "cross_encoder" if all(checks.values()) else None,
            "model": cross_encoder.model_name if all(checks.values()) else None,
            "reason": "Only an available provider passing every validation and confidence gate is selected.",
        },
        "checks": checks,
        "passed": all(checks.values()),
        "failures": [row for row in validation if not row["correct"]],
        "hard_negative_failures": [row for row in cross_hard_rows if not row["correct"]],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--hard-negatives", type=Path, default=DEFAULT_HARD_NEGATIVES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = evaluate(
        json.loads(args.dataset.read_text(encoding="utf-8")),
        json.loads(args.candidates.read_text(encoding="utf-8")),
        json.loads(args.hard_negatives.read_text(encoding="utf-8")),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("comparison", "calibration", "validation_by_required_family", "checks", "passed")}, ensure_ascii=False, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
