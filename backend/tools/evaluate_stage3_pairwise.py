from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from backend.app.bot.pairwise_reranker import pairwise_features
from backend.app.bot.scenario_reranker import scenario_family
from backend.tools.evaluate_stage3_reranker import _calibrate, _confidence_metrics


DEFAULT_DATASET = Path("tests/data/retrieval_v31_development_validation.json")
DEFAULT_CANDIDATES = Path("reports/semantic-retrieval-v31-development-validation.json")
DEFAULT_HARD = Path("tests/data/stage3_hard_negatives.json")
DEFAULT_OUTPUT = Path("reports/stage3-pairwise-reranker-evaluation.json")
REQUIRED_FAMILIES = {"onboarding", "payments", "registration", "bidding", "transfer", "search"}


def _cases(dataset: dict, candidate_report: dict, hard: dict) -> list[dict[str, Any]]:
    by_id = {row["id"]: row for row in candidate_report["results"]}
    result = []
    for case in dataset["cases"]:
        result.append({
            "id": case["id"], "split": case["split"], "source": "taxonomy",
            "text": case["text"], "expected": [case["expected_scenario_id"]],
            "family": scenario_family(case["expected_scenario_id"]),
            "candidates": by_id[case["id"]]["candidates"],
        })
    for case in hard["cases"]:
        result.append({
            "id": case["id"], "split": case["split"], "source": "real_hard_negative",
            "text": case["text"], "expected": case["expected_scenario_ids"],
            "family": scenario_family(case["expected_scenario_ids"][0]),
            "candidates": [
                {"scenario_id": item, "score": 0.0, "channels": {}}
                for rank, item in enumerate(case["historical_top10_candidate_ids"])
            ],
        })
    return result


def _metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    correct = sum(row["correct"] for row in rows)
    return {"total": total, "correct": correct, "top1_pct": round(correct / total * 100, 2) if total else 0.0}


def evaluate(dataset: dict, candidate_report: dict, hard: dict) -> dict:
    cases = _cases(dataset, candidate_report, hard)
    development = [case for case in cases if case["split"] == "development"]
    validation = [case for case in cases if case["split"] == "validation"]
    x_train, y_train, weights = [], [], []
    for case in development:
        for rank, candidate in enumerate(case["candidates"]):
            positive = candidate["scenario_id"] in case["expected"]
            x_train.append(pairwise_features(case["text"], candidate, rank))
            y_train.append(int(positive))
            weights.append(9.0 if positive else 1.0)
    model = HistGradientBoostingClassifier(
        learning_rate=0.06, max_iter=180, max_leaf_nodes=15,
        min_samples_leaf=12, l2_regularization=1.0, random_state=20260824,
    )
    model.fit(np.asarray(x_train, dtype=np.float32), np.asarray(y_train), sample_weight=np.asarray(weights))
    def score_cases(source_cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
        scored = []
        for case in source_cases:
            feature_rows = np.asarray([
                pairwise_features(case["text"], candidate, rank)
                for rank, candidate in enumerate(case["candidates"])
            ], dtype=np.float32)
            probabilities = model.predict_proba(feature_rows)[:, 1]
            order = np.argsort(-probabilities)
            top = case["candidates"][int(order[0])]["scenario_id"]
            second = float(probabilities[int(order[1])]) if len(order) > 1 else 0.0
            scored.append({
                "id": case["id"], "source": case["source"], "family": case["family"],
                "expected_scenario_ids": case["expected"], "top_scenario_id": top,
                "correct": top in case["expected"], "probability": float(probabilities[int(order[0])]),
                "margin": float(probabilities[int(order[0])]) - second,
            })
        return scored

    development_rows = score_cases(development)
    rows = score_cases(validation)
    taxonomy_rows = [row for row in rows if row["source"] == "taxonomy"]
    hard_rows = [row for row in rows if row["source"] == "real_hard_negative"]
    by_family = {
        family: _metrics([row for row in taxonomy_rows if row["family"] == family])
        for family in sorted(REQUIRED_FAMILIES)
    }
    thresholds = _calibrate(development_rows)
    confidence = _confidence_metrics(taxonomy_rows, thresholds)
    checks = {
        "validation_top1_gte_90": _metrics(taxonomy_rows)["top1_pct"] >= 90.0,
        "each_required_family_gte_80": all(item["total"] > 0 and item["top1_pct"] >= 80.0 for item in by_family.values()),
        "hard_negative_holdout_gte_50": _metrics(hard_rows)["top1_pct"] >= 50.0,
        "confident_wrong_lte_2": confidence["confident_wrong_pct"] <= 2.0,
        "high_confidence_coverage_gte_50": confidence["high_coverage_pct"] >= 50.0,
    }
    return {
        "schema_version": 1,
        "model": {
            "type": "HistGradientBoostingClassifier pointwise pair scorer",
            "random_state": 20260824,
            "feature_count": len(x_train[0]),
            "training_pairs": len(x_train),
            "query_text_usage": "training/evaluation only; never emitted into rules, aliases or KB",
        },
        "validation": _metrics(taxonomy_rows),
        "hard_negative_holdout": _metrics(hard_rows),
        "calibration": {
            "method": "per-family development threshold; probability and margin >=0.01",
            "thresholds": thresholds,
            "validation": confidence,
        },
        "validation_by_required_family": by_family,
        "checks": checks,
        "passed": all(checks.values()),
        "failures": [row for row in rows if not row["correct"]],
        "results": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--hard-negatives", type=Path, default=DEFAULT_HARD)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = evaluate(
        json.loads(args.dataset.read_text(encoding="utf-8")),
        json.loads(args.candidates.read_text(encoding="utf-8")),
        json.loads(args.hard_negatives.read_text(encoding="utf-8")),
    )
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("validation", "hard_negative_holdout", "validation_by_required_family", "checks", "passed")}, ensure_ascii=False, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
