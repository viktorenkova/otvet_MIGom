from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from backend.app.bot.knowledge_search import _load_v2_articles, _semantic_config, clear_knowledge_cache
from backend.app.bot.semantic_search import MultilingualHybridSemanticIndex
from backend.app.bot.scenario_engine import load_scenarios


DEFAULT_DATASET = Path("tests/data/routing_v3_independent_acceptance.json")
DEFAULT_OUTPUT = Path("reports/candidate-retrieval-v31-independent-116.json")


def _metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    passed = sum(item["hit_at_10"] for item in rows)
    return {
        "total": total,
        "passed": passed,
        "recall_at_10_pct": round(passed / total * 100, 2) if total else 0.0,
    }


def evaluate(dataset: dict[str, Any]) -> dict[str, Any]:
    clear_knowledge_cache()
    articles = _load_v2_articles()
    config = _semantic_config()
    index = MultilingualHybridSemanticIndex(articles, config)
    dense_rows = index.dense_similarities_many([case["text"] for case in dataset["cases"]])
    active_ids = {scenario.scenario_id for scenario in load_scenarios()}
    public_ids = {
        article.slug
        for article in articles
        if article.slug in active_ids
        and article.section in {"public", "guest"}
        and article.intent != "prohibited"
    }
    rows: list[dict[str, Any]] = []
    for case_index, case in enumerate(dataset["cases"]):
        ranked = index.rank(
            case["text"],
            public_ids,
            "unknown",
            top_k=10,
            dense_similarities=(dense_rows[case_index] if dense_rows is not None else None),
            lexical_weight=float(config.get("candidate_lexical_weight", 0.75)),
            dense_weight=float(config.get("candidate_dense_weight", 0.25)),
        )
        candidate_ids = [item.article_id for item in ranked]
        expected_ids = [
            item for item in case["expected"].get("expected_scenario_ids", []) if item
        ]
        rows.append({
            "id": case["id"],
            "group": case["group"],
            "class": case["class"],
            "expected_scenario_ids": expected_ids,
            "candidate_scenario_ids": candidate_ids,
            "hit_at_10": any(item in candidate_ids for item in expected_ids),
            "dense_available": bool(ranked and ranked[0].dense_available),
        })
    by_group = {
        group: _metrics([item for item in rows if item["group"] == group])
        for group in sorted({item["group"] for item in rows})
    }
    by_class = {
        case_class: _metrics([item for item in rows if item["class"] == case_class])
        for case_class in sorted({item["class"] for item in rows})
    }
    overall = _metrics(rows)
    checks = {
        "blind_recall_at_10_gte_95": overall["recall_at_10_pct"] >= 95.0,
        "each_group_recall_at_10_gte_90": all(
            item["recall_at_10_pct"] >= 90.0 for item in by_group.values()
        ),
        "each_language_class_recall_at_10_gte_90": all(
            item["recall_at_10_pct"] >= 90.0 for item in by_class.values()
        ),
        "dense_channel_available": all(item["dense_available"] for item in rows),
    }
    return {
        "schema_version": 1,
        "dataset": {
            "path": str(DEFAULT_DATASET),
            "version": dataset["version"],
            "case_count": dataset["case_count"],
            "cases_sha256": dataset["cases_sha256"],
            "policy": dataset["purpose"],
        },
        "retrieval_contract": {
            "output": "Top-10 candidates only; no final scenario decision",
            "fusion_weights": {
                "lexical": float(config.get("candidate_lexical_weight", 0.75)),
                "dense": float(config.get("candidate_dense_weight", 0.25)),
            },
        },
        "overall": overall,
        "by_group": by_group,
        "by_class": by_class,
        "checks": checks,
        "passed": all(checks.values()),
        "failures": [item for item in rows if not item["hit_at_10"]],
        "results": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate candidate Recall@10 on the independent 116-case set.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
    result = evaluate(dataset)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("overall", "by_class", "checks", "passed")}, ensure_ascii=False, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
