from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from backend.app.bot.knowledge_search import _load_v2_articles, _semantic_config, clear_knowledge_cache
from backend.app.bot.semantic_search import MultilingualHybridSemanticIndex


DEFAULT_DATASET = Path("tests/data/retrieval_v31_development_validation.json")
DEFAULT_OUTPUT = Path("reports/semantic-retrieval-v31-development-validation.json")


def _metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    return {
        "total": total,
        "recall_at_1_pct": round(sum(item["hit_at_1"] for item in rows) / total * 100, 2) if total else 0.0,
        "recall_at_5_pct": round(sum(item["hit_at_5"] for item in rows) / total * 100, 2) if total else 0.0,
        "recall_at_10_pct": round(sum(item["hit_at_10"] for item in rows) / total * 100, 2) if total else 0.0,
    }


def evaluate(dataset: dict[str, Any]) -> dict[str, Any]:
    clear_knowledge_cache()
    articles = _load_v2_articles()
    config = _semantic_config()
    index = MultilingualHybridSemanticIndex(articles, config)
    article_intents = {article.slug: article.intent for article in articles}
    dense_rows = index.dense_similarities_many([case["text"] for case in dataset["cases"]])
    rows: list[dict[str, Any]] = []
    for case_index, case in enumerate(dataset["cases"]):
        allowed_sections = {"public", "guest"} if case["role"] == "guest" else {"public", "guest", "authorized"}
        allowed_ids = {
            article.slug for article in articles if article.section in allowed_sections
        }
        ranked = index.rank(
            case["text"],
            allowed_ids,
            "unknown",
            top_k=10,
            dense_similarities=(dense_rows[case_index] if dense_rows is not None else None),
            lexical_weight=float(config.get("candidate_lexical_weight", 0.75)),
            dense_weight=float(config.get("candidate_dense_weight", 0.25)),
        )
        candidate_ids = [item.article_id for item in ranked]
        expected = case["expected_scenario_id"]
        rows.append({
            "id": case["id"],
            "split": case["split"],
            "variant": case["variant"],
            "expected_scenario_id": expected,
            "expected_intent": article_intents.get(expected, "unknown"),
            "candidate_scenario_ids": candidate_ids,
            "candidates": [
                {
                    "scenario_id": item.article_id,
                    "score": round(item.score, 6),
                    "channels": {
                        "lexical": round(item.lexical_score, 6),
                        "char": round(item.char_score, 6),
                        "word": round(item.word_score, 6),
                        "dense": round(item.dense_score, 6),
                        "dense_similarity": round(item.dense_similarity, 6),
                        "intent_boost": round(item.intent_boost, 6),
                    },
                }
                for item in ranked
            ],
            "dense_available": bool(ranked and ranked[0].dense_available),
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
    by_intent = {
        intent: _metrics([item for item in rows if item["expected_intent"] == intent])
        for intent in sorted({item["expected_intent"] for item in rows})
    }
    major_intents = {
        intent: metrics for intent, metrics in by_intent.items() if metrics["total"] >= 10
    }
    dev_metrics = _metrics(development)
    validation_metrics = _metrics(validation)
    checks = {
        "no_exact_frozen_overlap": dataset["exact_frozen_overlap_count"] == 0,
        "development_recall_at_1_gte_90": dev_metrics["recall_at_1_pct"] >= 90.0,
        "validation_recall_at_1_gte_90": validation_metrics["recall_at_1_pct"] >= 90.0,
        "development_recall_at_10_gte_97": dev_metrics["recall_at_10_pct"] >= 97.0,
        "validation_recall_at_10_gte_95": validation_metrics["recall_at_10_pct"] >= 95.0,
        "each_variant_recall_at_10_gte_90": all(item["recall_at_10_pct"] >= 90.0 for item in by_variant.values()),
        "each_major_intent_recall_at_10_gte_90": all(
            item["recall_at_10_pct"] >= 90.0 for item in major_intents.values()
        ),
        "dense_channel_available": all(item["dense_available"] for item in rows),
    }
    return {
        "schema_version": 2,
        "retrieval_contract": {
            "output": "Top-10 candidates only; no final scenario decision",
            "normalizer": "backend.app.bot.routing_v3.routing_normalize",
            "channels": ["lexical", "char", "word", "dense", "intent_boost"],
            "fusion_weights": {
                "lexical": float(config.get("candidate_lexical_weight", 0.75)),
                "dense": float(config.get("candidate_dense_weight", 0.25)),
            },
        },
        "dataset": {
            "version": dataset["version"],
            "case_count": dataset["case_count"],
            "cases_sha256": dataset["cases_sha256"],
        },
        "development": dev_metrics,
        "validation": validation_metrics,
        "by_variant": by_variant,
        "by_intent": by_intent,
        "checks": checks,
        "passed": all(checks.values()),
        "failures": [item for item in rows if not item["hit_at_10"]],
        "results": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate semantic fallback Top-K on KB v3.1 cases.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = evaluate(json.loads(args.dataset.read_text(encoding="utf-8")))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("development", "validation", "by_variant", "by_intent", "checks", "passed")}, ensure_ascii=False, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
