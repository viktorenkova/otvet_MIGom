from __future__ import annotations

import argparse
import json
from pathlib import Path

from backend.app.bot.knowledge_search import _load_v2_articles, _semantic_config, clear_knowledge_cache
from backend.app.bot.semantic_search import MultilingualHybridSemanticIndex
from backend.app.bot.scenario_engine import load_scenarios


DEFAULT_DATASET = Path("tests/data/stage2_language_validation.json")
DEFAULT_OUTPUT = Path("reports/stage2-language-validation.json")


def _metrics(rows: list[dict]) -> dict:
    total = len(rows)
    return {
        "total": total,
        "passed": sum(row["hit_at_10"] for row in rows),
        "recall_at_10_pct": round(sum(row["hit_at_10"] for row in rows) / total * 100, 2) if total else 0.0,
    }


def evaluate(dataset: dict) -> dict:
    clear_knowledge_cache()
    articles = _load_v2_articles()
    config = _semantic_config()
    index = MultilingualHybridSemanticIndex(articles, config)
    dense_rows = index.dense_similarities_many([case["text"] for case in dataset["cases"]])
    active_ids = {scenario.scenario_id for scenario in load_scenarios()}
    public_ids = {article.slug for article in articles if article.slug in active_ids and article.section in {"public", "guest"} and article.intent != "prohibited"}
    rows = []
    for offset, case in enumerate(dataset["cases"]):
        ranked = index.rank(
            case["text"], public_ids, "unknown", top_k=10,
            dense_similarities=dense_rows[offset] if dense_rows is not None else None,
            lexical_weight=float(config.get("candidate_lexical_weight", 0.75)),
            dense_weight=float(config.get("candidate_dense_weight", 0.25)),
        )
        candidate_ids = [item.article_id for item in ranked]
        rows.append({
            "id": case["id"], "class": case["class"], "text": case["text"],
            "expected_scenario_ids": case["expected_scenario_ids"],
            "candidate_scenario_ids": candidate_ids,
            "hit_at_10": any(expected in candidate_ids for expected in case["expected_scenario_ids"]),
            "dense_available": bool(ranked and ranked[0].dense_available),
        })
    by_class = {name: _metrics([row for row in rows if row["class"] == name]) for name in sorted({row["class"] for row in rows})}
    checks = {
        "no_exact_frozen_overlap": dataset["exact_frozen_overlap_count"] == 0,
        "overall_recall_at_10_gte_97": _metrics(rows)["recall_at_10_pct"] >= 97.0,
        "each_language_class_recall_at_10_gte_90": all(value["recall_at_10_pct"] >= 90.0 for value in by_class.values()),
        "dense_channel_available": all(row["dense_available"] for row in rows),
    }
    return {
        "schema_version": 1,
        "dataset": {key: dataset[key] for key in ("version", "case_count", "cases_sha256", "purpose")},
        "overall": _metrics(rows), "by_class": by_class, "checks": checks,
        "passed": all(checks.values()), "failures": [row for row in rows if not row["hit_at_10"]], "results": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = evaluate(json.loads(args.dataset.read_text(encoding="utf-8")))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("overall", "by_class", "checks", "passed")}, ensure_ascii=False, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
