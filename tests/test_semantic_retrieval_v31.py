from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "reports/semantic-retrieval-v31-development-validation.json"
BLIND_REPORT_PATH = ROOT / "reports/candidate-retrieval-v31-independent-116.json"


def test_semantic_fallback_has_balanced_top1_and_top10_recall() -> None:
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))

    assert report["passed"] is True
    assert report["development"]["recall_at_1_pct"] >= 90.0
    assert report["validation"]["recall_at_1_pct"] >= 90.0
    assert report["development"]["recall_at_10_pct"] >= 97.0
    assert report["validation"]["recall_at_10_pct"] >= 95.0
    assert all(item["recall_at_10_pct"] >= 90.0 for item in report["by_variant"].values())
    assert all(
        item["recall_at_10_pct"] >= 90.0
        for item in report["by_intent"].values()
        if item["total"] >= 10
    )
    assert report["checks"]["dense_channel_available"] is True
    assert report["retrieval_contract"]["output"].startswith("Top-10 candidates only")
    assert report["results"]
    first = report["results"][0]
    assert len(first["candidates"]) == 10
    assert set(first["candidates"][0]["channels"]) == {
        "lexical",
        "char",
        "word",
        "dense",
        "dense_similarity",
        "intent_boost",
    }


def test_candidate_retrieval_independent_gate_passes() -> None:
    report = json.loads(BLIND_REPORT_PATH.read_text(encoding="utf-8"))

    assert report["passed"] is True
    assert report["overall"]["recall_at_10_pct"] >= 95.0
    assert all(item["recall_at_10_pct"] >= 90.0 for item in report["by_group"].values())
    assert all(item["recall_at_10_pct"] >= 90.0 for item in report["by_class"].values())
    assert report["checks"]["dense_channel_available"] is True
