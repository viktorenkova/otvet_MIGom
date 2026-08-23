from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "reports/semantic-retrieval-v31-development-validation.json"


def test_semantic_fallback_has_balanced_top1_and_top10_recall() -> None:
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))

    assert report["passed"] is True
    assert report["development"]["recall_at_1_pct"] >= 90.0
    assert report["validation"]["recall_at_1_pct"] >= 90.0
    assert report["development"]["recall_at_10_pct"] >= 97.0
    assert report["validation"]["recall_at_10_pct"] >= 95.0
    assert all(item["recall_at_10_pct"] >= 90.0 for item in report["by_variant"].values())
