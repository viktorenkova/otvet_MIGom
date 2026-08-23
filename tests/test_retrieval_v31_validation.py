from __future__ import annotations

import json
from pathlib import Path

from backend.tools.build_retrieval_v31_validation import build


ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "tests/data/retrieval_v31_development_validation.json"
REPORT_PATH = ROOT / "reports/retrieval-v31-development-validation.json"


def _payload(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_retrieval_dataset_is_deterministic_and_has_no_frozen_phrase_overlap() -> None:
    committed = _payload(DATASET_PATH)
    rebuilt = build(
        ROOT / "knowledge/v3_1/scenarios.json",
        ROOT / "tests/data/regression_corpora_manifest.json",
    )

    assert rebuilt == committed
    assert committed["case_count"] == 471
    assert committed["development_count"] == 372
    assert committed["validation_count"] == 99
    assert committed["exact_frozen_overlap_count"] == 0
    assert not any(item["exact_frozen_overlap"] for item in committed["cases"])


def test_retrieval_development_and_validation_gate_passes() -> None:
    report = _payload(REPORT_PATH)

    assert report["passed"] is True
    assert report["development"]["recall_at_10_pct"] >= 97.0
    assert report["validation"]["recall_at_10_pct"] >= 97.0
    assert all(item["recall_at_10_pct"] >= 90.0 for item in report["by_variant"].values())
