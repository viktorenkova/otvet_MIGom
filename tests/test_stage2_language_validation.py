from __future__ import annotations

import json
from pathlib import Path

from backend.app.bot import text_processing
from backend.app.bot.text_processing import normalize_matching_text
from backend.tools.build_stage2_language_validation import build


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "tests/data/stage2_language_validation.json"
REPORT = ROOT / "reports/stage2-language-validation.json"


def test_matching_normalizer_handles_layout_transliteration_and_slang() -> None:
    assert normalize_matching_text("rfr cltkfnm cnfdre") == normalize_matching_text("как сделать ставку")
    assert normalize_matching_text("kak sdelat stavku") == normalize_matching_text("как сделать ставку")
    assert normalize_matching_text("доки") == normalize_matching_text("документы")
    assert normalize_matching_text("тачка") == normalize_matching_text("машина")
    assert normalize_matching_text("ordinary english text") == "ordinary english text"


def test_matching_token_repair_resolves_equal_scores_deterministically(monkeypatch) -> None:
    monkeypatch.setattr(
        text_processing,
        "_matching_vocabulary",
        lambda: frozenset({"абвгдежзийка", "абвгдежзийкб"}),
    )

    assert text_processing._repair_matching_token("абвгдежзийкв") == "абвгдежзийка"


def test_language_dataset_is_deterministic_and_not_copied_from_frozen_queries() -> None:
    committed = json.loads(DATASET.read_text(encoding="utf-8"))
    assert build(ROOT / "tests/data/regression_corpora_manifest.json") == committed
    assert committed["case_count"] == 30
    assert committed["exact_frozen_overlap_count"] == 0
    assert {case["class"] for case in committed["cases"]} == {"keyboard_layout", "transliteration", "slang"}


def test_language_candidate_retrieval_gate_passes() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    assert report["passed"] is True
    assert report["overall"]["recall_at_10_pct"] >= 97.0
    assert all(item["recall_at_10_pct"] >= 90.0 for item in report["by_class"].values())
