from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib

from backend.app.bot.scenario_engine import load_scenarios


ROOT = Path(__file__).resolve().parents[1]


def _json(relative_path: str) -> dict:
    return json.loads((ROOT / relative_path).read_text(encoding="utf-8"))


def _sha256(relative_path: str) -> str:
    return hashlib.sha256((ROOT / relative_path).read_bytes()).hexdigest()


def test_pairwise_reranker_passes_stage3_gate() -> None:
    report = _json("reports/stage3-pairwise-reranker-evaluation.json")
    assert report["passed"] is True
    assert report["validation"]["top1_pct"] >= 90.0
    assert report["calibration"]["validation"]["confident_wrong_pct"] <= 2.0
    assert all(
        metrics["total"] > 0 and metrics["top1_pct"] >= 80.0
        for metrics in report["validation_by_required_family"].values()
    )


def test_pairwise_artifact_is_versioned_and_bound_to_training_inputs() -> None:
    config = _json("configs/reranker_config.json")
    artifact_path = ROOT / config["model"]
    bundle = joblib.load(artifact_path)
    assert config["provider"] == "pairwise"
    assert bundle["model_version"] == config["model_version"]
    assert bundle["inputs"]["retrieval_v31_development_validation.json"] == _sha256(
        "tests/data/retrieval_v31_development_validation.json"
    )
    assert bundle["inputs"]["semantic-retrieval-v31-development-validation.json"] == _sha256(
        "reports/semantic-retrieval-v31-development-validation.json"
    )
    assert bundle["inputs"]["stage3_hard_negatives.json"] == _sha256(
        "tests/data/stage3_hard_negatives.json"
    )


def test_candidate_reports_contain_only_active_scenarios() -> None:
    active = {scenario.scenario_id for scenario in load_scenarios()}
    for report_path in (
        "reports/semantic-retrieval-v31-development-validation.json",
        "reports/candidate-retrieval-v31-independent-116.json",
        "reports/stage2-language-validation.json",
    ):
        report = _json(report_path)
        candidate_ids = {
            candidate_id
            for row in report["results"]
            for candidate_id in row["candidate_scenario_ids"]
        }
        assert candidate_ids <= active


def test_stage3_runtime_preserves_live_quality_and_safety() -> None:
    live = _json("reports/quality-stage3-live-160.json")
    closed = _json("reports/quality-stage3-closed-270.json")
    assert live["single_turn_summary"]["overall"]["quality_pass"]["rate_pct"] >= 94.38
    assert live["release_gate"]["observed"]["confident_wrong_max"] <= 2
    assert closed["single_turn_summary"]["overall"]["quality_pass"]["rate_pct"] >= 78.37
    assert live["release_gate"]["observed"]["forbidden_content_ok_pct"] == 100.0
    assert closed["release_gate"]["observed"]["forbidden_content_ok_pct"] == 100.0
