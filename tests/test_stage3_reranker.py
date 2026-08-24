from __future__ import annotations

import json
from pathlib import Path

from backend.app.bot.scenario_reranker import RerankedScenario, decide_reranked
from backend.tools.build_stage3_hard_negatives import build


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "tests/data/stage3_hard_negatives.json"


def test_hard_negative_dataset_is_deterministic_and_contains_real_conflicts() -> None:
    committed = json.loads(DATASET.read_text(encoding="utf-8"))
    rebuilt = build(ROOT / "reports/quality-stage0-adjudicated-error-taxonomy.json")
    assert rebuilt == committed
    assert committed["case_count"] == 32
    assert committed["development_count"] > committed["validation_count"] > 0
    assert all(case["expected_scenario_ids"] for case in committed["cases"])


def test_low_margin_reranker_requests_a_slot_instead_of_answering() -> None:
    ranked = (
        RerankedScenario("payment.not_visible", 1.0, 0.16, 0.7),
        RerankedScenario("payment.checkout_problem", 0.99, 0.155, 0.69),
    )
    decision = decide_reranked("проблема с оплатой", ranked)
    assert decision.scenario_id is None
    assert decision.confidence == "medium"
    assert decision.missing_slot == "states"
    assert decision.clarifying_options


def test_high_margin_reranker_can_select_scenario() -> None:
    ranked = (
        RerankedScenario("bid.place", 4.0, 0.72, 0.8),
        RerankedScenario("bid.not_visible", 1.0, 0.08, 0.5),
    )
    decision = decide_reranked("как сделать ставку", ranked)
    assert decision.scenario_id == "bid.place"
    assert decision.confidence == "high"
