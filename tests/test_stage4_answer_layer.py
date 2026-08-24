from __future__ import annotations

import json
from pathlib import Path

from backend.app.bot.answer_contracts import get_answer_contract, verify_answer
from backend.app.bot.answer_generator import generate_answer
from backend.app.bot.knowledge_search import get_article_by_id
from backend.tools.build_stage4_answer_contracts import build


ROOT = Path(__file__).resolve().parents[1]
CONTRACTS_PATH = ROOT / "knowledge/v3_1/answer_contracts.json"


def test_answer_contracts_are_deterministic_and_cover_every_active_scenario() -> None:
    committed = json.loads(CONTRACTS_PATH.read_text(encoding="utf-8"))
    rebuilt = build(
        ROOT / "knowledge/v3_1/scenarios.json",
        ROOT / "knowledge/v3_1/scenario_conflicts.json",
    )
    assert rebuilt == committed
    assert committed["record_count"] == 141
    assert {row["template_kind"] for row in committed["records"]} == {
        "direct", "clarification", "status", "contact"
    }


def test_fact_contract_references_are_valid_and_non_overlapping() -> None:
    payload = json.loads(CONTRACTS_PATH.read_text(encoding="utf-8"))
    known_fact_ids = {
        fact_id
        for row in payload["records"]
        for fact_id in row["facts"]
    }
    for row in payload["records"]:
        required = set(row["required_fact_ids"])
        allowed = set(row["allowed_fact_ids"])
        forbidden = set(row["forbidden_fact_ids"])
        assert row["approved_template"].strip()
        assert required
        assert required <= allowed <= known_fact_ids
        assert forbidden <= known_fact_ids
        assert allowed.isdisjoint(forbidden)
        assert row["llm_role"] == "wording_only"


def test_verifier_rejects_invented_amount_and_uses_deterministic_fallback() -> None:
    contract = get_answer_contract("buyer.get_started")
    assert contract is not None
    fallback = contract.approved_template
    candidate = fallback + " Гарантированный доход составит 500 000 рублей через 3 дня."
    result = verify_answer(candidate, fallback, contract)
    assert result.passed is False
    assert result.answer == fallback
    assert result.reason == "unsupported_protected_value"


def test_verifier_rejects_unrelated_wording_even_without_numbers() -> None:
    contract = get_answer_contract("buyer.get_started")
    assert contract is not None
    result = verify_answer(
        "Посетите районную налоговую инспекцию и оформите нотариальную доверенность.",
        contract.approved_template,
        contract,
    )
    assert result.passed is False
    assert result.answer == contract.approved_template
    assert result.reason == "insufficient_fact_support"


def test_deterministic_scenario_answer_returns_fact_trace() -> None:
    article = get_article_by_id("buyer.get_started", "guest")
    assert article is not None
    generated = generate_answer(
        "как начать участвовать в торгах",
        "bidding",
        "guest",
        article,
        False,
    )
    contract = get_answer_contract("buyer.get_started")
    assert contract is not None
    assert generated.answer
    assert generated.verification_passed is True
    assert generated.verification_reason == "deterministic_approved_template"
    assert set(generated.used_fact_ids) == set(contract.required_fact_ids)


def test_stage4_expert_gate_and_runtime_regression_pass() -> None:
    evaluation = json.loads(
        (ROOT / "reports/stage4-answer-layer-evaluation.json").read_text(encoding="utf-8")
    )
    runtime = json.loads(
        (ROOT / "reports/quality-stage4-live-160.json").read_text(encoding="utf-8")
    )
    assert evaluation["passed"] is True
    assert evaluation["all_criteria"]["rate_pct"] >= 93.0
    assert evaluation["critical_unsupported"]["count"] == 0
    assert evaluation["irrelevant_blocks"]["rate_pct"] <= 2.0
    assert runtime["single_turn_summary"]["overall"]["quality_pass"]["rate_pct"] >= 94.38
    assert runtime["release_gate"]["observed"]["forbidden_content_ok_pct"] == 100.0
