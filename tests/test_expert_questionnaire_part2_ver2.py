import json
from pathlib import Path

import pytest

from backend.app.bot.scenario_engine import load_scenarios, match_scenario


@pytest.fixture(autouse=True)
def clear_scenario_cache():
    load_scenarios.cache_clear()
    yield
    load_scenarios.cache_clear()


def _scenario_map():
    payload = json.loads(Path("knowledge/v2/scenarios.json").read_text(encoding="utf-8"))
    return {item["scenario_id"]: item for item in payload["records"]}


def test_approved_commission_list_and_deadline_rule_are_published():
    scenarios = _scenario_map()
    commission = scenarios["commission.explained"]
    overdue = scenarios["commission.unpaid"]

    assert any("перечень комиссий исчерпывается" in fact for fact in commission["facts"])
    assert "иных комиссий нет" in commission["detailed_answer"]
    assert any("5 рабочих дней" in fact for fact in overdue["facts"])
    assert any("10 календарных дней" in fact for fact in overdue["facts"])


@pytest.mark.parametrize(
    ("message", "scenario_id"),
    [
        ("какие причины отказа не принимаются", "refusal.invalid_reasons"),
        ("какие доказательства приложить к отказу", "refusal.evidence"),
        ("куда отправить акт об отказе", "refusal.deadline_and_submission"),
        ("кто решает по мотивированному отказу", "refusal.seller_decision"),
    ],
)
def test_approved_refusal_answers_route_to_specific_scenarios(message, scenario_id):
    decision = match_scenario(message, "authorized")
    assert decision.confidence == "high"
    assert decision.scenario is not None
    assert decision.scenario.scenario_id == scenario_id


def test_all_static_knowledge_is_public_for_guest_and_authorized_users():
    scenarios = _scenario_map()
    assert scenarios
    for scenario in scenarios.values():
        assert scenario["roles"] == ["guest", "authorized"]

    guest_decision = match_scenario("какие комиссии есть", "guest")
    assert guest_decision.scenario is not None
    assert guest_decision.scenario.scenario_id == "commission.explained"


def test_demo_tariff_scope_and_confirmed_identification_are_recorded():
    scenarios = _scenario_map()
    joined = " ".join(
        fact
        for scenario_id in ("tariff.demo", "tariff.choose", "account.identification_for_contract")
        for fact in scenarios[scenario_id]["facts"]
    )
    assert "Демо-режим" in joined
    assert "разделе «Имущество»" in joined
    assert "телефона и электронной почты" in joined
    assert "паспортные данные" in joined
    assert "ИП или организация" in joined
