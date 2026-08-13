import json
from pathlib import Path

import pytest

from backend.app.bot.scenario_engine import clear_scenario_cache, match_scenario


@pytest.fixture(autouse=True)
def clear_cache():
    clear_scenario_cache()
    yield
    clear_scenario_cache()


@pytest.mark.parametrize(
    ("message", "scenario_id"),
    [
        ("какие тарифы есть", "tariff.choose"),
        ("премиум или разовый", "tariff.choose"),
        ("что дает разовый тариф", "tariff.one_time"),
        ("премиум каждый год платить", "tariff.premium"),
        ("что дает демо режим", "tariff.demo"),
        ("можно делать ставки в демо", "tariff.demo"),
        ("пополнил кошелек но тариф не включился", "tariff.status"),
        ("есть скидка на тариф", "tariff.promotion"),
        ("что дает топ 10", "commission.discount"),
    ],
)
def test_tariff_questions_route_to_v2(message, scenario_id):
    decision = match_scenario(message, "authorized")
    assert decision.confidence == "high"
    assert decision.scenario is not None
    assert decision.scenario.scenario_id == scenario_id


def _answer(message):
    return match_scenario(message, "authorized").scenario.answer.casefold()


def test_tariff_answers_keep_confirmed_limits_and_do_not_guarantee_transfer():
    choose = _answer("какие тарифы есть")
    premium = _answer("что входит в премиум")
    demo = _answer("что дает демо режим")

    assert "обеспечительный платёж" in choose
    assert "не гарантирует передачу" in choose
    assert "один раз" in premium
    assert "до расторжения договора" in premium
    assert "не гарантирует передачу" in premium
    assert "ставкам и результатам" in demo
    assert "имущество" in demo
    assert "платный автомобильный тариф не нужен" in demo


def test_tariff_status_and_promotion_use_safe_manual_verification():
    status = _answer("пополнил кошелек но тариф не включился")
    promotion = _answer("обещали скидку на тариф")

    assert "не оплачивайте повторно" in status
    assert "подтверждение списания" in status
    assert "администрация проверит" in promotion
    assert "не обещает скидку" in promotion
    assert "не фиксирует цену" in promotion


def test_tariff_batch_is_linked_and_no_longer_blocks_migration():
    scenarios = json.loads(Path("knowledge/v2/scenarios.json").read_text(encoding="utf-8"))["records"]
    target_ids = {
        "tariff.choose",
        "tariff.one_time",
        "tariff.premium",
        "tariff.demo",
        "tariff.status",
        "tariff.promotion",
    }
    migrated_ids = {
        legacy_id
        for scenario in scenarios
        if scenario["scenario_id"] in target_ids
        for legacy_id in scenario["legacy_ids"]
    }
    migrated_ids.add("kb-107-программа-топ-10-покупателей")
    assert len(migrated_ids) == 26

    inventory = json.loads(Path("knowledge/v2/legacy_inventory.json").read_text(encoding="utf-8"))
    by_id = {item["legacy_id"]: item for item in inventory["records"]}
    assert migrated_ids <= set(by_id)
    assert all(by_id[item]["status"] in {"migrated_to_v2", "merged_into_v2"} for item in migrated_ids)
    assert all(by_id[item]["target_scenario_ids"] for item in migrated_ids)
    assert all(not by_id[item]["blocks_production"] for item in migrated_ids)
