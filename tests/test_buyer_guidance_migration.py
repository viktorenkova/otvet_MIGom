import json
from pathlib import Path

import pytest

from backend.app.bot.scenario_engine import clear_scenario_cache, match_scenario


@pytest.fixture(autouse=True)
def clear_cache():
    clear_scenario_cache()
    yield
    clear_scenario_cache()


@pytest.mark.parametrize(("message", "scenario_id"), [
    ("хочу купить одну машину для себя", "buyer.personal_purchase"),
    ("хочу перепродавать автомобили", "buyer.resale_strategy"),
    ("что подготовить до первой ставки", "buyer.first_bid_checklist"),
    ("как выбирать лоты новичку", "buyer.beginner_lot_selection"),
    ("как зарабатывают на битых авто", "vehicle.business_use_cases"),
    ("как рассчитать ставку", "bid.maximum_budget"),
    ("зачем смотреть завершенные торги", "auction.completed_analytics"),
    ("какой тариф выбрать для одной машины", "tariff.choose"),
    ("как узнать минимальную ставку", "bid.price_terms"),
    ("как узнать победил ли я", "auction.result"),
])
def test_buyer_guidance_questions_route_to_correct_v2_scenario(message, scenario_id):
    decision = match_scenario(message, "authorized")
    assert decision.confidence == "high"
    assert decision.scenario and decision.scenario.scenario_id == scenario_id


def test_personal_purchase_does_not_promise_vehicle_transfer():
    answer = match_scenario("хочу купить одну машину для себя", "authorized").scenario.answer.casefold()
    assert "гарантировать нельзя" in answer
    assert "низкая ставка не означает" in answer
    assert "ремонт, доставку" in answer


def test_resale_strategy_does_not_promise_profit():
    answer = match_scenario("хочу перепродавать автомобили", "authorized").scenario.answer.casefold()
    assert "воронки ставок" in answer
    assert "не гарантирует сделку" in answer
    assert "без предположения о гарантированной прибыли" in answer


def test_first_bid_checklist_includes_full_costs_and_resources():
    answer = match_scenario("что подготовить до первой ставки", "authorized").scenario.answer.casefold()
    assert "не только цену лота" in answer
    assert "осмотрщика, перевозчика, сто" in answer
    assert "резерв на непредвиденные расходы" in answer


def test_beginner_selection_refuses_guaranteed_liquidity():
    answer = match_scenario("как выбирать лоты новичку", "authorized").scenario.answer.casefold()
    assert "прозрачных повреждений" in answer
    assert "плохо описанных лотов" in answer
    assert "не гарантирует выгодную покупку" in answer


def test_vehicle_use_cases_are_not_profit_promises():
    answer = match_scenario("как зарабатывают на битых авто", "authorized").scenario.answer.casefold()
    assert "разбирать на запчасти" in answer
    assert "один и тот же лот" in answer
    assert "не гарантирует доходность" in answer


def test_maximum_bid_formula_is_conservative_and_not_personalized():
    answer = match_scenario("как рассчитать ставку", "authorized").scenario.answer.casefold()
    assert "ожидаемая цена выхода минус" in answer
    assert "резерв на риски" in answer
    assert "не гарантирует прибыль" in answer
    assert "не повышайте её из-за азарта" in answer


def test_completed_auction_analysis_is_not_market_price_guarantee():
    answer = match_scenario("зачем смотреть завершенные торги", "authorized").scenario.answer.casefold()
    assert "собственные ставки" in answer
    assert "один результат не подтверждает текущую рыночную цену" in answer
    assert "сопоставимых завершённых лотов" in answer


def test_buyer_guidance_batch_migrated_7_legacy_records():
    targets = {
        "buyer.personal_purchase", "buyer.resale_strategy", "buyer.first_bid_checklist",
        "buyer.beginner_lot_selection", "vehicle.business_use_cases",
        "bid.maximum_budget", "auction.completed_analytics",
    }
    prefixes = ("kb-100", "kb-101", "kb-102", "kb-103", "kb-104", "kb-105", "kb-106")
    scenarios = json.loads(Path("knowledge/v2/scenarios.json").read_text(encoding="utf-8"))["records"]
    batch = {
        legacy
        for row in scenarios if row["scenario_id"] in targets
        for legacy in row["legacy_ids"] if legacy.startswith(prefixes)
    }
    assert len(batch) == 7
    inventory = json.loads(Path("knowledge/v2/legacy_inventory.json").read_text(encoding="utf-8"))
    by_id = {row["legacy_id"]: row for row in inventory["records"]}
    assert all(not by_id[item]["blocks_production"] for item in batch)
