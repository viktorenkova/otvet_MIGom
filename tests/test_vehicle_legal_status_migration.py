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
    ("на машине есть ограничения", "vehicle.encumbrance_detected"),
    ("на машине есть залоги", "vehicle.encumbrance_detected"),
    ("что значит тс кредитное", "vehicle.credit_lease_pledge"),
    ("что значит тс в лизинге", "vehicle.credit_lease_pledge"),
    ("что значит тс в залоге", "vehicle.credit_lease_pledge"),
    ("нужно ли мне гасить кредит", "vehicle.credit_lease_pledge"),
    ("машина попадет в Автотеку после торгов", "vehicle.autoteka_visibility"),
    ("страховая выставила мой автомобиль", "vehicle.owner_listing_dispute"),
    ("какие правила торгов по ОСАГО", "insurance.osago_rules"),
    ("сколько дней на оплату годных остатков", "insurance.osago_rules"),
])
def test_vehicle_legal_status_questions_route_to_v2(message, scenario_id):
    decision = match_scenario(message, "authorized")
    assert decision.confidence == "high"
    assert decision.scenario and decision.scenario.scenario_id == scenario_id


def test_detected_encumbrance_stops_payment_and_requires_evidence():
    scenario = match_scenario("на машине есть ограничения", "authorized").scenario
    answer = scenario.answer.casefold()
    assert "не оплачивайте" in answer and "не забирайте" in answer
    assert "не должен погашать чужую задолженность" in answer
    assert "не гарантирует автоматическое признание отказа" in answer
    assert any(action["type"] == "open_ticket" for action in scenario.actions)


def test_credit_or_lease_debt_is_not_shifted_to_buyer():
    answer = match_scenario("нужно ли мне гасить кредит", "authorized").scenario.answer.casefold()
    assert "покупатель не должен" in answer
    assert "предыдущего собственника" in answer
    assert "не оплачивайте чужую задолженность" in answer


def test_autoteka_answer_does_not_promise_absence_of_other_records():
    answer = match_scenario("машина попадет в Автотеку после торгов", "authorized").scenario.answer.casefold()
    assert "не означает автоматическую запись" in answer
    assert "определяет сама автотека" in answer
    assert "других записей" in answer


def test_osago_answer_preserves_all_confirmed_deadlines():
    answer = match_scenario("какие правила торгов по ОСАГО", "authorized").scenario.answer.casefold()
    assert "не менее 3 календарных дней" in answer
    assert "от 20 до 40 календарных дней" in answer
    assert "не более 14 календарных дней" in answer


def test_vehicle_legal_status_batch_migrated_10_legacy_records():
    targets = {
        "vehicle.encumbrance_detected", "vehicle.credit_lease_pledge",
        "vehicle.autoteka_visibility", "vehicle.owner_listing_dispute",
        "insurance.osago_rules",
    }
    prefixes = (
        "faq-2026-07-10-faq-04", "site-doc-008", "site-doc-012", "site-doc-030",
        "manual-review-2026-07-11-remaining54-q-102",
        "manual-review-2026-07-11-remaining54-q-103",
        "manual-review-2026-07-11-remaining54-q-104",
        "manual-review-2026-07-11-remaining54-q-105",
        "manual-review-2026-07-11-remaining54-q-106",
        "manual-review-2026-07-11-remaining54-q-107",
    )
    scenarios = json.loads(Path("knowledge/v2/scenarios.json").read_text(encoding="utf-8"))["records"]
    batch = {
        legacy
        for row in scenarios if row["scenario_id"] in targets
        for legacy in row["legacy_ids"] if legacy.startswith(prefixes)
    }
    assert len(batch) == 10
    inventory = json.loads(Path("knowledge/v2/legacy_inventory.json").read_text(encoding="utf-8"))
    by_id = {row["legacy_id"]: row for row in inventory["records"]}
    assert all(not by_id[item]["blocks_production"] for item in batch)
