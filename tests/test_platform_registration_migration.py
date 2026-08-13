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
    ("что такое migtorg", "platform.about"),
    ("кто выставляет машины", "platform.about"),
    ("кто может участвовать", "platform.about"),
    ("как зарегистрироваться на сайте", "account.registration"),
    ("мне нужно ооо", "account.registration"),
    ("какие документы нужны для регистрации", "account.registration"),
    ("как проверить юридическую чистоту автомобиля", "vehicle.legal_check"),
])
def test_platform_and_registration_questions_route_to_v2(message, scenario_id):
    decision = match_scenario(message, "guest")
    assert decision.confidence == "high"
    assert decision.scenario is not None
    assert decision.scenario.scenario_id == scenario_id


def test_platform_answer_does_not_promise_transfer_or_ownership():
    answer = match_scenario("что такое migtorg", "guest").scenario.answer.casefold()
    assert "не выкупает лоты заранее" in answer
    assert "решение о передаче и продаже принимает продавец" in answer
    assert "не автоматическое заключение сделки" in answer


def test_registration_answer_keeps_low_entry_bar_and_data_boundary():
    answer = match_scenario("какие документы нужны для регистрации", "guest").scenario.answer.casefold()
    assert "создавать ооо или ип необязательно" in answer
    assert "документы для первичной регистрации не нужны" in answer
    assert "не сообщая пароль и код подтверждения" in answer


def test_legal_check_does_not_guarantee_clean_title():
    answer = match_scenario("гарантируете что машина без обременений", "guest").scenario.answer.casefold()
    assert "не гарантирует юридическую чистоту" in answer
    assert "гибдд" in answer
    assert "фнп" in answer
    assert "не является универсальной гарантией" in answer


def test_foundational_batch_migrated_23_legacy_records():
    scenarios = json.loads(Path("knowledge/v2/scenarios.json").read_text(encoding="utf-8"))["records"]
    target_ids = {"platform.about", "account.registration", "vehicle.legal_check", "tariff.demo"}
    expected = {legacy for row in scenarios if row["scenario_id"] in target_ids for legacy in row["legacy_ids"]}
    current_batch = {item for item in expected if item.startswith("kb-00") or item.startswith("kb-01") or "manual-review-2026-07-11-q-00" in item or item.endswith("q-010-если-я-физик-могу-участвовать") or item.endswith("q-011-обязательно-ли-наличие-юридического-лица") or item.endswith("q-012-мне-нужно-ооо")}
    assert len(current_batch) == 23
    inventory = json.loads(Path("knowledge/v2/legacy_inventory.json").read_text(encoding="utf-8"))
    by_id = {row["legacy_id"]: row for row in inventory["records"]}
    assert all(not by_id[item]["blocks_production"] for item in current_batch)
