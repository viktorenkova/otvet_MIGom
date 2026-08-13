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
    ("как узнать победил ли я", "auction.result"),
    ("можно поставить ради просмотра результата", "auction.result"),
    ("лот не передают", "transfer.not_confirmed"),
    ("продавец не связался после победы", "transfer.not_confirmed"),
    ("что значит лот передан", "transfer.confirmed"),
    ("лот передали когда придет письмо", "transfer.confirmed"),
    ("как получить выигранный лот", "win.next_steps"),
])
def test_transfer_result_questions_route_to_v2(message, scenario_id):
    decision = match_scenario(message, "authorized")
    assert decision.confidence == "high"
    assert decision.scenario is not None
    assert decision.scenario.scenario_id == scenario_id


def test_result_and_nontransfer_answers_do_not_promise_a_deal():
    result = match_scenario("можно поставить ради просмотра результата", "authorized").scenario.answer.casefold()
    pending = match_scenario("лот не передают", "authorized").scenario.answer.casefold()
    assert "может стать победной" in result
    assert "не означает автоматическую передачу" in result
    assert "окончательное решение" in pending
    assert "не оплачивайте автомобиль" in pending


def test_confirmed_transfer_uses_only_official_payment_documents():
    answer = match_scenario("что значит лот передан", "authorized").scenario.answer.casefold()
    assert "продавец подтвердил готовность" in answer
    assert "info@migtorg.com" in answer
    assert "официальные документы" in answer


def test_transfer_batch_migrated_ten_legacy_records():
    ids = {"kb-007-почему-победа-не-гарантирует-покупку", "kb-048-ставка-ради-просмотра-результата", "kb-053-если-продавец-не-связался", "kb-055-почему-нужно-формировать-воронку", "kb-056-лот-не-передали-это-нормально", "kb-057-что-значит-лот-передан", "kb-058-уведомление-о-передаче", "site-doc-004-pobeditel-i-reshenie-sobstvennika", "manual-review-2026-07-11-q-044-как-узнать-победил-ли-я", "manual-review-2026-07-11-q-045-как-получить-выигранный-лот"}
    inventory = json.loads(Path("knowledge/v2/legacy_inventory.json").read_text(encoding="utf-8"))
    by_id = {row["legacy_id"]: row for row in inventory["records"]}
    assert all(not by_id[item]["blocks_production"] for item in ids)
    assert all(by_id[item]["target_scenario_ids"] for item in ids)
