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
    ("что такое раздел имущество", "property.overview"),
    ("как оценивать имущественный лот", "property.evaluate"),
    ("можно разделить партию", "property.bulk_goods"),
    ("как оценить готовый бизнес", "property.business"),
    ("какие документы будут по имуществу", "property.inspection_documents"),
    ("не открывается раздел имущество", "property.section_error"),
    ("что находится в разделе КАСКО", "vehicle.insurance_catalog"),
    ("что находится в разделе ОСАГО", "vehicle.insurance_catalog"),
    ("что такое лот", "lot.definition"),
])
def test_property_catalog_questions_route_to_v2(message, scenario_id):
    decision = match_scenario(message, "authorized")
    assert decision.confidence == "high"
    assert decision.scenario and decision.scenario.scenario_id == scenario_id


def test_bulk_goods_answer_does_not_promise_splitting():
    scenario = match_scenario("можно разделить партию", "authorized").scenario
    answer = scenario.answer.casefold()
    assert "может продаваться только целиком" in answer
    assert "прямо указано в карточке" in answer
    assert "не может согласовать частичный выкуп" in answer


def test_business_answer_does_not_treat_label_as_proof():
    answer = match_scenario("как оценить готовый бизнес", "authorized").scenario.answer.casefold()
    assert "подтверждение выручки" in answer
    assert "долги" in answer
    assert "название само по себе не подтверждает состав бизнеса" in answer


def test_property_documents_require_lot_context_and_safe_escalation():
    scenario = match_scenario("какие документы будут по имуществу", "authorized").scenario
    answer = scenario.answer.casefold()
    assert "зависят от условий конкретного лота и продавца" in answer
    assert "не предполагайте его наличие" in answer
    assert "номером лота" in answer
    assert any(action["type"] == "open_ticket" for action in scenario.actions)


def test_property_section_error_uses_safe_diagnostics():
    scenario = match_scenario("не открывается раздел имущество", "authorized").scenario
    answer = scenario.answer.casefold()
    assert "другой браузер или приватное окно" in answer
    assert "не отправляйте пароль или платёжные данные" in answer
    assert "скриншот" in answer
    assert any(action["type"] == "open_ticket" for action in scenario.actions)


def test_property_catalog_batch_migrated_11_legacy_records():
    targets = {
        "property.overview", "property.evaluate", "property.bulk_goods",
        "property.business", "property.inspection_documents", "property.section_error",
        "vehicle.insurance_catalog", "lot.definition",
    }
    prefixes = (
        "kb-088", "kb-089", "kb-090", "kb-091", "kb-092", "kb-093",
        "site-doc-020", "site-doc-021", "site-doc-022", "site-doc-023", "site-doc-026",
    )
    scenarios = json.loads(Path("knowledge/v2/scenarios.json").read_text(encoding="utf-8"))["records"]
    batch = {
        legacy
        for row in scenarios if row["scenario_id"] in targets
        for legacy in row["legacy_ids"] if legacy.startswith(prefixes)
    }
    assert len(batch) == 11
    inventory = json.loads(Path("knowledge/v2/legacy_inventory.json").read_text(encoding="utf-8"))
    by_id = {row["legacy_id"]: row for row in inventory["records"]}
    assert all(not by_id[item]["blocks_production"] for item in batch)
