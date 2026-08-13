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
    ("где искать лоты", "lot.catalog_search"), ("мне нужна конкретная машина", "lot.catalog_search"),
    ("как читать карточку лота", "lot.card_information"), ("лот исчез или изменился", "lot.card_information"),
    ("почему нет птс в карточке", "lot.card_information"), ("как запросить vin по лоту", "vehicle.vin_request"),
    ("сколько стоит проверка vin", "vehicle.vin_request"), ("как посмотреть машину", "inspection.arrange"),
    ("где находится машина", "lot.location"),
])
def test_catalog_card_questions_route_to_v2(message, scenario_id):
    decision = match_scenario(message, "authorized")
    assert decision.confidence == "high"
    assert decision.scenario and decision.scenario.scenario_id == scenario_id


def test_missing_card_data_is_not_invented_or_treated_as_transfer_guarantee():
    answer = match_scenario("почему мало информации о машине", "authorized").scenario.answer.casefold()
    assert "отсутствующие характеристики бот не придумывает" in answer
    assert "безопаснее не делать ставку" in answer
    assert "не гарантирует передачу" in answer


def test_vin_answer_keeps_price_freshness_and_legal_guardrail():
    answer = match_scenario("сколько стоит проверка vin", "authorized").scenario.answer.casefold()
    assert "500 рублей" in answer
    assert "проверьте актуальную цену" in answer
    assert "не являются гарантией юридической чистоты" in answer


def test_catalog_batch_migrated_twenty_legacy_records():
    targets = {"lot.catalog_search", "lot.card_information", "vehicle.vin_request", "inspection.arrange", "lot.location"}
    scenarios = json.loads(Path("knowledge/v2/scenarios.json").read_text(encoding="utf-8"))["records"]
    batch = {legacy for row in scenarios if row["scenario_id"] in targets for legacy in row["legacy_ids"] if legacy.startswith(("kb-034", "kb-035", "kb-036", "kb-037", "kb-038", "kb-039", "kb-040", "faq-2026-07-10-faq-02", "faq-2026-07-10-faq-03", "manual-review-2026-07-11-next25-q-05", "manual-review-2026-07-11-next25-q-06"))}
    assert len(batch) == 20
    inventory = json.loads(Path("knowledge/v2/legacy_inventory.json").read_text(encoding="utf-8"))
    by_id = {row["legacy_id"]: row for row in inventory["records"]}
    assert all(not by_id[item]["blocks_production"] for item in batch)
