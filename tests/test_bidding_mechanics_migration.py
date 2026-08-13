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
    ("что такое открытые торги", "auction.formats"), ("что такое закрытые торги", "auction.formats"),
    ("что такое шаг ставки", "bid.price_terms"), ("как узнать минимальную ставку", "bid.price_terms"),
    ("как работает автоставка", "bid.autobid_extension"), ("почему торги продлились", "bid.autobid_extension"),
    ("могу отменить ставку", "bid.modify_cancel"), ("можно уменьшить ставку", "bid.modify_cancel"),
    ("торги завершены можно отменить ставку", "bid.modify_cancel"),
    ("что такое узнать позицию ставки", "bid.position_service"),
])
def test_bidding_questions_route_to_v2(message, scenario_id):
    decision = match_scenario(message, "authorized")
    assert decision.confidence == "high"
    assert decision.scenario and decision.scenario.scenario_id == scenario_id


def test_bid_cancellation_depends_on_type_and_status():
    answer = match_scenario("могу отменить ставку", "authorized").scenario.answer.casefold()
    assert "закрытых торгах" in answer and "до 1 рубля" in answer
    assert "активных открытых торгах" in answer and "создайте обращение" in answer
    assert "после окончания любых торгов ставка фиксируется" in answer


def test_position_service_does_not_reveal_competitors_or_guarantee_win():
    answer = match_scenario("что такое узнать позицию ставки", "authorized").scenario.answer.casefold()
    assert "1 200 рублей" in answer
    assert "не раскрывает чужие суммы" in answer
    assert "не гарантирует победу" in answer


def test_bidding_batch_migrated_31_legacy_records():
    targets = {"auction.formats", "bid.price_terms", "bid.autobid_extension", "bid.modify_cancel", "bid.position_service", "lot.insufficient_information"}
    scenarios = json.loads(Path("knowledge/v2/scenarios.json").read_text(encoding="utf-8"))["records"]
    prefixes = ("kb-041", "kb-042", "kb-043", "kb-044", "kb-046", "kb-047", "kb-049", "kb-050", "site-doc-001", "site-doc-002", "site-doc-003", "site-doc-024", "site-doc-025", "site-doc-027", "site-doc-031", "site-doc-032", "faq-2026-07-10-faq-01", "faq-2026-07-10-faq-06", "manual-review-2026-07-11-q-034", "manual-review-2026-07-11-q-035", "manual-review-2026-07-11-q-036", "manual-review-2026-07-11-q-038", "manual-review-2026-07-11-q-039", "manual-review-2026-07-11-q-040", "manual-review-2026-07-11-q-041", "manual-review-2026-07-11-q-042", "manual-review-2026-07-11-q-043", "manual-review-2026-07-11-q-048", "manual-review-2026-07-11-q-049", "manual-review-2026-07-11-q-050", "manual-review-2026-07-11-next25-q-056")
    batch = {legacy for row in scenarios if row["scenario_id"] in targets for legacy in row["legacy_ids"] if legacy.startswith(prefixes)}
    assert len(batch) == 31
    inventory = json.loads(Path("knowledge/v2/legacy_inventory.json").read_text(encoding="utf-8"))
    by_id = {row["legacy_id"]: row for row in inventory["records"]}
    assert all(not by_id[item]["blocks_production"] for item in batch)
