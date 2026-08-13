import json
from pathlib import Path

import pytest

from backend.app.bot.scenario_engine import clear_scenario_cache, match_scenario
from backend.app.main import process_chat_message
from backend.app.models.chat import ChatRequest


@pytest.fixture(autouse=True)
def clear_cache():
    clear_scenario_cache()
    yield
    clear_scenario_cache()


@pytest.mark.parametrize("message", [
    "что такое иной аукцион",
    "что значит надпись иной аукцион",
    "почему в лоте появилась ставка с другого аукциона",
    "иной аукцион это участник",
    "откуда взялась сумма из другого аукциона",
    "что делать если написано иной аукцион",
    "машину одновременно торгуют на другой площадке",
    "почему предлагают поставить больше другого аукциона",
    "если победил другой аукцион могу перебить",
    "лот уже продан если появился статус иной аукцион",
])
def test_other_auction_questions_route_to_dedicated_scenario(message):
    decision = match_scenario(message, "authorized")
    assert decision.confidence == "high"
    assert decision.scenario
    assert decision.scenario.scenario_id == "auction.other_platform_offer"


def test_other_auction_answer_preserves_approved_boundaries():
    scenario = match_scenario("что такое иной аукцион", "authorized").scenario
    assert scenario
    answer = scenario.answer.casefold()
    assert "предложение по этому же лоту на другой площадке" in answer
    assert "не отдельный участник migtorg" in answer
    assert "не подтверждение продажи" in answer
    assert "если в карточке доступна опция «предложить больше»" in answer
    assert "не гарантирует получение лота" in answer
    assert "не переводите доплату продавцу вне площадки" in answer
    assert scenario.escalation["when"] == []


def test_other_auction_live_response_does_not_create_ticket():
    response = process_chat_message(ChatRequest(message="лот уже продан если появился статус иной аукцион"))
    assert response.scenario_id == "auction.other_platform_offer"
    assert response.needs_ticket is False
    assert "не подтверждение продажи" in response.answer.casefold()


def test_other_auction_legacy_record_is_migrated_without_blocking_production():
    inventory = json.loads(Path("knowledge/v2/legacy_inventory.json").read_text(encoding="utf-8"))
    row = next(item for item in inventory["records"] if item["legacy_id"] == "faq-2026-07-10-faq-05-other-auction")
    assert row["status"] == "migrated_to_v2"
    assert row["target_scenario_ids"] == ["auction.other_platform_offer"]
    assert row["blocks_production"] is False
