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


@pytest.mark.parametrize(("message", "scenario_id"), [
    ("не могу до вас дозвониться", "support.callback"),
    ("можем договориться", "compliance.private_arrangement"),
    ("можем настроить индивидуальное сотрудничество", "partnership.official_proposal"),
    ("реализация через меня обсуждаема", "partnership.official_proposal"),
    ("готов доплатить чтобы забрать лот", "compliance.off_platform_payment"),
    ("мне угрожает участник площадки", "safety.reported_threat"),
])
def test_contact_compliance_and_safety_questions_route_correctly(message, scenario_id):
    decision = match_scenario(message, "guest")
    assert decision.confidence == "high"
    assert decision.scenario and decision.scenario.scenario_id == scenario_id


def test_failed_call_redirects_to_written_correspondence():
    scenario = match_scenario("не могу до вас дозвониться", "guest").scenario
    answer = scenario.answer.casefold()
    assert "по переписке" in answer
    assert "номер лота" in answer
    assert "не можем обещать обратный звонок" in answer
    assert {action["type"] for action in scenario.actions} == {"open_ticket"}


def test_private_arrangement_refuses_bypass_and_offers_official_path():
    scenario = match_scenario("можем договориться", "guest").scenario
    answer = scenario.answer.casefold()
    assert "в обход торгов" in answer
    assert "не помогу с частной договорённостью" in answer
    assert "официальный вопрос" in answer
    assert scenario.actions == ()


def test_partnership_request_is_separated_from_current_auction_influence():
    scenario = match_scenario("можем настроить индивидуальное сотрудничество", "guest").scenario
    answer = scenario.answer.casefold()
    assert "коммерческое или партнёрское предложение" in answer
    assert "индивидуальные преимущества" in answer
    assert "можно подтвердить только после рассмотрения" in answer
    assert any(action["type"] == "open_ticket" for action in scenario.actions)


def test_off_platform_payment_is_refused_with_official_alternatives():
    answer = match_scenario("готов доплатить чтобы забрать лот", "guest").scenario.answer.casefold()
    assert "вне площадки нельзя" in answer
    assert "переторги" in answer
    assert "предложить больше" in answer
    assert "не помогу организовать" in answer


def test_reported_threat_prioritizes_immediate_safety_and_evidence():
    scenario = match_scenario("мне угрожает участник площадки", "guest").scenario
    answer = scenario.answer.casefold()
    assert "не вступайте в конфликт" in answer
    assert "экстренные службы" in answer
    assert "не публикуйте персональные данные" in answer
    assert "без обещания результата" in answer
    assert any(action.get("priority") == "urgent" for action in scenario.actions)


@pytest.mark.parametrize(("message", "marker"), [
    ("можем договориться", "договорённости в обход торгов"),
    ("готов доплатить чтобы забрать лот", "вне площадки нельзя"),
    ("мне угрожает участник площадки", "экстренные службы"),
])
def test_live_chat_keeps_safe_specific_guidance(message, marker):
    response = process_chat_message(
        ChatRequest(message=message, session_id=f"contact-compliance-{abs(hash(message))}")
    )
    assert marker in response.answer.casefold()


def test_contact_compliance_safety_batch_migrated_6_legacy_records():
    prefixes = (
        "manual-review-2026-07-11-remaining54-q-124",
        "manual-review-2026-07-11-remaining54-q-126",
        "manual-review-2026-07-11-remaining54-q-127",
        "manual-review-2026-07-11-remaining54-q-128",
        "manual-review-2026-07-11-remaining54-q-129",
        "kb-safety-reported-threat",
    )
    scenarios = json.loads(Path("knowledge/v2/scenarios.json").read_text(encoding="utf-8"))["records"]
    batch = {
        legacy
        for row in scenarios
        for legacy in row["legacy_ids"]
        if legacy.startswith(prefixes)
    }
    assert len(batch) == 6
    inventory = json.loads(Path("knowledge/v2/legacy_inventory.json").read_text(encoding="utf-8"))
    by_id = {row["legacy_id"]: row for row in inventory["records"]}
    assert all(not by_id[item]["blocks_production"] for item in batch)
