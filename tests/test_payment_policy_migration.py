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
    "message",
    [
        "минимальный депозит 100 рублей",
        "минимальный депозит 1 usd",
        "в правилах написано возврат только на карту",
        "можно вернуть случайную оплату",
        "какие способы пополнения указаны в правилах",
        "почему правила оплаты отличаются от кабинета",
        "visa mastercard мир или сбп чем пополнять",
        "общие правила возврата относятся к тарифу",
    ],
)
def test_document_payment_policy_routes_to_safe_v2_scenario(message):
    decision = match_scenario(message, "authorized")
    assert decision.confidence == "high"
    assert decision.scenario is not None
    assert decision.scenario.scenario_id == "payment.document_policy"


def test_document_policy_distinguishes_published_rules_from_current_ui():
    answer = match_scenario("почему правила оплаты отличаются от кабинета", "authorized").scenario.answer.casefold()
    assert "1 usd/100 rub" in answer
    assert "комиссией 2,5%" in answer
    assert "сбп с комиссией 1%" in answer
    assert "минимальную доступную сумму показывает сама форма" in answer


def test_document_policy_does_not_promise_refund_or_force_card_destination():
    answer = match_scenario("можно вернуть случайную оплату", "authorized").scenario.answer.casefold()
    assert "не обещает его возврат" in answer
    assert "исходный инструмент" in answer
    assert "банковский счёт" in answer
    assert "баланс migtorg" in answer
    assert "деньги точно вернут" not in answer
    assert "возврат одобрен" not in answer


def test_expert_review_queue_is_empty_after_payment_policy_migration():
    inventory = json.loads(Path("knowledge/v2/legacy_inventory.json").read_text(encoding="utf-8"))
    by_id = {item["legacy_id"]: item for item in inventory["records"]}
    item = by_id["site-doc-018-platezhnaya-politika-v-dokumentah"]
    assert item["status"] in {"migrated_to_v2", "merged_into_v2"}
    assert item["target_scenario_ids"] == ["payment.document_policy"]
    assert not item["blocks_production"]
    assert not [row for row in inventory["records"] if row["status"] == "expert_review_required"]
