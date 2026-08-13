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
    ("message", "scenario_id"),
    [
        ("что такое срок действия ставки", "bid.validity"),
        ("20 или 60 дней действует ставка", "bid.validity"),
        ("где посмотреть срок предложения", "bid.validity"),
        ("срок действия ставки истек", "bid.expired"),
        ("ставка истекла но лот в выигранных", "bid.expired"),
        ("продавец написал после окончания срока", "bid.expired"),
        ("сколько готовят документы после передачи", "documents.preparation_delay"),
        ("лот передан документов нет", "documents.preparation_delay"),
        ("не успеваю забрать автомобиль", "pickup.delay"),
        ("будет плата за стоянку", "pickup.delay"),
        ("сколько стоит хранение лота", "pickup.delay"),
        ("какой срок оплаты и вывоза лота", "lot.payment.overdue"),
    ],
)
def test_deadline_questions_route_to_v2(message, scenario_id):
    decision = match_scenario(message, "authorized")
    assert decision.confidence == "high"
    assert decision.scenario is not None
    assert decision.scenario.scenario_id == scenario_id


def _answer(message):
    return match_scenario(message, "authorized").scenario.answer.casefold()


def test_bid_validity_keeps_confirmed_range_and_card_boundary():
    answer = _answer("что такое срок действия ставки")
    assert "от 20 до 60 рабочих дней" in answer
    assert "точный срок" in answer
    assert "в карточке" in answer
    assert "платить за него не нужно" in answer
    assert "не более 90 дней" not in answer


def test_expired_bid_requires_new_confirmation_instead_of_guessing_consent():
    answer = _answer("ставка истекла но лот в выигранных")
    assert "не продлевается автоматически" in answer
    assert "отдельно предложить подтвердить" in answer
    assert "не считает молчание согласием" in answer


def test_documents_and_storage_answers_do_not_invent_exact_deadlines_or_prices():
    documents = _answer("лот передан документов нет")
    storage = _answer("сколько стоит хранение лота")
    assert "единого подтверждённого срока нет" in documents
    assert "не называет точную дату" in documents
    assert "единого опубликованного срока" in storage
    assert "не может рассчитать сумму заранее" in storage


def test_deadline_batch_is_linked_and_no_longer_blocks_migration():
    migrated_ids = {
        "kb-052-срок-действия-ставки",
        "kb-054-продавец-обратился-после-срока",
        "kb-061-срок-подготовки-документов",
        "kb-081-стоянка-и-задержка-вывоза",
        "site-doc-005-spornye-sroki-deystviya-predlozheniya",
        "site-doc-006-spornye-sroki-oplaty-i-vyvoza",
        "manual-review-2026-07-11-q-047-что-такое-срок-действия-ставки",
        "manual-review-2026-07-11-next25-q-055-горит-срок-действия-ставки-истек-висит-в-выигранных-что-значит",
    }
    inventory = json.loads(Path("knowledge/v2/legacy_inventory.json").read_text(encoding="utf-8"))
    by_id = {item["legacy_id"]: item for item in inventory["records"]}
    assert migrated_ids <= set(by_id)
    assert all(by_id[item]["status"] in {"migrated_to_v2", "merged_into_v2"} for item in migrated_ids)
    assert all(by_id[item]["target_scenario_ids"] for item in migrated_ids)
    assert all(not by_id[item]["blocks_production"] for item in migrated_ids)
