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
        ("какое имущество можно продавать на площадке", "seller.categories"),
        ("можно выставить транспорт", "seller.categories"),
        ("как продать автомобиль через migtorg", "seller.categories"),
        ("можно продать партию товаров оптом", "seller.categories"),
        ("когда продавцу дадут логин и пароль", "seller.access_security"),
        ("можно передать пароль продавца менеджеру", "seller.access_security"),
        ("информация продавца это оферта", "seller.information_status"),
        ("сколько действует предложение покупателя по оферте продавца", "seller.information_status"),
        ("есть обучение для продавца", "seller.training"),
        ("обучите сотрудников продавца", "seller.training"),
        ("на сколько заключается договор продавца", "seller.contract_term"),
        ("как продавцу расторгнуть договор", "seller.contract_term"),
    ],
)
def test_seller_questions_route_to_v2(message, scenario_id):
    decision = match_scenario(message, "authorized")
    assert decision.confidence == "high"
    assert decision.scenario is not None
    assert decision.scenario.scenario_id == scenario_id


def _answer(message):
    return match_scenario(message, "authorized").scenario.answer.casefold()


def test_categories_allow_transport_but_require_object_review():
    answer = _answer("можно выставить транспорт")
    assert "транспортные средства" in answer
    assert "готс" in answer
    assert "недвижимость" in answer
    assert "не принимается автоматически" in answer
    assert "право распоряжения" in answer


def test_seller_credentials_are_never_requested_in_ticket():
    answer = _answer("потерял пароль продавца")
    assert "в течение 1 рабочего дня" in answer
    assert "не отправляйте пароль" in answer
    assert "коды подтверждения" in answer


def test_seller_offer_term_is_not_mixed_with_bid_validity():
    answer = _answer("сколько действует предложение покупателя по оферте продавца")
    assert "25 рабочих дней" in answer
    assert "не подменяет срок победной ставки" in answer
    assert "автоматически подтверждать обязанность" in answer
    assert "от 20 до 60" not in answer


def test_seller_contract_answer_requires_written_notice():
    answer = _answer("как продавцу расторгнуть договор")
    assert "действует 1 год" in answer
    assert "письменное уведомление за 30 дней" in answer
    assert "сообщение в чате само по себе договор не расторгает" in answer


def test_seller_batch_is_linked_and_no_longer_blocks_migration():
    migrated_ids = {
        "seller-offer-002-kakoe-imuschestvo-mozhno-razmeschat",
        "seller-offer-003-prodazha-avto-transporta-i-gots",
        "seller-offer-004-prodazha-partii-tovarov-optom",
        "seller-offer-005-dostup-login-parol-prodavca",
        "seller-offer-006-informatsiya-prodavca-ne-oferta",
        "seller-offer-009-obuchenie-sotrudnikov-prodavca",
        "seller-offer-010-srok-i-rastorzhenie-dogovora-prodavca",
        "site-doc-013-prodavets-i-informatsiya-ne-oferta",
    }
    inventory = json.loads(Path("knowledge/v2/legacy_inventory.json").read_text(encoding="utf-8"))
    by_id = {item["legacy_id"]: item for item in inventory["records"]}
    assert migrated_ids <= set(by_id)
    assert all(by_id[item]["status"] in {"migrated_to_v2", "merged_into_v2"} for item in migrated_ids)
    assert all(by_id[item]["target_scenario_ids"] for item in migrated_ids)
    assert all(not by_id[item]["blocks_production"] for item in migrated_ids)
