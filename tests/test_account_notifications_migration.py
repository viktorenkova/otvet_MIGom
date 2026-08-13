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
    ("аккаунт не активирован", "account.activation_pending"),
    ("какие разделы есть в личном кабинете", "account.dashboard_sections"),
    ("что значат статусы лотов", "lot.status_guide"),
    ("не приходят уведомления по лотам", "notification.delivery_problem"),
    ("письма уходят в спам", "notification.delivery_problem"),
    ("есть официальный бот в телеграме", "support.telegram_channels"),
    ("можно работать с телефона", "account.mobile_access"),
    ("сайт плохо открывается на мобильном", "technical.site_error"),
    ("что значит статус передан", "transfer.confirmed"),
    ("лот передан когда придет письмо", "transfer.confirmed"),
])
def test_account_notification_questions_route_to_correct_v2_scenario(message, scenario_id):
    decision = match_scenario(message, "authorized")
    assert decision.confidence == "high"
    assert decision.scenario and decision.scenario.scenario_id == scenario_id


def test_activation_answer_does_not_promise_timing_or_request_secrets():
    scenario = match_scenario("аккаунт не активирован", "authorized").scenario
    answer = scenario.answer.casefold()
    assert "может проходить проверку" in answer
    assert "не обещает точный срок" in answer
    assert "не отправляйте пароль" in answer
    assert any(action["type"] == "open_ticket" for action in scenario.actions)


def test_status_guide_preserves_transfer_uncertainty():
    answer = match_scenario("что значат статусы лотов", "authorized").scenario.answer.casefold()
    assert "активный — торги идут" in answer
    assert "выигранный" in answer and "продавец ещё решает" in answer
    assert "не гарантирует передачу" in answer


def test_notification_problem_collects_safe_diagnostic_context():
    scenario = match_scenario("не приходят уведомления по лотам", "authorized").scenario
    answer = scenario.answer.casefold()
    assert "настройки уведомлений" in answer
    assert "ожидаемое событие" in answer
    assert "не отправляйте пароль" in answer
    assert any(action["type"] == "open_ticket" for action in scenario.actions)


def test_telegram_answer_does_not_publish_unconfirmed_link():
    answer = match_scenario("есть официальный бот в телеграме", "authorized").scenario.answer.casefold()
    assert "неподтверждённую ссылку" in answer
    assert "веб-чат" in answer and "info@migtorg.com" in answer
    assert "официальном сайте" in answer


def test_mobile_answer_does_not_promise_an_application():
    answer = match_scenario("есть приложение MIGTORG", "authorized").scenario.answer.casefold()
    assert "через веб-сайт" in answer
    assert "наличие отдельного приложения не подтверждено" in answer


def test_account_notifications_batch_migrated_7_legacy_records():
    targets = {
        "account.activation_pending", "account.dashboard_sections", "lot.status_guide",
        "notification.delivery_problem", "support.telegram_channels", "account.mobile_access",
    }
    prefixes = ("kb-017", "kb-094", "kb-095", "kb-096", "kb-097", "kb-098", "kb-118")
    scenarios = json.loads(Path("knowledge/v2/scenarios.json").read_text(encoding="utf-8"))["records"]
    batch = {
        legacy
        for row in scenarios if row["scenario_id"] in targets
        for legacy in row["legacy_ids"] if legacy.startswith(prefixes)
    }
    assert len(batch) == 7
    inventory = json.loads(Path("knowledge/v2/legacy_inventory.json").read_text(encoding="utf-8"))
    by_id = {row["legacy_id"]: row for row in inventory["records"]}
    assert all(not by_id[item]["blocks_production"] for item in batch)
