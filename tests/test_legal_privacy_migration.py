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
    ("что такое аккредитация", "registration.accreditation"),
    ("почему отказали в аккредитации", "registration.accreditation"),
    ("можно передать логин другому человеку", "account.credential_responsibility"),
    ("украли доступ к аккаунту", "account.credential_responsibility"),
    ("зачем вы обрабатываете персональные данные", "privacy.processing_purposes"),
    ("какие данные вы собираете", "privacy.data_categories"),
    ("обрабатываете биометрию", "privacy.data_categories"),
    ("как отозвать согласие", "privacy.rights_withdrawal"),
    ("удалить мои персональные данные", "privacy.rights_withdrawal"),
    ("как обжаловать действия организатора", "complaint.formal_dispute"),
])
def test_legal_and_privacy_questions_route_correctly(message, scenario_id):
    decision = match_scenario(message, "guest")
    assert decision.confidence == "high"
    assert decision.scenario and decision.scenario.scenario_id == scenario_id


def test_accreditation_does_not_invent_universal_document_list_or_decision():
    scenario = match_scenario("почему отказали в аккредитации", "guest").scenario
    answer = scenario.answer.casefold()
    assert "физлица, ип и организации" in answer
    assert "точный комплект зависит" in answer
    assert "только после проверки" in answer
    assert any(action["type"] == "open_ticket" for action in scenario.actions)


def test_credential_security_states_deadline_and_excludes_secrets():
    scenario = match_scenario("украли доступ к аккаунту", "guest").scenario
    answer = scenario.answer.casefold()
    assert "в течение 1 рабочего дня" in answer
    assert "проверит владельца аккаунта" in answer
    assert "не отправляйте пароль" in answer
    assert any(action["type"] == "open_ticket" for action in scenario.actions)


def test_privacy_purposes_stay_general_and_do_not_expose_user_data():
    answer = match_scenario("зачем вы обрабатываете персональные данные", "guest").scenario.answer.casefold()
    assert "идентификации, договоров, обратной связи" in answer
    assert "статистику и аналитику при обезличивании" in answer
    assert "не о раскрытии данных конкретного пользователя" in answer


def test_privacy_categories_include_technical_data_and_policy_exclusions():
    answer = match_scenario("какие данные вы собираете", "guest").scenario.answer.casefold()
    assert "фио, адрес, телефон, email и cookie" in answer
    assert "история запросов и просмотров" in answer
    assert "не обрабатывает биометрические" in answer
    assert "не осуществляет трансграничную передачу" in answer


def test_consent_withdrawal_preserves_lawful_processing_caveat():
    scenario = match_scenario("как отозвать согласие", "guest").scenario
    answer = scenario.answer.casefold()
    assert "отзыв согласия на обработку персональных данных" in answer
    assert "в течение 5 рабочих дней" in answer
    assert "если нет законных оснований" in answer
    assert "не означает автоматическое подтверждение удаления" in answer
    assert any(action["type"] == "open_ticket" for action in scenario.actions)


def test_formal_complaint_explains_channels_without_predicting_outcome():
    scenario = match_scenario("как обжаловать действия организатора", "guest").scenario
    answer = scenario.answer.casefold()
    assert "направить организатору" in answer
    assert "30 календарных дней" in answer
    assert "только после рассмотрения" in answer
    assert any(action["type"] == "open_ticket" for action in scenario.actions)


@pytest.mark.parametrize(("message", "marker"), [
    ("как отозвать согласие", "5 рабочих дней"),
    ("как обжаловать действия организатора", "30 календарных дней"),
    ("украли доступ к аккаунту", "не отправляйте пароль"),
])
def test_live_chat_preserves_specific_legal_guidance(message, marker):
    response = process_chat_message(
        ChatRequest(message=message, session_id=f"legal-privacy-{abs(hash(message))}")
    )
    assert marker in response.answer.casefold()


def test_legal_privacy_batch_migrated_6_legacy_records():
    prefixes = (
        "site-doc-009",
        "site-doc-010",
        "site-doc-015",
        "site-doc-016",
        "site-doc-017",
        "site-doc-019",
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
