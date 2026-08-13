import json
from pathlib import Path
from uuid import uuid4

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
    ("где находится офис", "support.office_visit"),
    ("как приехать к вам в офис", "support.office_visit"),
    ("мне нужен пропуск в офис", "support.office_visit"),
    ("у меня проблема с документами по лоту", "support.lot_case_review"),
    ("когда вопрос передают сотруднику", "support.manual_review_scope"),
    ("оплата на сайте не проходит", "payment.checkout_problem"),
    ("банк отклоняет платеж", "payment.checkout_problem"),
    ("платеж списался но не отображается", "payment.not_visible"),
    ("как связаться с поддержкой", "support.contact"),
])
def test_support_questions_route_to_correct_v2_scenario(message, scenario_id):
    decision = match_scenario(message, "authorized")
    assert decision.confidence == "high"
    assert decision.scenario and decision.scenario.scenario_id == scenario_id


def test_office_visit_requires_confirmation_and_protects_personal_data():
    scenario = match_scenario("мне нужен пропуск в офис", "authorized").scenario
    answer = scenario.answer.casefold()
    assert "смирновская" in answer and "25" in answer and "каб. 30" in answer
    assert "строго с 10:00 до 18:00" in answer
    assert "по предварительной записи" in answer
    assert "государственный регистрационный номер" in answer
    assert "марку транспортного средства" in answer
    assert "не является продавцом автомобиля" in answer
    assert any(action["type"] == "open_ticket" for action in scenario.actions)


def test_generic_visit_question_clarifies_office_inspection_or_pickup():
    response = process_chat_message(
        ChatRequest(message="как к вам попасть", session_id=f"office-purpose-clarification-{uuid4()}")
    )

    assert response.resolution == "clarified"
    assert "уточните цель визита" in response.answer.casefold()
    assert response.clarifying_options == [
        "Визит в офис и оформление пропуска",
        "Как организовать осмотр автомобиля",
        "Доступ на стоянку и кто выдаёт лот",
        "Другая тема",
    ]


def test_platform_contract_visit_routes_to_office_not_lot_contract():
    decision = match_scenario("хочу приехать подписать договор на доступ к площадке", "guest")

    assert decision.confidence == "high"
    assert decision.scenario and decision.scenario.scenario_id == "support.office_visit"


def test_vehicle_pickup_question_never_returns_office_address():
    decision = match_scenario("где получать автомобиль", "guest")

    assert decision.scenario and decision.scenario.scenario_id == "pickup.access_issuer"
    assert "смирновская" not in decision.scenario.answer.casefold()


def test_lot_case_review_does_not_promise_result_or_collect_secrets():
    scenario = match_scenario("у меня проблема с документами по лоту", "authorized").scenario
    answer = scenario.answer.casefold()
    assert "номером лота" in answer
    assert "не отправляйте пароль" in answer
    assert "не подтверждает результат до проверки" in answer
    assert any(action["type"] == "open_ticket" for action in scenario.actions)


def test_manual_review_scope_does_not_perform_staff_actions():
    answer = match_scenario("когда вопрос передают сотруднику", "authorized").scenario.answer.casefold()
    assert "не подтверждает платёж" in answer
    assert "не снимает штраф" in answer
    assert "не одобряет возврат" in answer
    assert "минимальным набором подтверждений" in answer


def test_checkout_problem_distinguishes_failed_from_debited_payment():
    scenario = match_scenario("оплата на сайте не проходит", "authorized").scenario
    answer = scenario.answer.casefold()
    assert "убедитесь, что списания не было" in answer
    assert "если деньги уже списаны, не повторяйте платёж" in answer
    assert "cvc/cvv" in answer
    assert any(action["type"] == "open_ticket" for action in scenario.actions)


def test_support_manual_review_batch_migrated_5_legacy_records():
    targets = {
        "support.office_visit", "support.lot_case_review",
        "support.manual_review_scope", "payment.checkout_problem",
    }
    prefixes = ("kb-111", "kb-112", "kb-113", "kb-114", "kb-116")
    scenarios = json.loads(Path("knowledge/v2/scenarios.json").read_text(encoding="utf-8"))["records"]
    batch = {
        legacy
        for row in scenarios if row["scenario_id"] in targets
        for legacy in row["legacy_ids"] if legacy.startswith(prefixes)
    }
    assert len(batch) == 5
    inventory = json.loads(Path("knowledge/v2/legacy_inventory.json").read_text(encoding="utf-8"))
    by_id = {row["legacy_id"]: row for row in inventory["records"]}
    assert all(not by_id[item]["blocks_production"] for item in batch)
