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


def _scenario_map():
    payload = json.loads(Path("knowledge/v2/scenarios.json").read_text(encoding="utf-8"))
    return {item["scenario_id"]: item for item in payload["records"]}


@pytest.mark.parametrize(
    ("message", "scenario_id"),
    [
        ("продавец задержал осмотр что будет со сроком", "refusal.deadline_and_submission"),
        ("можно отказаться если двигатель снят", "refusal.evidence"),
        ("можно отказаться из-за дубликата ПТС", "refusal.invalid_reasons"),
        ("как оспорить штраф", "penalty.explain_or_dispute"),
        ("оплатил штраф но аккаунт заблокирован", "penalty.explain_or_dispute"),
        ("достаточно телефона и почты для демо", "account.identification_for_contract"),
        ("не могу поставить машину на учет", "vehicle.registration_after_purchase"),
        ("как считают место в рейтинге", "commission.discount"),
    ],
)
def test_final_expert_answers_route_to_published_scenarios(message, scenario_id):
    decision = match_scenario(message, "authorized")
    assert decision.confidence == "high"
    assert decision.scenario is not None
    assert decision.scenario.scenario_id == scenario_id


def test_hidden_substantial_damage_is_not_misclassified_as_invalid_reason():
    scenario = _scenario_map()["refusal.invalid_reasons"]
    joined = " ".join([*scenario["facts"], scenario["detailed_answer"]])
    assert "может быть основанием для мотивированного отказа" in joined
    assert "отсутствие сведений в карточке нельзя автоматически считать" in joined.lower()
    assert "Дубликат ПТС" in joined


def test_seller_delay_pauses_refusal_deadline():
    answer = match_scenario("продавец задержал документы что будет со сроком", "authorized").scenario.answer
    assert "приостанавливается" in answer
    assert "с продавцом" in answer


def test_penalty_answer_uses_invoice_and_manual_confirmation_without_formula():
    answer = match_scenario("как оспорить штраф", "authorized").scenario.answer
    assert "указаны в счёте" in answer
    assert "не рассчитывает" in answer
    assert "info@migtorg.com" in answer
    assert "платёжное поручение" in answer


def test_loyalty_answer_explains_ranking_and_no_retroactive_recalculation():
    answer = match_scenario("как считают место в рейтинге", "authorized").scenario.answer
    assert "оплачена продавцу" in answer
    assert "общему объёму" in answer
    assert "не пересчитывает закрытый месяц" in answer
    assert "300 000" in answer


def test_identification_answer_does_not_request_documents_in_chat():
    answer = match_scenario("какие документы нужны для тарифа", "authorized").scenario.answer
    assert "паспортные данные" in answer
    assert "реквизиты" in answer
    assert "не в открытом чате" in answer


def test_uncertain_1500_ruble_figure_was_not_published():
    payload = Path("knowledge/v2/scenarios.json").read_text(encoding="utf-8")
    assert "1 500 рублей" not in payload
    assert "1500 рублей" not in payload


@pytest.mark.parametrize(
    ("message", "scenario_id", "recipient_marker"),
    [
        ("когда начислят кэшбэк", "loyalty.unconfirmed_details", "администрация"),
        ("в договоре неправильная ссылка на приложение", "contract.unconfirmed_details", "администрация"),
        ("какие последствия отказа на каждом этапе", "refusal.unconfirmed_details", "продав"),
        ("какой срок оплаты штрафа", "penalty.unconfirmed_details", "администрация"),
        ("какие документы нужны представителю организации", "account.identification_edge_case", "администрация"),
        ("какой штраф если не поставил машину на учет", "vehicle.registration_penalty_details", "официальн"),
        ("какие сведения доступны только сотруднику", "support.staff_only_details", "сотрудник"),
    ],
)
def test_unconfirmed_details_receive_safe_contextual_fallback(message, scenario_id, recipient_marker):
    decision = match_scenario(message, "authorized")
    assert decision.confidence == "high"
    assert decision.scenario is not None
    assert decision.scenario.scenario_id == scenario_id
    assert recipient_marker.lower() in decision.scenario.answer.lower()
    assert any(action.get("type") == "open_ticket" for action in decision.scenario.actions)
    assert decision.scenario.escalation.get("required_fields")


def test_loyalty_fallback_is_available_to_guest_without_promising_a_date():
    scenario = match_scenario("какого числа приходит кэшбэк", "guest").scenario
    assert scenario is not None
    assert scenario.scenario_id == "loyalty.unconfirmed_details"
    answer = scenario.answer.lower()
    assert "не установлены" in answer
    assert "создайте обращение" in answer
    assert "завтра" not in answer


def test_refusal_fallback_routes_lot_facts_to_seller_and_platform_consequences_to_admin():
    scenario = match_scenario("что будет если отказаться после оплаты", "authorized").scenario
    assert scenario is not None
    assert scenario.scenario_id == "refusal.unconfirmed_details"
    answer = scenario.answer.lower()
    assert "продав" in answer
    assert "info@migtorg.com" in answer
    assert "администрац" in answer
    assert "создать обращение" in answer


def test_penalty_fallback_does_not_invent_formula_deadline_or_outcome():
    scenario = match_scenario("какая формула штрафа", "authorized").scenario
    assert scenario is not None
    assert scenario.scenario_id == "penalty.unconfirmed_details"
    answer = scenario.answer.lower()
    assert "не подставляет неподтверждённый процент или срок" in answer
    assert "не обещает его отмену" in " ".join(scenario.facts).lower()
    assert "создайте обращение" in answer


def test_registration_penalty_fallback_rejects_uncertain_amount_and_routes_document_problem():
    scenario = match_scenario("сколько штраф за неперерегистрацию", "authorized").scenario
    assert scenario is not None
    assert scenario.scenario_id == "vehicle.registration_penalty_details"
    answer = scenario.answer.lower()
    assert "не может подтвердить его сумму" in answer
    assert "официальном постановлении" in answer
    assert "продавц" in answer
    assert "info@migtorg.com" in answer


def test_every_review_queue_gap_has_an_active_safe_fallback():
    scenarios = _scenario_map()
    queue = json.loads(Path("knowledge/v2/review_queue.json").read_text(encoding="utf-8"))
    for record in queue["records"]:
        fallback_id = record.get("safe_fallback_scenario_id")
        assert fallback_id, record["candidate_id"]
        assert fallback_id in scenarios
        assert scenarios[fallback_id]["actions"]


def test_unconfirmed_detail_is_returned_with_ticket_offer_but_without_automatic_creation():
    response = process_chat_message(
        ChatRequest(
            message="Когда начислят кэшбэк за прошлый месяц?",
            session_id="unconfirmed-detail-ticket-offer",
        )
    )

    assert response.scenario_id == "loyalty.unconfirmed_details"
    assert response.ticket_id is None
    assert any(action.type == "open_ticket" for action in response.actions)
    assert any(action.label == "Проверить кэшбэк" for action in response.actions)
