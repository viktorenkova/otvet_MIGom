import pytest

from backend.app.bot.scenario_engine import clear_scenario_cache, match_scenario
from backend.app.bot.knowledge_search import clear_knowledge_cache, search_knowledge_match


@pytest.mark.parametrize(
    ("message", "scenario_id"),
    [
        ("какие деньги можно вернуть", "refund.eligibility"),
        ("вернут комиссию если сделка не состоялась", "refund.eligibility"),
        ("можно вернуть 1200 за позицию ставки", "refund.eligibility"),
        ("как подать заявление на возврат", "refund.application"),
        ("можно оформить возврат в чате", "refund.application"),
        ("куда вернут деньги", "refund.destination"),
        ("можно оставить возврат на балансе", "refund.destination"),
        ("сколько ждать возврат", "refund.timing_status"),
        ("срок возврата депозита премиум", "refund.timing_status"),
        ("почему отказали в возврате", "refund.denied_or_blocked"),
        ("могу вернуть баланс если есть штраф", "refund.denied_or_blocked"),
    ],
)
def test_expert_part2_refund_questions_use_published_scenarios(message: str, scenario_id: str):
    clear_scenario_cache()
    decision = match_scenario(message, "authorized")
    assert decision.confidence == "high"
    assert decision.scenario is not None
    assert decision.scenario.scenario_id == scenario_id


def test_refund_eligibility_distinguishes_refundable_and_nonrefundable_payments():
    clear_scenario_cache()
    answer = match_scenario("какие деньги можно вернуть", "authorized").scenario.answer
    assert "обеспечительная часть" in answer
    assert "плата за доступ" in answer
    assert "не возвращается" in answer
    assert "1 200" in answer
    assert "комиссия MIGTORG возвращается полностью" in answer


def test_premium_refund_has_confirmed_deadline_and_application_rule():
    clear_scenario_cache()
    answer = match_scenario("срок возврата депозита премиум", "authorized").scenario.answer
    assert "до 15 дней" in answer
    assert "подписанного заявления" in answer


def test_refund_answers_never_request_card_secrets():
    clear_scenario_cache()
    answer = match_scenario("как подать заявление на возврат", "authorized").scenario.answer
    assert "CVC/CVV" in answer
    assert "кода из SMS" in answer
    assert "полного номера карты" in answer


def test_generic_refund_request_asks_what_payment_to_return():
    clear_scenario_cache()
    decision = match_scenario("как вернуть деньги", "authorized")
    assert decision.scenario is None
    assert decision.confidence == "medium"
    assert decision.clarifying_question == "Что именно вы хотите вернуть?"
    assert {item.scenario_id for item in decision.candidates} == {
        "refund.eligibility",
        "refund.application",
        "refund.destination",
        "refund.timing_status",
    }


def test_runtime_search_preserves_generic_refund_clarification():
    clear_scenario_cache()
    clear_knowledge_cache()
    result = search_knowledge_match("как вернуть деньги", "refund", "authorized")
    assert result.article is None
    assert result.confidence == "medium"
    assert result.clarifying_question == "Что именно вы хотите вернуть?"
