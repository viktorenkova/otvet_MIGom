import pytest

from backend.app.bot.scenario_engine import clear_scenario_cache, match_scenario


@pytest.mark.parametrize(
    ("message", "scenario_id"),
    [
        ("какие способы оплаты есть", "payment.methods"),
        ("что можно оплатить с баланса", "payment.methods"),
        ("как оплатить без комиссии", "payment.methods"),
        ("можно оплатить по QR", "payment.methods"),
        ("платеж списался но не отображается", "payment.not_visible"),
        ("какая комиссия за пополнение", "balance.topup.commission"),
        ("есть ли комиссия за пополнение", "balance.topup.commission"),
        ("какая комиссия через СБП", "balance.topup.commission"),
        ("комиссия при пополнении кошелька", "balance.topup.commission"),
        ("сколько удержат через СБП", "balance.topup.commission"),
        ("какой процент комиссии за лот", "commission.explained"),
        ("скидка за быструю оплату комиссии", "commission.discount"),
        ("неоплаченная комиссия заблокирует аккаунт", "commission.unpaid"),
        ("нужно платить при статусе выигран", "lot.payment.start"),
        ("кто выставляет счет за лот", "lot.payment.details"),
        ("может организация оплатить мой лот", "lot.payment.details"),
        ("где виден статус оплаты лота", "lot.payment.details"),
        ("что будет если просрочить оплату лота", "lot.payment.overdue"),
        ("за что могут расторгнуть договор", "contract.termination_and_restriction"),
        ("могут скрыть лоты конкретного продавца", "contract.termination_and_restriction"),
    ],
)
def test_expert_approved_questions_match_published_scenarios(message: str, scenario_id: str):
    clear_scenario_cache()
    decision = match_scenario(message, "authorized")
    assert decision.confidence == "high"
    assert decision.scenario is not None
    assert decision.scenario.scenario_id == scenario_id


def test_lot_paid_status_is_not_described_as_payment_of_vehicle():
    clear_scenario_cache()
    decision = match_scenario("где виден статус оплаты лота", "authorized")
    answer = decision.scenario.answer
    assert "только к комиссии площадки" in answer
    assert "подтверждает оплату самого автомобиля" not in answer


def test_payment_check_never_requests_secret_card_data():
    clear_scenario_cache()
    decision = match_scenario("какие данные нужны для проверки платежа", "authorized")
    answer = decision.scenario.answer
    assert "CVC/CVV" in answer
    assert "код из SMS" in answer


def test_balance_topup_answer_contains_current_fees():
    clear_scenario_cache()
    decision = match_scenario("какая комиссия за пополнение баланса", "authorized")
    answer = decision.scenario.answer
    assert "2,5%" in answer
    assert "1%" in answer
    assert "СБП" in answer


def test_cashback_answer_contains_confirmed_groups_and_per_buyer_limit():
    clear_scenario_cache()
    decision = match_scenario("как работает кэшбэк MIGTORG", "authorized")
    assert decision.scenario is not None
    assert decision.scenario.scenario_id == "commission.discount"
    answer = decision.scenario.answer
    assert "1–5" in answer
    assert "100%" in answer
    assert "6–10" in answer
    assert "50%" in answer
    assert "300 000" in answer
    assert "каждого покупателя" in answer


def test_contract_answer_uses_approved_seller_specific_visibility_rule():
    clear_scenario_cache()
    decision = match_scenario("могут скрыть лоты конкретного продавца", "authorized")
    assert decision.scenario is not None
    assert decision.scenario.scenario_id == "contract.termination_and_restriction"
    answer = decision.scenario.answer
    assert "срок и условия снятия определяет продавец" in answer
    assert "ветке переписки по лоту" in answer
    assert "всей платформе" in answer
    assert "7 календарных дней" in answer
    assert "не возвращается" in answer
