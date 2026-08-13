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
        ("где получить закрывающие документы", "payment.accounting_documents"),
        ("пришлите акт и счет фактуру", "payment.accounting_documents"),
        ("что такое возвратный депозит", "deposit.explained"),
        ("депозит это тариф или залог", "deposit.explained"),
        ("покупатель не оплатил имущество", "seller.buyer_nonpayment"),
        ("где бабки", "finance.status.clarify"),
        ("можно оплатить штраф онлайн", "penalty.explain_or_dispute"),
        ("почему мне выставили такую комиссию", "commission.explained"),
        ("на что действует скидка 10 процентов", "commission.discount"),
        ("можно оплатить наличными в офисе", "payment.methods"),
    ],
)
def test_financial_legacy_questions_route_to_v2(message, scenario_id):
    decision = match_scenario(message, "authorized")
    assert decision.confidence == "high"
    assert decision.scenario is not None
    assert decision.scenario.scenario_id == scenario_id


def test_accounting_documents_answer_contains_required_route_and_safe_data_boundary():
    answer = match_scenario("нужны документы для отчетности", "authorized").scenario.answer.casefold()
    assert "info@migtorg.com" in answer
    assert "платёжное поручение" in answer
    assert "cvc/cvv" in answer
    assert "по телефону" in answer


def test_deposit_answer_does_not_promise_refund_amount():
    answer = match_scenario("вернут ли депозит при расторжении", "authorized").scenario.answer.casefold()
    assert "задолженности" in answer
    assert "не может заранее подтвердить" in answer
    assert "вернём весь депозит" not in answer


def test_penalty_payment_answer_keeps_manual_confirmation_and_no_formula():
    answer = match_scenario("можно оплатить штраф онлайн", "authorized").scenario.answer.casefold()
    assert "с баланса" in answer
    assert "официальному счёту" in answer
    assert "ручного подтверждения" in answer
    assert "не рассчитывает" in answer


def test_informal_money_question_clarifies_instead_of_guessing_operation():
    scenario = match_scenario("что с деньгами", "guest").scenario
    assert scenario is not None
    assert scenario.scenario_id == "finance.status.clarify"
    assert len(scenario.actions) == 3
    assert all(action.get("type") == "clarify" for action in scenario.actions)
    assert "уточните" in scenario.answer.casefold()


def test_financial_batch_is_linked_and_no_longer_blocks_migration():
    migrated_ids = {
        "kb-028-тариф-и-баланс-разные-вещи",
        "kb-032-финансовые-и-бухгалтерские-вопросы",
        "kb-033-закрывающие-документы",
        "kb-075-когда-оплачивать-автомобиль",
        "kb-076-кому-оплачивается-автомобиль",
        "kb-078-скидка-при-быстрой-оплате-комиссии",
        "kb-082-что-такое-гарантийный-обеспечительный-депозит",
        "site-doc-011-obespechitelnyy-platezh-ogranicheniya",
        "faq-2026-07-10-faq-07-commission-discount-10",
        "manual-review-2026-07-11-q-028-можно-ли-оплатить-по-счету",
        "manual-review-2026-07-11-q-029-можно-ли-оплатить-наличными-в-офисе",
        "manual-review-2026-07-11-q-030-как-оплатить-тариф",
        "manual-review-2026-07-11-q-031-как-пополнить-кошелек-как-пополнить-баланс",
        "manual-review-2026-07-11-next25-q-071-почему-мне-выставили-такую-комиссию",
        "manual-review-2026-07-11-next25-q-072-откуда-взялась-такая-сумма",
        "manual-review-2026-07-11-next25-q-075-я-могу-оплатить-только-наличными",
        "manual-review-2026-07-11-remaining54-q-111-что-за-штраф-мне-выставили",
        "manual-review-2026-07-11-remaining54-q-112-могу-ли-я-делать-ставки-не-оплатив-штраф",
        "manual-review-2026-07-11-remaining54-q-113-можно-ли-оплатить-штраф-онлайн",
        "manual-review-2026-07-11-remaining54-q-114-что-такое-возвратный-депозит",
        "manual-review-2026-07-11-remaining54-q-115-не-хочу-больше-участвовать",
        "seller-offer-008-pokupatel-ne-oplatil-imuschestvo",
        "kb-money-status-informal",
    }
    inventory = json.loads(Path("knowledge/v2/legacy_inventory.json").read_text(encoding="utf-8"))
    by_id = {item["legacy_id"]: item for item in inventory["records"]}

    assert migrated_ids <= set(by_id)
    assert all(by_id[item]["status"] in {"migrated_to_v2", "merged_into_v2"} for item in migrated_ids)
    assert all(by_id[item]["target_scenario_ids"] for item in migrated_ids)
    assert all(not by_id[item]["blocks_production"] for item in migrated_ids)
