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
    ("как получить контакт продавца", "transfer.notification_contact"),
    ("пришло письмо просят реквизиты", "transfer.requisites"),
    ("документы не приходят", "documents.preparation_delay"),
    ("как подписать договор", "contract.parties_signing"),
    ("как запросить осмотр", "inspection.arrange"),
    ("меня не пустили на осмотр", "inspection.problem"),
    ("кто выдает лот", "pickup.access_issuer"),
    ("нужна ли нотариальная доверенность", "pickup.representative"),
    ("может ли другой человек оплатить лот", "documents.payer_change"),
    ("можно указать другую сумму в договоре", "documents.actual_amount"),
])
def test_deal_document_questions_route_to_v2(message, scenario_id):
    decision = match_scenario(message, "authorized")
    assert decision.confidence == "high"
    assert decision.scenario and decision.scenario.scenario_id == scenario_id


def test_requisites_answer_keeps_sensitive_data_out_of_chat():
    answer = match_scenario("какие реквизиты отправить физлицу", "authorized").scenario.answer.casefold()
    assert "паспортные данные" in answer
    assert "официальной ветке" in answer
    assert "не присылайте паспорт" in answer
    assert "cvc/cvv" in answer


def test_payer_change_requires_reissued_documents_before_payment():
    answer = match_scenario("можно поменять плательщика", "authorized").scenario.answer.casefold()
    assert "до оплаты" in answer
    assert "новый счёт" in answer
    assert "старый счёт" in answer and "не оплачивайте" in answer


def test_false_contract_amount_is_refused_safely():
    answer = match_scenario("можно указать другую сумму в договоре", "authorized").scenario.answer.casefold()
    assert "фактическая сумма сделки" in answer
    assert "нельзя" in answer
    assert "официально исправленные" in answer


def test_inspection_problem_does_not_promise_automatic_refusal():
    answer = match_scenario("на осмотре обнаружил расхождения", "authorized").scenario.answer.casefold()
    assert "фотографии или видео" in answer
    assert "не гарантируют мотивированный отказ" in answer


def test_deal_documents_batch_migrated_28_legacy_records():
    targets = {
        "transfer.notification_contact", "transfer.requisites", "documents.preparation_delay",
        "contract.parties_signing", "inspection.arrange", "inspection.problem",
        "pickup.access_issuer", "pickup.representative", "documents.payer_change",
        "documents.actual_amount",
    }
    prefixes = (
        "faq-2026-07-10-faq-08", "kb-059", "kb-060", "kb-062", "kb-067", "kb-068",
        "site-doc-014", "manual-review-2026-07-11-next25-q-054",
        "manual-review-2026-07-11-next25-q-058", "manual-review-2026-07-11-next25-q-070",
        "manual-review-2026-07-11-next25-q-073", "manual-review-2026-07-11-next25-q-074",
        "manual-review-2026-07-11-remaining54-q-076", "manual-review-2026-07-11-remaining54-q-077",
        "manual-review-2026-07-11-remaining54-q-078", "manual-review-2026-07-11-remaining54-q-079",
        "manual-review-2026-07-11-remaining54-q-082", "manual-review-2026-07-11-remaining54-q-083",
        "manual-review-2026-07-11-remaining54-q-085", "manual-review-2026-07-11-remaining54-q-088",
        "manual-review-2026-07-11-remaining54-q-089", "manual-review-2026-07-11-remaining54-q-091",
        "manual-review-2026-07-11-remaining54-q-092", "manual-review-2026-07-11-remaining54-q-093",
        "manual-review-2026-07-11-remaining54-q-094", "manual-review-2026-07-11-remaining54-q-095",
        "manual-review-2026-07-11-remaining54-q-096", "manual-review-2026-07-11-remaining54-q-131",
    )
    scenarios = json.loads(Path("knowledge/v2/scenarios.json").read_text(encoding="utf-8"))["records"]
    batch = {
        legacy
        for row in scenarios if row["scenario_id"] in targets
        for legacy in row["legacy_ids"] if legacy.startswith(prefixes)
    }
    assert len(batch) == 28
    inventory = json.loads(Path("knowledge/v2/legacy_inventory.json").read_text(encoding="utf-8"))
    by_id = {row["legacy_id"]: row for row in inventory["records"]}
    assert all(not by_id[item]["blocks_production"] for item in batch)
