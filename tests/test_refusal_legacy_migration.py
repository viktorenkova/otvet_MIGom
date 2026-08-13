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
        ("можно отказаться от выигранного лота", "refusal.change_mind"),
        ("выиграл но передумал", "refusal.change_mind"),
        ("лот передали но он мне не нужен", "refusal.change_mind"),
        ("что будет если просто пропасть", "refusal.no_response"),
        ("не буду отвечать продавцу", "refusal.no_response"),
        ("какой отказ считается обоснованным", "refusal.evidence"),
        ("что значит обоснованный отказ", "refusal.evidence"),
        ("какой размер дополнительных повреждений для отказа", "refusal.invalid_reasons"),
        ("машина не соответствует описанию", "refusal.evidence"),
        ("продавец не отвечает по отказу", "refusal.seller_decision"),
        ("покупатель отказался от моего имущества", "seller.buyer_refusal"),
        ("когда продавцу придет акт отказа", "seller.buyer_refusal"),
    ],
)
def test_refusal_questions_route_to_v2(message, scenario_id):
    decision = match_scenario(message, "authorized")
    assert decision.confidence == "high"
    assert decision.scenario is not None
    assert decision.scenario.scenario_id == scenario_id


def _answer(message):
    return match_scenario(message, "authorized").scenario.answer.casefold()


def test_change_of_mind_answer_does_not_invent_automatic_penalty():
    answer = _answer("выиграл но передумал")
    assert "не является мотивированным отказом" in answer
    assert "info@migtorg.com" in answer
    assert "не подтверждает штраф" in answer
    assert "штраф составит" not in answer
    assert "отказ точно примут" not in answer


def test_damage_threshold_is_not_presented_as_automatic_approval():
    answer = _answer("какой размер дополнительных повреждений для отказа")
    assert "менее 10%" in answer
    assert "сам по себе не подтверждает" in answer
    assert "существенное" in answer
    assert "может обосновывать отказ" in answer
    assert "10% гарантирует" not in answer


def test_seller_refusal_answer_keeps_role_and_manual_review_boundary():
    answer = _answer("покупатель отказался от моего имущества")
    assert "в течение 3 рабочих дней" in answer
    assert "с даты отказа" in answer
    assert "бот не подтверждает обоснованность автоматически" in answer


def test_refusal_batch_is_linked_and_no_longer_blocks_migration():
    migrated_ids = {
        "kb-069-можно-ли-отказаться-от-выигранного-лота",
        "kb-070-что-будет-если-просто-пропасть",
        "site-doc-007-spornye-osnovaniya-otkaza",
        "faq-2026-07-10-faq-09-refuse-won-lot",
        "manual-review-2026-07-11-remaining54-q-097-мне-передали-лот-но-он-мне-не-нужен",
        "manual-review-2026-07-11-remaining54-q-098-какой-отказ-считается-обоснованным",
        "manual-review-2026-07-11-remaining54-q-100-что-значит-обоснованный-отказ",
        "manual-review-2026-07-11-remaining54-q-101-какой-размер-доп-повреждений-входит-в-обоснованный-отказ",
        "seller-offer-007-pokupatel-otkazalsya-ot-imuschestva",
    }
    inventory = json.loads(Path("knowledge/v2/legacy_inventory.json").read_text(encoding="utf-8"))
    by_id = {item["legacy_id"]: item for item in inventory["records"]}

    assert migrated_ids <= set(by_id)
    assert all(by_id[item]["status"] in {"migrated_to_v2", "merged_into_v2"} for item in migrated_ids)
    assert all(by_id[item]["target_scenario_ids"] for item in migrated_ids)
    assert all(not by_id[item]["blocks_production"] for item in migrated_ids)
