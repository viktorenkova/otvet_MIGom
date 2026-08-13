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
    ("ваш сайт говно", "feedback.platform_complaint"),
    ("площадка бесполезная", "feedback.platform_complaint"),
    ("вы вообще ничего нормально сделать не можете", "feedback.platform_complaint"),
    ("не работает ничего", "technical.site_error"),
    ("добавьте новый фильтр", "feedback.improvement_suggestion"),
    ("бот не помог", "feedback.bot_answer_complaint"),
    ("фильтр работает некорректно", "technical.catalog_search_filter"),
    ("у меня некорректно работает фильтр", "technical.catalog_search_filter"),
    ("после выбора марки фильтр сбрасывается", "technical.catalog_search_filter"),
    ("неправильно работает поиск", "technical.catalog_search_filter"),
    ("поиск не находит лот", "technical.catalog_search_filter"),
    ("фото пропало", "technical.lot_image_missing"),
    ("в карточке лота не загружаются фотографии", "technical.lot_image_missing"),
    ("сделайте поиск по VIN", "feedback.improvement_suggestion"),
])
def test_feedback_and_technical_questions_route_to_correct_scenario(message, scenario_id):
    decision = match_scenario(message, "guest")
    assert decision.confidence == "high"
    assert decision.scenario and decision.scenario.scenario_id == scenario_id


def test_platform_complaint_stays_calm_and_does_not_promise_a_fix():
    scenario = match_scenario("ваш сайт говно", "guest").scenario
    answer = scenario.answer.casefold()
    assert "давайте коротко зафиксируем" in answer
    assert "что вы ожидали получить" in answer
    assert "нельзя обещать исправление или срок" in answer
    assert any(action["type"] == "open_ticket" for action in scenario.actions)


def test_improvement_suggestion_does_not_promise_implementation():
    answer = match_scenario("добавьте новый фильтр", "guest").scenario.answer.casefold()
    assert "какую задачу она решит" in answer
    assert "срок и сам факт реализации" in answer


def test_bot_complaint_collects_safe_context_for_handoff():
    scenario = match_scenario("бот не помог", "guest").scenario
    answer = scenario.answer.casefold()
    assert "что осталось непонятно" in answer
    assert "не отправляйте пароль" in answer
    assert "без обещания результата" in answer
    assert any(action["type"] == "open_ticket" for action in scenario.actions)


def test_search_filter_error_collects_reproduction_details():
    scenario = match_scenario("фильтр работает некорректно", "guest").scenario
    answer = scenario.answer.casefold()
    assert "сбросьте фильтры" in answer
    assert "точный запрос или выбранные параметры" in answer
    assert "номер или ссылку" in answer
    assert any(action["type"] == "open_ticket" for action in scenario.actions)


def test_missing_image_warns_against_bidding_without_material_photo():
    scenario = match_scenario("фото пропало", "guest").scenario
    answer = scenario.answer.casefold()
    assert "другом браузере" in answer
    assert "не делайте ставку" in answer
    assert "info@migtorg.com" in answer


@pytest.mark.parametrize(("message", "scenario_id"), [
    ("у меня некорректно работает фильтр", "technical.catalog_search_filter"),
    ("после выбора марки фильтр сбрасывается", "technical.catalog_search_filter"),
    ("в карточке лота не загружаются фотографии", "technical.lot_image_missing"),
    ("сделайте поиск по VIN", "feedback.improvement_suggestion"),
])
def test_problem_recheck_phrases_route_through_full_dialogue_pipeline(message, scenario_id):
    response = process_chat_message(
        ChatRequest(message=message, session_id=f"problem-recheck-{abs(hash(message))}")
    )

    assert response.scenario_id == scenario_id
    assert response.confidence_level == "high"
    assert response.needs_ticket is True


def test_feedback_technical_batch_migrated_9_legacy_records():
    prefixes = ("kb-119", "kb-120", "kb-121", "kb-122", "manual-review-2026-07-11-remaining54-q-119", "manual-review-2026-07-11-remaining54-q-120", "manual-review-2026-07-11-remaining54-q-121", "manual-review-2026-07-11-remaining54-q-122", "manual-review-2026-07-11-remaining54-q-123")
    scenarios = json.loads(Path("knowledge/v2/scenarios.json").read_text(encoding="utf-8"))["records"]
    batch = {
        legacy
        for row in scenarios
        for legacy in row["legacy_ids"]
        if legacy.startswith(prefixes)
    }
    assert len(batch) == 9
    inventory = json.loads(Path("knowledge/v2/legacy_inventory.json").read_text(encoding="utf-8"))
    by_id = {row["legacy_id"]: row for row in inventory["records"]}
    assert all(not by_id[item]["blocks_production"] for item in batch)
