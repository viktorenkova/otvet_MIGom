import json
from pathlib import Path

import pytest

from backend.app.bot.scenario_engine import clear_scenario_cache, match_scenario
from backend.app.main import process_chat_message
from backend.app.models.chat import ChatRequest


REVIEW_PATH = Path("knowledge/v2/manual_dialogue_review.json")


@pytest.fixture(autouse=True)
def clear_cache():
    clear_scenario_cache()
    yield
    clear_scenario_cache()


def _review():
    return json.loads(REVIEW_PATH.read_text(encoding="utf-8"))


def test_manual_dialogue_review_is_complete_and_internally_consistent():
    review = _review()
    assert review["completed"] is True
    assert review["systematic_false_answers_found"] is False
    assert review["review_owner"]
    assert review["reviewed_at"] == "2026-08-12"
    assert review["reviewed_cases"] == len(review["cases"]) == 35
    assert review["approved_answers"] + review["safe_clarifications"] == 35
    assert review["issues_found"] == review["issues_resolved"] == len(review["resolved_issues"])


@pytest.mark.parametrize("case", _review()["cases"], ids=lambda case: case["message"])
def test_reviewed_dialogue_routing_remains_stable(case):
    response = process_chat_message(
        ChatRequest(message=case["message"], session_id=f"release-review-{abs(hash(case['message']))}")
    )
    assert response.scenario_id == case["expected_scenario_id"]
    assert response.confidence_level == case["expected_confidence"]


def test_resolved_refund_request_uses_application_flow():
    answer = match_scenario("верните деньги за разовый тариф", "authorized").scenario.answer.casefold()
    assert "кнопку возврата в личном кабинете" in answer
    assert "операцию всё равно проверяет сотрудник" in answer
    assert "какой тариф выбрать" not in answer


def test_resolved_hidden_damage_request_uses_evidence_flow():
    answer = match_scenario("скрытое повреждение могу отказаться", "authorized").scenario.answer.casefold()
    assert "существенное нераскрытое повреждение" in answer
    assert "акт по форме приложения № 2" in answer
    assert "фотограф" in answer
