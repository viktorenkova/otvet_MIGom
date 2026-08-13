from uuid import uuid4

import pytest

from backend.app.bot.scenario_engine import clear_scenario_cache
from backend.app.main import process_chat_message
from backend.app.models.chat import ChatRequest


@pytest.fixture(autouse=True)
def clear_cache():
    clear_scenario_cache()
    yield
    clear_scenario_cache()


def test_generic_how_to_use_question_offers_meaningful_paths():
    response = process_chat_message(
        ChatRequest(message="как работать", session_id=f"start-navigation-{uuid4()}")
    )

    assert response.scenario_id is None
    assert response.resolution == "clarified"
    assert response.confidence_level == "medium"
    assert "как устроена площадка" in response.answer.casefold()
    assert response.clarifying_options == [
        "Что такое MIGTORG и как распределены роли",
        "Как начать покупать и участвовать в торгах",
        "Как подключиться и стать продавцом",
        "Другая тема",
    ]


def test_selected_buying_path_uses_the_linked_scenario_not_free_text_search():
    session_id = f"start-navigation-{uuid4()}"
    first = process_chat_message(ChatRequest(message="как работать", session_id=session_id))

    response = process_chat_message(
        ChatRequest(message=first.clarifying_options[1], session_id=session_id)
    )

    assert response.scenario_id == "buyer.get_started"
    assert response.confidence_level == "high"
    assert "зарегистрируйтесь" in response.answer.casefold()
    assert "завершённые торги помогают" not in response.answer.casefold()


@pytest.mark.parametrize(
    ("option_index", "scenario_id", "action_types"),
    [
        (0, "platform.about", ["navigate", "open_ticket"]),
        (1, "buyer.get_started", ["navigate", "clarify"]),
        (2, "seller.get_started", ["navigate", "open_ticket"]),
    ],
)
def test_every_start_navigation_topic_opens_expected_scenario(option_index, scenario_id, action_types):
    session_id = f"start-navigation-{uuid4()}"
    first = process_chat_message(ChatRequest(message="как работать", session_id=session_id))

    response = process_chat_message(
        ChatRequest(message=first.clarifying_options[option_index], session_id=session_id)
    )

    assert response.scenario_id == scenario_id
    assert response.confidence_level == "high"
    assert [action.type for action in response.actions] == action_types
    assert response.model_used == "mock"


def test_other_topic_returns_to_free_text_without_wrong_answer():
    session_id = f"start-navigation-{uuid4()}"
    first = process_chat_message(ChatRequest(message="как работать", session_id=session_id))

    response = process_chat_message(
        ChatRequest(message=first.clarifying_options[3], session_id=session_id)
    )

    assert response.scenario_id is None
    assert response.resolution == "clarified"
    assert response.clarifying_options == []
    assert "опишите вопрос своими словами" in response.answer.casefold()


def test_buyer_tariff_button_opens_choice_before_connection_instructions():
    session_id = f"start-navigation-{uuid4()}"
    first = process_chat_message(ChatRequest(message="как работать", session_id=session_id))
    buyer = process_chat_message(
        ChatRequest(message=first.clarifying_options[1], session_id=session_id)
    )
    tariff_action = next(action for action in buyer.actions if action.id == "buyer.tariff")

    response = process_chat_message(
        ChatRequest(
            message=tariff_action.label,
            session_id=session_id,
            selected_action_id=tariff_action.id,
        )
    )

    assert response.scenario_id == "tariff.choose"
    assert [action.label for action in response.actions] == [
        "Одна покупка",
        "Регулярные торги",
        "Демо и Имущество",
    ]
    assert "выберите цель" in response.answer.casefold()


@pytest.mark.parametrize(
    ("action_id", "scenario_id"),
    [
        ("tariff.choose.one_time", "tariff.one_time"),
        ("tariff.choose.premium", "tariff.premium"),
        ("tariff.choose.demo", "tariff.demo"),
    ],
)
def test_each_tariff_choice_button_opens_its_own_scenario(action_id, scenario_id):
    session_id = f"start-navigation-{uuid4()}"
    first = process_chat_message(ChatRequest(message="как работать", session_id=session_id))
    buyer = process_chat_message(
        ChatRequest(message=first.clarifying_options[1], session_id=session_id)
    )
    tariff_action = next(action for action in buyer.actions if action.id == "buyer.tariff")
    tariff_choice = process_chat_message(
        ChatRequest(
            message=tariff_action.label,
            session_id=session_id,
            selected_action_id=tariff_action.id,
        )
    )
    selected = next(action for action in tariff_choice.actions if action.id == action_id)

    response = process_chat_message(
        ChatRequest(
            message=selected.label,
            session_id=session_id,
            selected_action_id=selected.id,
        )
    )

    assert response.scenario_id == scenario_id
    assert response.confidence_level == "high"
    assert response.model_used == "mock"


def test_navigation_and_ticket_buttons_have_valid_widget_contracts():
    for option_index in (0, 1, 2):
        session_id = f"start-navigation-{uuid4()}"
        first = process_chat_message(ChatRequest(message="как работать", session_id=session_id))
        branch = process_chat_message(
            ChatRequest(message=first.clarifying_options[option_index], session_id=session_id)
        )
        for action in branch.actions:
            if action.type == "navigate":
                assert action.payload.get("url") == "https://migtorg.com/auth/sign-up"
            if action.type == "open_ticket":
                assert action.requires_confirmation is True


def test_generic_bidding_topic_does_not_route_to_completed_auction_analytics():
    response = process_chat_message(
        ChatRequest(message="Торги и ставки", session_id=f"start-navigation-{uuid4()}")
    )

    assert response.scenario_id == "auction.formats"
    assert response.confidence_level == "high"
    assert "в открытых торгах" in response.answer.casefold()
    assert "завершённые торги помогают" not in response.answer.casefold()
