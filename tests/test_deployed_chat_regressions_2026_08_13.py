from uuid import uuid4

import pytest

from backend.app.bot.answer_generator import generate_answer
from backend.app.bot.knowledge_search import clear_knowledge_cache, get_article_by_id
from backend.app.bot.scenario_engine import clear_scenario_cache
from backend.app.config import Settings
from backend.app.main import process_chat_message, settings
from backend.app.models.chat import ChatRequest
from backend.app.models.llm import LLMResult


def ask(message: str, session_id: str | None = None):
    return process_chat_message(
        ChatRequest(message=message, session_id=session_id or f"deploy-regression-{uuid4()}")
    )


@pytest.fixture(autouse=True)
def deterministic_answers(monkeypatch):
    monkeypatch.setattr(settings, "llm_enabled", False)
    clear_knowledge_cache()
    clear_scenario_cache()
    yield
    clear_knowledge_cache()
    clear_scenario_cache()


@pytest.mark.parametrize(
    "message",
    [
        "соедините с Михаилом",
        "соедините с Алексеем",
        "пусть со мной свяжется Михаил",
        "пусть перезвонит Игорь",
        "позовите менеджера",
    ],
)
def test_employee_requests_use_written_support_without_repeating_a_name(message):
    response = ask(message)

    assert response.scenario_id in {"support.contact", "support.callback"}
    assert response.intent == "support"
    assert response.confidence_level == "high"
    assert "письмен" in response.answer.casefold()
    assert "михаил" not in response.answer.casefold()
    assert "алекс" not in response.answer.casefold()
    assert any(action.type == "open_ticket" for action in response.actions)


@pytest.mark.parametrize(
    ("message", "scenario_id"),
    [
        ("сайт не работает", "technical.site_error"),
        ("фильтры не работают", "technical.catalog_search_filter"),
    ],
)
def test_short_technical_failures_do_not_fall_out_of_scope(message, scenario_id):
    response = ask(message)

    assert response.scenario_id == scenario_id
    assert response.intent == "support"
    assert response.confidence_level == "high"
    assert "не удалось определить связь" not in response.answer.casefold()


@pytest.mark.parametrize(
    "message",
    ["как попасть к вам в офис", "хочу к вам в офис приехать", "где находится офис", "где ваш адрес"],
)
def test_address_requests_clarify_purpose_before_disclosing_office_address(message):
    response = ask(message)

    assert response.action == "clarify"
    assert response.scenario_id is None
    assert "смирновская" not in response.answer.casefold()
    assert response.clarifying_options == [
        "Визит в офис и оформление пропуска",
        "Как организовать осмотр автомобиля",
        "Доступ на стоянку и кто выдаёт лот",
        "Другая тема",
    ]


def test_office_address_is_available_after_user_selects_office_purpose():
    session_id = f"office-purpose-{uuid4()}"
    first = ask("где ваш адрес", session_id)
    response = ask(first.clarifying_options[0], session_id)

    assert response.scenario_id == "support.office_visit"
    assert "смирновская" in response.answer.casefold()
    assert "предварительной записи" in response.answer.casefold()


def test_connect_premium_answer_excludes_unasked_demo_instructions():
    response = ask("подключить премиум")

    assert response.scenario_id == "tariff.connect"
    assert "выберите «премиум»" in response.answer.casefold()
    assert "демо" not in response.answer.casefold()
    assert len(response.answer) < 500


@pytest.mark.parametrize("message", ["возврат", "возврат денежных средств"])
def test_generic_refund_request_asks_which_refund_is_needed(message):
    response = ask(message)

    assert response.action == "clarify"
    assert response.answer == "Что именно вы хотите вернуть?"
    assert response.clarifying_options[:4] == [
        "Какие средства можно вернуть",
        "Как подать запрос на возврат",
        "Куда возвращаются деньги",
        "Срок и статус возврата",
    ]


def test_refund_template_request_returns_the_document_not_a_generic_fallback():
    response = ask("шаблон возврата")

    assert response.scenario_id == "refund.application"
    assert response.action == "show_document"
    assert "шаблон заявления" in response.answer.casefold()
    assert response.template_links
    assert response.template_links[0].url.endswith("Шаблон_заявления_на_возврат_депозита.docx")


def test_short_bid_definition_does_not_route_to_missing_bid_troubleshooting():
    response = ask("ставка это")

    assert response.scenario_id == "bid.price_terms"
    assert response.action == "answer"
    assert "ценовое предложение" in response.answer.casefold()
    assert "не отображается" not in response.answer.casefold()


@pytest.mark.parametrize(
    "message",
    ["тинек гасится", "альфа молчит", "реник не отвечает", "ресо гасится"],
)
def test_colloquial_seller_names_route_to_seller_contact_delay(message):
    response = ask(message)

    assert response.scenario_id == "transfer.seller_no_response"
    assert response.intent == "transfer"
    assert "ожидает подтверждения" in response.answer.casefold()
    assert "номер лота" in response.answer.casefold()
    assert "папку «спам»" not in response.answer.casefold()


def test_rude_followup_does_not_inherit_an_employee_request():
    session_id = f"employee-then-rude-{uuid4()}"
    first = ask("соедините с Михаилом", session_id)
    second = ask("хуй", session_id)

    assert first.action != "clarify"
    assert second.scenario_id is None
    assert "обратн" not in second.answer.casefold()
    assert "михаил" not in second.answer.casefold()


def test_llm_output_is_deduplicated_and_cannot_echo_requested_employee_name(monkeypatch):
    class FakeProvider:
        def generate(self, request):
            return LLMResult(
                text="Михаил ответит позже. Создайте письменное обращение. Создайте письменное обращение.",
                provider="fake",
                model="fake-model",
                task_type=request.task_type,
            )

    monkeypatch.setattr("backend.app.bot.answer_generator.build_llm_provider", lambda _settings: FakeProvider())
    article = get_article_by_id("support.contact", "guest")
    result = generate_answer(
        "соедините с Михаилом",
        "support",
        "guest",
        article,
        False,
        settings=Settings(
            llm_enabled=True,
            llm_provider="fake",
            llm_primary_model="fake-model",
            llm_fallback_model="fake-model",
        ),
    )

    assert "михаил" not in result.answer.casefold()
    assert result.answer.casefold().count("создайте письменное обращение") == 1
