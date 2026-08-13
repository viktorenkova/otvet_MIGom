import pytest

from backend.app.bot.intent_classifier import classify_intent
from backend.app.bot.knowledge_search import (
    _canonical_phrase_rules,
    has_ambiguous_phrase_rule,
    load_articles,
    search_knowledge_match,
)
from backend.app.bot.text_processing import load_matching_config
from backend.app.main import process_chat_message
from backend.app.models.chat import ChatRequest


def ask(message: str, session_suffix: str):
    return process_chat_message(
        ChatRequest(message=message, session_id=f"manual-review-{session_suffix}")
    )


@pytest.mark.parametrize(
    "message",
    [
        "Кто такой Глеб Миронов",
        "почему трава зеленая",
        "поросенок пигги",
    ],
)
def test_out_of_scope_questions_do_not_receive_knowledge_base_answers(message: str):
    response = ask(message, str(abs(hash(message))))

    assert response.intent == "unknown"
    assert response.action == "clarify"
    assert response.confidence_level == "low"
    assert "только на вопросы о работе MIGTORG" in response.answer


def test_informal_accounting_documents_question_uses_document_scenario():
    response = ask("где мои доки", "accounting-documents")

    assert response.intent == "payment"
    assert response.needs_ticket is True
    assert "info@migtorg.com" in response.answer
    assert "номер лота" in response.answer
    assert "ассортимент" not in response.answer
    assert "статус платежа" not in response.answer


def test_missing_lot_answer_contains_only_user_facing_instructions():
    response = ask("где мой лот", "missing-lot")

    assert response.intent == "lot"
    assert response.needs_ticket is True
    assert "создайте обращение" in response.answer.casefold()
    assert "техническим/процессным" not in response.answer
    assert "нужно запросить" not in response.answer.casefold()
    assert "передать специалисту" not in response.answer.casefold()


@pytest.mark.parametrize(
    ("message", "expected_intent"),
    [
        ("Начислили штраф", "penalty"),
        ("Не могу сделать ставку", "support"),
        ("Оплатил тариф, доступа нет", "tariffs"),
        ("Лот не передают", "lot"),
    ],
)
def test_existing_quick_scenarios_keep_their_routing(message: str, expected_intent: str):
    response = ask(message, expected_intent)

    assert response.intent == expected_intent
    assert response.confidence_level == "high"


def test_active_answers_do_not_expose_internal_instruction_language():
    forbidden_markers = (
        "бот должен",
        "бот не должен",
        "пользователь должен",
        "нужно запросить",
        "затем передать",
        "передать специалисту",
        "техническим/процессным",
        "внутренним материалам",
        "корректный ответ",
        "ответ:",
        "эскалир",
        "маршрутиз",
    )

    offenders = {
        article.slug: marker
        for article in load_articles()
        for marker in forbidden_markers
        if marker in (article.user_answer or "").casefold()
    }
    assert offenders == {}


def test_penalty_or_deposit_quick_topic_asks_user_to_choose_one_topic():
    response = ask("Вопрос по штрафу или депозиту", "penalty-or-deposit")

    assert response.action == "clarify"
    assert response.clarifying_options == ["Штраф", "Депозит"]
    assert response.needs_ticket is False


@pytest.mark.parametrize("message", ["где бабки", "где мои бабки", "куда бабки", "что с деньгами"])
def test_informal_money_questions_show_requested_topic_menu(message: str):
    response = ask(message, f"money-{abs(hash(message))}")

    assert response.action == "clarify"
    assert response.clarifying_options == [
        "Оплата и платежи",
        "Возврат денежных средств",
        "Другая тема",
    ]


@pytest.mark.parametrize("message", ["мало инфы в карточке", "нет инфы в карточке лота"])
def test_missing_lot_card_information_uses_seller_information_policy(message: str):
    response = ask(message, f"lot-info-{abs(hash(message))}")

    assert response.intent == "lot"
    assert response.needs_ticket is False
    assert "определяет продавец" in response.answer
    assert "MIGTORG не может дополнить карточку" in response.answer


def test_parking_location_is_not_confused_with_pickup_denial():
    response = ask("где стоянка", "parking-location")

    assert response.intent == "lot"
    assert response.needs_ticket is False
    assert "продавец сообщит после подтверждения передачи лота" in response.answer
    assert "автомобиль не выдают" not in response.answer


@pytest.mark.parametrize(
    "message",
    ["торги не состоялись", "могут ли торги не состояться", "почему торги не состоялись"],
)
def test_auction_without_bids_has_dedicated_answer(message: str):
    response = ask(message, f"auction-without-bids-{abs(hash(message))}")

    assert response.intent == "bidding"
    assert response.needs_ticket is False
    assert "не было ни одной ставки" in response.answer
    assert "выставить его повторно" in response.answer
    assert "более низкой стартовой стоимостью" in response.answer


@pytest.mark.parametrize(
    "message",
    ["претензия", "обращение", "создать обращение", "составить обращение", "подать жалобу"],
)
def test_ticket_requests_start_with_category_selection(message: str):
    response = ask(message, f"ticket-request-{abs(hash(message))}")

    assert response.action == "clarify"
    assert response.clarifying_options == [
        "Лот, торги или передача",
        "Оплата или возврат",
        "Штраф, депозит или отказ",
        "Работа сайта или другая тема",
    ]


def test_ticket_category_selection_returns_visible_ticket_action():
    session_id = "manual-review-ticket-category-flow"
    process_chat_message(ChatRequest(message="создать обращение", session_id=session_id))
    response = process_chat_message(ChatRequest(message="Оплата или возврат", session_id=session_id))

    assert response.intent == "payment"
    assert response.action == "create_ticket"
    assert response.needs_ticket is True
    assert "Нажмите «Создать обращение» ниже" in response.answer


def test_payment_crediting_answer_explicitly_opens_ticket_flow():
    response = ask("зачисление платежа", "payment-crediting")

    assert response.intent == "payment"
    assert response.needs_ticket is True
    assert response.action == "create_ticket"
    assert "нажмите «создать обращение»" in response.answer.casefold()
    assert "дату, сумму и способ оплаты" in response.answer


@pytest.mark.parametrize("message", ["премиум", "премиум это", "что такое премиум"])
def test_premium_short_questions_use_tariff_overview(message: str):
    response = ask(message, f"premium-overview-{abs(hash(message))}")

    assert response.intent == "tariffs"
    assert response.confidence_level == "high"
    assert "тариф для регулярного участия" in response.answer
    assert "неограниченное количество ставок" in response.answer


@pytest.mark.parametrize(
    "message",
    ["чьи авто", "чьи автомобили", "чьи машины", "кому принадлежат автомобили"],
)
def test_vehicle_owner_questions_explain_seller_ownership(message: str):
    response = ask(message, f"vehicle-owner-{abs(hash(message))}")

    assert response.intent == "platform"
    assert response.confidence_level == "high"
    assert "имущество продавцов площадки" in response.answer
    assert "MIGTORG" in response.answer
    assert "не является владельцем лотов" in response.answer


def test_every_canonical_phrase_keeps_its_assigned_article():
    canonical_phrases = _canonical_phrase_rules()
    failures = {}

    for phrase, expected_article_id in canonical_phrases.items():
        if has_ambiguous_phrase_rule(phrase):
            continue
        result = search_knowledge_match(
            phrase,
            classify_intent(phrase),
            "guest",
        )
        actual_article_id = result.article.slug if result.article else None
        if actual_article_id != expected_article_id:
            failures[phrase] = {
                "expected": expected_article_id,
                "actual": actual_article_id,
                "features": result.matched_features,
                "fallback_reason": result.fallback_reason,
            }

    assert failures == {}


def test_screenshot_lot_number_explains_where_to_find_it():
    response = ask("номер лота", "screenshot-lot-number")

    assert response.intent == "lot"
    assert response.confidence_level == "high"
    assert "в карточке лота" in response.answer.casefold()
    assert "личного кабинета" in response.answer.casefold()


def test_screenshot_top_sellers_is_clarified_instead_of_misrouted():
    response = ask("топ продавцов", "screenshot-top-sellers")

    assert response.action == "clarify"
    assert response.clarifying_options == [
        "Какие продавцы размещают лоты",
        "Программа ТОП-10 покупателей",
        "Другая тема",
    ]
    assert "связаться с продавцом нельзя" not in response.answer.casefold()


@pytest.mark.parametrize("message", ["ресо гасится", "Тинькофф молчит", "РЕСО не отвечает", "тинькоф игнорирует"])
def test_named_seller_silence_routes_to_contact_delay(message: str):
    response = ask(message, f"seller-silence-{abs(hash(message))}")

    assert response.intent == "transfer"
    assert response.confidence_level == "high"
    assert response.needs_ticket is True
    assert "ожидает подтверждения" in response.answer.casefold()
    assert "номер лота" in response.answer.casefold()
    assert "кредит предыдущего собственника" not in response.answer.casefold()


@pytest.mark.parametrize("message", ["документы где мои", "мои где документы"])
def test_reordered_accounting_documents_question_keeps_meaning(message: str):
    response = ask(message, f"reordered-docs-{abs(hash(message))}")

    assert response.intent == "payment"
    assert response.confidence_level == "high"
    assert response.needs_ticket is True
    assert "info@migtorg.com" in response.answer


@pytest.mark.parametrize("message", ["как участвовать", "участвовать как"])
def test_short_participation_question_returns_bidding_steps(message: str):
    response = ask(message, f"short-participation-{abs(hash(message))}")

    assert response.intent == "bidding"
    assert response.confidence_level == "high"
    assert "зарегистр" in response.answer.casefold()
    assert "ставк" in response.answer.casefold()


@pytest.mark.parametrize("message", ["где авто", "где машина"])
def test_short_vehicle_location_question_uses_lot_location_guidance(message: str):
    response = ask(message, f"short-vehicle-location-{abs(hash(message))}")

    assert response.intent == "lot"
    assert response.confidence_level == "high"
    assert "карточк" in response.answer.casefold()
    assert "мест" in response.answer.casefold() or "адрес" in response.answer.casefold()


def test_screenshot_parking_question_remains_supported():
    response = ask("где стоянка", "screenshot-parking")

    assert response.intent == "lot"
    assert response.confidence_level == "high"
    assert "место хранения" in response.answer.casefold()


@pytest.mark.parametrize(
    ("message", "expected_intent"),
    [
        ("хочу начать покупать машины с чего начать", "bidding"),
        ("как подключиться к аукциону", "bidding"),
        ("можно посмотреть тачку до торгов", "inspection"),
        ("куда ехать за выигранной машиной", "transfer"),
        ("страховщик игнорит после победы", "transfer"),
        ("договор так и не прислали", "transfer"),
        ("когда платить за выигранное авто", "payment"),
        ("торги закончились а статус старый", "lot"),
        ("где искать вин", "lot"),
        ("не пускают за машиной", "pickup"),
    ],
)
def test_first_line_paraphrase_eval_routes_safely(message: str, expected_intent: str):
    response = ask(message, f"first-line-eval-{abs(hash(message))}")

    assert response.intent == expected_intent
    assert response.confidence_level == "high"
    assert response.action != "clarify"
    assert response.intent != "prohibited"
