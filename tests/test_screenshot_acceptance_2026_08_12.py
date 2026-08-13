from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import pytest

from backend.app.main import process_chat_message
from backend.app.models.chat import ChatRequest


@dataclass(frozen=True)
class ScreenshotCase:
    photo_number: int
    query: str
    scenario_id: str | None
    intent: str
    resolution: str
    required_text: tuple[str, ...] = ()
    required_options: tuple[str, ...] = ()


# Each photo existed in two identical dated copies (2026-08-07 and 2026-08-12).
# This table intentionally contains one case per unique user input, not per file.
SCREENSHOT_CASES = (
    ScreenshotCase(1, "где стоянка", "lot.location", "lot", "answered", ("место хранения",)),
    ScreenshotCase(2, "где авто", "lot.location", "lot", "answered", ("карточке лота", "адрес")),
    ScreenshotCase(3, "как участвовать", "buyer.get_started", "bidding", "answered", ("зарегистр", "ставк")),
    ScreenshotCase(4, "документы где мои", "payment.accounting_documents", "payment", "escalated", ("info@migtorg.com",)),
    ScreenshotCase(5, "тинкоф молчит", "documents.preparation_delay", "transfer", "escalated", ("номер лота", "продавец")),
    ScreenshotCase(6, "ресо гасится", "documents.preparation_delay", "transfer", "escalated", ("номер лота", "продавец")),
    ScreenshotCase(
        7,
        "топ продавцов",
        None,
        "unknown",
        "clarified",
        required_options=("Какие продавцы размещают лоты", "Программа ТОП-10 покупателей", "Другая тема"),
    ),
    ScreenshotCase(8, "номер лота", None, "lot", "answered", ("карточке лота", "личного кабинета")),
    ScreenshotCase(9, "как купить товар ?", "buyer.get_started", "bidding", "answered", ("зарегистр", "ставк")),
    ScreenshotCase(10, "как проверить статус торгов ?", "auction.status", "lot", "escalated", ("статус торгов", "номеру лота")),
    ScreenshotCase(11, "как подключиться продавцу ?", "seller.get_started", "registration", "answered", ("ролью продавца",)),
    ScreenshotCase(12, "как выставить лот на продажу ?", "seller.publish_lot", "platform", "answered", ("подключённый", "документ")),
    ScreenshotCase(13, "как вам позвонить ?", "support.contact", "support", "answered", ("по переписке", "письменное обращение")),
    ScreenshotCase(14, "пусть со мной свяжется Михаил", "support.callback", "support", "answered", ("по переписке", "по имени")),
    ScreenshotCase(
        15,
        "Регистрация и вход",
        None,
        "registration",
        "clarified",
        required_options=("Регистрация покупателя и необходимые данные", "Не получается войти в личный кабинет"),
    ),
    ScreenshotCase(16, "Пусть со мной свяжется Алексей", "support.callback", "support", "answered", ("по переписке", "по имени")),
    ScreenshotCase(
        18,
        "как подключиться",
        None,
        "unknown",
        "clarified",
        required_options=("Как начать покупать и участвовать в торгах", "Как подключиться и стать продавцом", "Как подключить тариф"),
    ),
    ScreenshotCase(19, "как участвовать в аукционе ?", "buyer.get_started", "bidding", "answered", ("зарегистр", "ставк")),
    ScreenshotCase(20, "перезвоните мне", "support.callback", "support", "answered", ("по переписке", "письменное обращение")),
    ScreenshotCase(21, "не видна моя ставка", "bid.not_visible", "bidding", "escalated", ("не отправляйте её повторно", "проверить ставку")),
    ScreenshotCase(22, "где можно забрать мой договор ?", "contract.receive", "transfer", "escalated", ("почтовую ветку", "номер лота")),
)


@pytest.mark.parametrize("case", SCREENSHOT_CASES, ids=lambda case: f"photo-{case.photo_number}")
def test_each_unique_screenshot_query(case: ScreenshotCase):
    response = process_chat_message(
        ChatRequest(message=case.query, session_id=f"screenshot-acceptance-{case.photo_number}-{uuid4()}")
    )

    assert response.scenario_id == case.scenario_id
    assert response.intent == case.intent
    assert response.resolution == case.resolution
    assert response.confidence_level in {"high", "medium"}
    answer = response.answer.casefold()
    for marker in case.required_text:
        assert marker.casefold() in answer
    for option in case.required_options:
        assert option in response.clarifying_options


def test_photo_17_ticket_topic_prefill_uses_editable_category_select():
    source = Path("frontend/chat-widget/widget.js").read_text(encoding="utf-8")
    static_page = Path("frontend/chat-widget/index.html").read_text(encoding="utf-8")

    for markup in (source, static_page):
        assert '<select name="topic">' in markup
        assert '<option value="Другая тема">Другая тема</option>' in markup
        assert 'name="custom_topic"' in markup
    assert 'topicSelect.value = ticketContext' in source
    assert 'topicSelect.value +=' not in source
    assert 'support: "Обращение в поддержку"' in source
    assert 'selectedTopic === "Другая тема" ? customTopic : selectedTopic' in source


def test_ticket_description_is_cleared_only_by_explicit_user_action():
    source = Path("frontend/chat-widget/widget.js").read_text(encoding="utf-8")

    assert 'class="ticket-form__clear-description" type="button"' in source
    assert 'clearDescriptionButton?.addEventListener("click"' in source
    assert 'descriptionInput.value = "";' in source
    assert 'descriptionInput.dataset.userEdited = "true";' in source
    assert 'descriptionInput && descriptionInput.dataset.userEdited !== "true"' in source
