import re

from backend.app.bot.text_processing import extract_entities
from backend.app.models.ticket import Ticket
from backend.app.models.user_context import UserContext, UserRole


TOPIC_BY_INTENT = {
    "payment": "Проверка платежа",
    "refund": "Возврат денег",
    "penalty": "Спор по штрафу",
    "refusal": "Отказ от лота",
    "pickup": "Проблема при получении автомобиля",
    "lot": "Вопрос по конкретному лоту",
    "support": "Обращение в поддержку",
    "tariffs": "Проблема с тарифом",
}


def _extract_lot_id(message: str, context: UserContext | None) -> str | None:
    entities = extract_entities(message, context)
    if entities["lot_id"]:
        return entities["lot_id"][0]
    if context and context.lot_id:
        return context.lot_id
    match = re.search(r"(?:лот[ауе]?|lot)\s*#?\s*(\d+)", message.lower())
    return match.group(1) if match else None


def _extract_payment_id(message: str) -> str | None:
    entities = extract_entities(message)
    if entities["payment_id"]:
        return entities["payment_id"][0]
    match = re.search(r"(?:платеж[ау]?|payment|счет)\s*#?\s*(\d+)", message.lower())
    return match.group(1) if match else None


def build_ticket(
    message: str,
    intent: str,
    role: UserRole,
    context: UserContext,
    contact: str | None,
    dialog_history: list[dict],
    attachments: list[str] | None = None,
) -> Ticket:
    lot_id = _extract_lot_id(message, context)
    resolved_contact = contact or context.user_email or context.user_phone
    return Ticket(
        topic=TOPIC_BY_INTENT.get(intent, "Обращение в поддержку"),
        description=message,
        contact=resolved_contact,
        role=role,
        user_id=context.user_id,
        lot_id=lot_id,
        payment_id=_extract_payment_id(message),
        session_id=context.session_id or "",
        page_type=context.page_type,
        dialog_history=dialog_history,
        attachments=attachments or [],
    )
