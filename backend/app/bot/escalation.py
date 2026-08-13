import re

from backend.app.bot.text_processing import TextAnalysis, analyze_text, has_specific_problem, normalize_text


TICKET_INTENTS = {"refund", "penalty", "refusal", "feedback"}
SPECIFIC_LOT_WORDS = ("адрес", "vin", "документ", "статус", "выдач", "стоянк", "передач")


def needs_ticket(
    message: str,
    intent: str,
    context: object | None = None,
    analysis: TextAnalysis | None = None,
) -> bool:
    analysis = analysis or analyze_text(message, context)
    text = analysis.corrected or normalize_text(message)
    general_vin_reference = any(
        phrase in text
        for phrase in (
            "сколько стоит опция проверки по vin",
            "сколько стоит проверить vin",
            "сколько стоит пробить машину по vin",
            "проверка vin платная",
            "проверка vin у вас платная",
            "цена запроса по vin",
            "стоимость проверки машины по vin",
            "услуга по vin",
            "кнопка запросить",
            "вместо vin написано запросить",
            "почему вместо vin",
            "запросить vin",
            "запросить вин",
        )
    )
    vin_problem_reference = any(
        phrase in text
        for phrase in (
            "не работает",
            "не открыл",
            "не открывается",
            "не отображ",
            "ошибка",
            "деньги спис",
            "списали",
            "результат не",
        )
    )
    if intent == "lot" and general_vin_reference and not vin_problem_reference:
        return False
    if intent == "lot" and general_vin_reference and vin_problem_reference:
        return True
    if intent == "tariffs" and any(
        phrase in text
        for phrase in (
            "обеспечительный платеж",
            "отдельно платить деньги за торги",
            "платить деньги за торги за лот",
        )
    ):
        return False
    if intent == "payment" and any(
        phrase in text
        for phrase in (
            "выставили такую комиссию",
            "откуда взялась такая сумма",
            "почему такая сумма",
            "не согласен с комиссией",
            "не согласна с комиссией",
        )
    ):
        return True
    if intent == "transfer" and any(
        phrase in text
        for phrase in (
            "ожидает подтверждения",
            "ожидание подтверждения",
            "лот передается",
            "лот передаётся",
            "смс что лот передается",
            "смс что лот передаётся",
            "контакт продавца",
            "контакты продавца",
            "связаться с продавцом",
            "как позвонить продавцу",
            "позвонить продавцу",
            "документы нужны для забора лота",
            "какие документы нужны для забора",
            "как получить дкп",
            "получить дкп",
            "как подписать договор",
            "подписать договор",
            "как попасть на стоянку",
            "контакты стоянки",
            "у кого просить контакты",
            "кто выдает лот",
            "кто выдаёт лот",
            "нужно ли ехать к вам в офис",
            "ехать к вам в офис",
            "доверенность на получение машины",
            "доверенность на получение автомобиля",
            "нотариальная доверенность",
            "другой чел забрать мой лот",
            "другой человек забрать мой лот",
            "реквизиты",
            "просят реквизиты",
            "что такое реквизиты",
        )
    ):
        return False
    if intent == "lot" and any(
        phrase in text
        for phrase in (
            "информацию об автомобиле",
            "информация об автомобиле",
            "ожидает подтверждения",
            "ожидание подтверждения",
            "где у вас стоянка",
            "где стоянка",
            "где находится стоянка",
            "адрес стоянки",
            "как узнать адрес стоянки",
            "где находится машина",
            "где находится автомобиль",
            "точный адрес нахождения машины",
            "точный адрес нахождения автомобиля",
            "почему нет птс",
            "нет птс",
            "скрываете информацию",
            "фильтр для поиска",
            "поиск по марке",
            "поиск по модели",
            "определенной марке",
            "определенной модели",
            "конкретная машина",
            "тс кредитное",
            "тс в лизинге",
            "тс в залоге",
            "нужно ли мне гасить кредит",
            "гасить кредит предыдущего",
        )
    ) and not analysis.entities["lot_id"]:
        return False
    if intent == "refund" and any(
        phrase in text
        for phrase in (
            "что такое возвратный депозит",
            "возвратный депозит это",
        )
    ):
        return False
    if intent == "support" and any(
        phrase in text
        for phrase in (
            "мне угрожает",
            "мне угрожают",
            "меня шантажируют",
            "продавец угрожает",
            "сотрудник угрожает",
            "меня обманул продавец",
            "меня обманули",
        )
    ):
        return True
    if intent == "prohibited":
        return False
    if has_specific_problem(message, intent, context, analysis=analysis):
        return True
    if any(phrase in text for phrase in ("мой автомобиль", "моя машина", "мою машину", "мое авто", "моё авто")) and any(
        phrase in text for phrase in ("торг", "лот", "площадк", "выстав")
    ):
        return True
    if intent in TICKET_INTENTS:
        return True
    if intent == "payment" and any(
        term in text for term in ("платеж", "оплата", "списание", "списали", "списались", "чек", "квитанция")
    ) and any(
        problem in text for problem in ("нет", "не видно", "не отображается", "завис", "не зачислилось")
    ):
        return True
    if analysis.entities["payment_id"]:
        return True
    if analysis.entities["lot_id"] and (
        intent in TICKET_INTENTS
        or any(
            word in text
            for word in (
                "адрес",
                "vin",
                "статус",
                "выдач",
                "стоянк",
                "передач",
                "не передают",
                "не отдают",
                "не выдают",
                "когда",
                "срок",
            )
        )
    ):
        return True
    if re.search(r"\b(лот\w*|платеж\w*|заявк[аи]|счет|vin)\s*#?\s*\d+", text):
        return True
    if intent == "lot" and any(word in text for word in SPECIFIC_LOT_WORDS):
        return True
    if any(
        phrase in text
        for phrase in (
            "платеж точно прошел",
            "платёж точно прошел",
            "подтвердить что мой платеж прошел",
            "подтвердить что мой платёж прошел",
            "платеж списался",
            "платёж списался",
            "оплата зависла",
            "нужен специалист",
            "нужен менеджер",
            "не открывается карточка",
            "карточка пустая",
            "еду в офис",
            "еду к вам",
        )
    ):
        return True
    if any(
        phrase in text
        for phrase in (
            "не появился доступ",
            "не отображается",
            "не пришли документы",
            "стоянка не выдает",
            "не соответствует описанию",
            "двойное списание",
            "списали деньги",
            "нужен сотрудник",
            "позовите оператора",
        )
    ):
        return True
    return False


def suggested_fields_for(intent: str, has_contact: bool = False) -> list[str]:
    fields: list[str] = []
    if not has_contact:
        fields.append("contact")
    if intent in {"lot", "transfer", "pickup", "inspection", "refusal", "penalty"}:
        fields.append("lot_id")
    if intent in {"payment", "refund"}:
        fields.extend(["payment_date", "payment_amount", "payment_method"])
    if intent in {"refusal", "penalty"}:
        fields.extend(["reason", "photos_or_documents"])
    return list(dict.fromkeys(fields))
