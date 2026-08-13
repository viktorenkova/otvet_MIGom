from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from backend.app.bot.text_processing import normalize_text


@dataclass(frozen=True)
class TopicRoute:
    intent: str
    score: int
    margin: int
    candidates: tuple[str, ...]
    evidence: tuple[str, ...]

    @property
    def decisive(self) -> bool:
        return self.intent != "unknown" and self.score >= 2 and self.margin >= 1

    @property
    def ambiguous(self) -> bool:
        return self.intent == "unknown" and self.score >= 2 and len(self.candidates) >= 2


TOPIC_GROUPS: dict[str, dict[str, tuple[str, ...]]] = {
    "platform": {
        "service": ("площадк", "сервис", "сайт", "migtorg", "мигторг"),
        "identity": ("зачем", "смысл", "что это", "как устро", "чем отлич"),
        "source_actor": ("кто размещ", "кто публику", "кто выстав", "компани", "организац", "продавц"),
        "source_origin": ("откуда", "появля", "берут", "источник предлож"),
        "market_model": ("магазин", "фиксирован", "ценник", "витрин", "прямая продаж"),
    },
    "registration": {
        "account": ("регистрац", "профил", "аккаунт", "кабинет", "учетн", "анкет"),
        "access": ("вход", "войти", "парол", "код", "подтвержд", "привязан"),
        "change_or_problem": ("не приш", "не приним", "восстанов", "смен", "измен", "замен", "не получ"),
    },
    "tariffs": {
        "plan": ("тариф", "подписк", "доступ", "обслужив"),
        "choice_or_period": ("период", "месяц", "окончан", "выбрат", "вариант", "разов", "премиум"),
        "activation_problem": ("не актив", "не включ", "не подключ", "нет доступ", "не появ"),
    },
    "payment": {
        "money_operation": ("платеж", "оплат", "спис", "банк", "карт", "перевод", "чек", "счет", "квитанц"),
        "status_or_error": ("не отображ", "отсутств", "отклон", "завис", "ошиб", "истори", "не зачис", "пропал"),
        "accounting": ("бухгалтер", "закрыва", "счет фактур", "акт свер"),
    },
    "bidding": {
        "auction": ("торг", "аукцион", "ставк", "предложен", "цен"),
        "bid_action": ("повыс", "отправ", "переб", "лидир", "участв", "срок", "момент", "шаг"),
    },
    "lot": {
        "asset": ("лот", "автомоб", "машин", "имущест", "позици", "карточк"),
        "information": ("описан", "характер", "фото", "vin", "информац", "данн", "состояни", "избран", "сохран"),
        "information_problem": ("противореч", "расход", "не совпад", "отсутств", "не указан"),
    },
    "inspection": {
        "asset": ("лот", "автомоб", "машин", "транспорт", "имущест"),
        "inspection_action": ("осмотр", "увид", "посмотр", "провер", "посет", "вживую", "показ", "знакомств", "место хран"),
        "inspection_stage": ("до став", "перед став", "перед предлож", "до торгов", "до участ", "перед участ", "на месте", "согласов", "запрос", "запис"),
    },
    "transfer": {
        "won_stage": ("побед", "выигр", "торг заверш", "аукцион заверш"),
        "receiving": ("передач", "получ", "забрат", "выдач"),
        "logistics": ("адрес", "время", "координат", "часы", "место выдач"),
        "deal_documents": ("документ", "договор", "дкп", "бумаг", "продавц"),
        "delay": ("не приш", "задерж", "жду", "не отвеч", "долго"),
    },
    "refusal": {
        "deal": ("побед", "выигр", "сделк", "покупк", "лот", "автомоб", "машин"),
        "refusal_action": ("отказ", "передум", "выйти", "не заверш", "не выкуп", "расторг"),
        "defect": ("несоответ", "не соответств", "не совпад", "дефект", "поврежд", "расхожд"),
    },
    "penalty": {
        "penalty": ("штраф", "санкц", "удержан", "задолж", "начислен"),
        "dispute_or_payment": ("оспор", "не соглас", "возраж", "оплат", "внести", "кнопк", "откуда"),
    },
    "refund": {
        "return_funds": ("возврат", "депозит", "обеспечитель", "деньги обратно", "обратн"),
        "request_or_status": ("заявлен", "заявк", "запрос", "реквизит", "не приш", "не поступ", "жду", "срок", "давн", "перечис"),
    },
    "support": {
        "technical_surface": (
            "страниц",
            "интерфейс",
            "браузер",
            "файл",
            "загруз",
            "экран",
            "приложен",
            "карточк",
            "мобильн",
            "телефон",
            "смартфон",
            "планшет",
            "айфон",
            "iphone",
            "android",
        ),
        "technical_problem": (
            "не работ",
            "не откры",
            "не дает",
            "не груз",
            "не загруж",
            "не прогруж",
            "не реаг",
            "перестал реаг",
            "завис",
            "пуст",
            "белый экран",
            "ошиб",
            "пропал",
        ),
        "support_request": ("поддержк", "помощ", "специалист", "сотрудник"),
    },
    "feedback": {
        "feedback_action": ("отзыв", "идея", "предложен", "улучш", "пожелан"),
        "product_target": ("команд", "разработ", "сервис", "интерфейс", "поиск", "сайт"),
    },
}

MIN_TOPIC_SCORES = {"inspection": 3}
TOPIC_GROUP_WEIGHTS: dict[str, dict[str, int]] = {
    "refusal": {"refusal_action": 2, "defect": 2},
    "penalty": {"penalty": 2},
    "refund": {"return_funds": 2},
    "feedback": {"feedback_action": 2},
}


def _marker_matches(text: str, tokens: tuple[str, ...], marker: str) -> bool:
    if " " in marker:
        return marker in text
    return any(token.startswith(marker) for token in tokens)


def _matched_groups(text: str, groups: dict[str, tuple[str, ...]]) -> tuple[str, ...]:
    tokens = tuple(text.split())
    return tuple(
        group_name
        for group_name, markers in groups.items()
        if any(_marker_matches(text, tokens, marker) for marker in markers)
    )


def _composite_evidence(intent: str, text: str, evidence: tuple[str, ...]) -> tuple[str, ...]:
    tokens = tuple(text.split())
    result = list(evidence)

    if intent == "platform" and "source_actor" not in result:
        asks_about_actor = "кто" in tokens
        has_publication_action = any(
            token.startswith(("размещ", "публик", "выстав"))
            for token in tokens
        )
        if asks_about_actor and has_publication_action:
            result.append("source_actor")

    if intent == "inspection" and "inspection_stage" not in result:
        has_inspection_context = {"asset", "inspection_action"}.issubset(result)
        has_temporal_marker = any(token in {"до", "перед"} for token in tokens)
        has_auction_marker = any(
            token.startswith(("став", "торг", "аукцион", "цен", "предлож", "участ"))
            for token in tokens
        )
        if has_inspection_context and has_temporal_marker and has_auction_marker:
            result.append("inspection_stage")

    return tuple(result)


def route_topic(message: str, allowed_intents: Iterable[str] | None = None) -> TopicRoute:
    text = normalize_text(message)
    allowed = set(allowed_intents or TOPIC_GROUPS)
    ranked: list[tuple[int, str, tuple[str, ...]]] = []
    for intent, groups in TOPIC_GROUPS.items():
        if intent not in allowed:
            continue
        evidence = _composite_evidence(intent, text, _matched_groups(text, groups))
        weights = TOPIC_GROUP_WEIGHTS.get(intent, {})
        score = sum(weights.get(group, 1) for group in evidence)
        if len(evidence) >= 2 and score >= MIN_TOPIC_SCORES.get(intent, 2):
            ranked.append((score, intent, evidence))
    ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
    if not ranked:
        return TopicRoute("unknown", 0, 0, (), ())

    best_score, best_intent, best_evidence = ranked[0]
    second_score = ranked[1][0] if len(ranked) > 1 else 0
    tied_intents = tuple(intent for score, intent, _ in ranked if score == best_score and score >= 2)
    if len(tied_intents) > 1:
        return TopicRoute("unknown", best_score, 0, tied_intents[:3], best_evidence)
    return TopicRoute(
        intent=best_intent,
        score=best_score,
        margin=best_score - second_score,
        candidates=(best_intent,),
        evidence=best_evidence,
    )
