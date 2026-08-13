import re

from backend.app.models.safety import SafetyCheckResult
from backend.app.bot.text_processing import load_matching_config, normalize_text


PROHIBITED_PATTERNS: tuple[tuple[str, str], ...] = (
    ("обойти закон", "legal_risk"),
    ("обход закона", "legal_risk"),
    ("обойти санкции", "legal_risk"),
    ("обойти штраф", "legal_risk"),
    ("обход штрафа", "legal_risk"),
    ("чужие персональные", "personal_data_risk"),
    ("чужие данные", "personal_data_risk"),
    ("данные другого пользователя", "personal_data_risk"),
    ("персональные данные другого", "personal_data_risk"),
    ("чужую ставку", "personal_data_risk"),
    ("чужая ставка", "personal_data_risk"),
    ("телефон другого пользователя", "personal_data_risk"),
    ("контакты другого участника", "personal_data_risk"),
    ("наркот", "prohibited_topic"),
    ("оруж", "prohibited_topic"),
    ("террор", "prohibited_topic"),
    ("экстрем", "prohibited_topic"),
)

PROHIBITED_REGEX_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"\b(?:как\s+)?(?:подделать|подделывать|сфальсифицировать|нарисовать)\b", "document_fraud"),
    (r"\b(?:липовый|фейковый|поддельный)\s+(?:акт|чек|договор|платеж|документ)\b", "document_fraud"),
    (r"\b(?:указать|вписать|поставить)\s+(?:другую|неверную|меньшую)\s+сумм\w*\s+в\s+договор", "document_fraud"),
    (r"\b(?:как\s+)?(?:взломать|получить доступ к)\s+(?:аккаунт|кабинет|личн\w* кабинет|учетн\w+ запис\w*)\b", "account_intrusion"),
    (r"\b(?:как\s+)?(?:кинуть|обмануть|развести)\s+(?:продавца|покупателя|площадку|участника)\b", "fraud_risk"),
    (r"\b(?:дать|предложить|передать)\s+взятк\w*\b", "bribery_risk"),
    (r"\b(?:договориться за деньги|заплатить сотруднику в обход|занести сотруднику)\b", "bribery_risk"),
    (r"\b(?:я\s+)?(?:убью|изобью|зарежу|подожгу|сломаю)\b", "threat_risk"),
    (r"\bкак\s+(?:угрожать|шантажировать|надавить|заставить силой)\b", "threat_risk"),
    (r"\b(?:скрыть|спрятать|не указывать)\s+(?:дефект\w*|поврежден\w*|авари\w*)\b", "fraud_risk"),
    (r"\bкак\s+(?:не платить|уклониться от|обойти)\s+(?:штраф\w*|налог\w*|санкци\w*|ограничени\w*)\b", "legal_risk"),
    (r"\b(?:дайте|покажите|узнать|получить)\s+(?:телефон|контакты|email|почту)\s+(?:другого|чужого)\b", "personal_data_risk"),
)

ABUSIVE_LANGUAGE_PATTERNS: tuple[str, ...] = (
    r"(?<!\w)хуй\w*(?!\w)",
    r"(?<!\w)охуел\w*(?!\w)",
    r"(?<!\w)пизд\w*(?!\w)",
    r"(?<!\w)бля(?:ть|дь)?(?!\w)",
    r"(?<!\w)ебан\w*(?!\w)",
    r"(?<!\w)сука(?!\w)",
    r"(?<!\w)мудак\w*(?!\w)",
    r"(?<!\w)говно(?!\w)",
    r"\bидиот\w*\b",
    r"\bурод\w*\b",
    r"\bиздеваетесь\b",
    r"\bбезмозгл\w*\b",
)

REPORTING_MARKERS: tuple[str, ...] = (
    "мне угрож",
    "меня шантаж",
    "меня обман",
    "хочу пожаловаться",
    "подать жалоб",
    "куда жаловаться",
    "куда сообщить",
    "прошу проверить",
    "требует с меня",
    "намекает на",
    "намекнул на",
)

FORBIDDEN_ANSWER_PATTERNS: tuple[tuple[str, str], ...] = (
    ("точно вернем", "refund_promise"),
    ("гарантируем возврат", "refund_promise"),
    ("вам положен возврат", "refund_promise"),
    ("штраф отменят", "penalty_cancellation"),
    ("отменим штраф", "penalty_cancellation"),
    ("платеж прошел", "payment_confirmation_without_data"),
    ("платеж точно зачислят", "payment_confirmation_without_data"),
    ("у вас активный тариф", "business_forbidden_promise"),
    ("ваш лот передан", "lot_status_without_data"),
    ("точно передадут", "business_forbidden_promise"),
    ("гарантирую выигрыш", "business_forbidden_promise"),
    ("гарантируем выигрыш", "business_forbidden_promise"),
    ("вы точно заработаете", "business_forbidden_promise"),
)

FORBIDDEN_ANSWER_REGEX_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"\b(?:деньги|депозит)\s+(?:вам\s+)?(?:точно|гарантированно|обязательно)\s+вернут\b", "refund_promise"),
    (r"\b(?:штраф|санкци\w*)\s+(?:точно\s+)?(?:отменят|снимут|аннулируют)\b", "penalty_cancellation"),
    (r"\b(?:платеж|оплата)\s+(?:успешно\s+)?(?:прошел|подтвержден\w*|зачислен\w*)\b", "payment_confirmation_without_data"),
    (r"\b(?:тариф|доступ)\s+(?:уже\s+)?(?:активен|подключен|активирован)\b", "business_forbidden_promise"),
    (r"\b(?:лот|автомобиль|машина)\s+(?:уже\s+)?(?:ваш|передан|точно будет передан)\b", "lot_status_without_data"),
    (r"\b(?:вы\s+)?(?:точно|гарантированно)\s+(?:выиграете|заработаете|получите прибыль)\b", "business_forbidden_promise"),
)

REFUSAL_ANSWER = (
    "Я не могу помогать с действиями, которые могут нарушать закон, правила площадки "
    "или права других участников. Могу подсказать только законный порядок работы с лотом, "
    "оплатой, документами или обращением в поддержку."
)
PRIVACY_REFUSAL_ANSWER = (
    "Я не могу помогать с получением или предоставлением чужих контактов, ставок, данных аккаунта "
    "или других персональных сведений. "
    "Могу помочь с вашими данными или подготовить официальное обращение в поддержку."
)
THREAT_REFUSAL_ANSWER = (
    "Я не могу помогать с угрозами, давлением или причинением вреда. Опишите спорную ситуацию без угроз — "
    "я подскажу официальный порядок обращения в MIGTORG."
)
CALM_TONE_PREFIX = "Понимаю, что ситуация вызывает раздражение. Давайте разберемся по существу. "
NEGATED_CLAIM_MARKERS = (
    "не могу подтвердить",
    "не подтверждаю",
    "не буду подтверждать",
    "нельзя подтвердить",
    "нет данных что",
    "без проверки нельзя сказать",
)
CONDITIONAL_CLAIM_MARKERS = (
    "если",
    "когда",
    "после того как",
    "при условии что",
    "отображается какой",
    "отображаться какой",
)


def _configured_categories(text: str) -> set[str]:
    config = load_matching_config()
    category_map = {
        "document_fraud": "document_fraud",
        "penalty_evasion": "legal_risk",
        "hide_defects": "fraud_risk",
        "personal_data_risk": "personal_data_risk",
        "illegal_pressure": "threat_risk",
        "fraud": "fraud_risk",
        "bribery": "bribery_risk",
        "threats": "threat_risk",
    }
    categories: set[str] = set()
    safety_triggers = config.get("safety_triggers", {})
    if not isinstance(safety_triggers, dict):
        return categories
    for raw_category, phrases in safety_triggers.items():
        if not isinstance(phrases, list):
            continue
        category = category_map.get(str(raw_category), str(raw_category))
        if any(normalize_text(str(phrase)) in text for phrase in phrases):
            categories.add(category)
    return categories


def _contains_stem(text: str, stems: tuple[str, ...]) -> bool:
    return any(stem in text for stem in stems)


def _stems_are_near(
    text: str,
    left_stems: tuple[str, ...],
    right_stems: tuple[str, ...],
    *,
    max_token_distance: int,
) -> bool:
    tokens = text.split()
    left_positions = [
        index
        for index, token in enumerate(tokens)
        if any(token.startswith(stem) for stem in left_stems)
    ]
    right_positions = [
        index
        for index, token in enumerate(tokens)
        if any(token.startswith(stem) for stem in right_stems)
    ]
    return any(
        abs(left - right) <= max_token_distance
        for left in left_positions
        for right in right_positions
    )


def _is_report_of_violation(text: str) -> bool:
    return _contains_stem(text, REPORTING_MARKERS)


def _concept_categories(text: str) -> set[str]:
    categories: set[str] = set()
    reporting = _is_report_of_violation(text)

    if (
        _contains_stem(text, ("подобр", "перебр", "угад", "взлом", "получить доступ"))
        and _contains_stem(text, ("парол", "код", "аккаунт", "кабинет", "учетн", "профил"))
        and _contains_stem(text, ("чуж", "другого", "другой", "конкурент", "участник"))
    ):
        categories.add("account_intrusion")

    private_field_stems = ("телефон", "номер", "мобильн", "почт", "email", "адрес", "ставк")
    private_field = _contains_stem(text, private_field_stems)
    generic_contact = "контакт" in text
    explicit_third_party = _contains_stem(
        text,
        (
            "чуж",
            "другого",
            "другой",
            "участник",
            "победител",
            "покупател",
            "конкурент",
            "владел",
            "лидер",
        ),
    )
    described_third_party = (
        _stems_are_near(
            text,
            ("человек", "лиц"),
            ("выигр", "побед", "заня", "участв", "лидир", "ставк", "заявк", "купил", "приобрел"),
            max_token_distance=6,
        )
        and _stems_are_near(
            text,
            private_field_stems,
            ("человек", "лиц"),
            max_token_distance=6,
        )
    )
    own_private_field = _stems_are_near(
        text,
        private_field_stems,
        (
            "мой",
            "моя",
            "мое",
            "моег",
            "моем",
            "мою",
            "мои",
            "свой",
            "своя",
            "свое",
            "своег",
            "своем",
            "свою",
            "свои",
        ),
        max_token_distance=2,
    )
    official_business_contact = (
        "официальн" in text
        and _contains_stem(text, ("продавц", "организатор", "площадк", "поддержк", "migtorg"))
        and not _contains_stem(text, ("личн", "частн", "домашн"))
    )
    seller_private_field = "продавц" in text and private_field
    if (
        (private_field or generic_contact)
        and _contains_stem(
            text,
            (
                "дай",
                "дайте",
                "покаж",
                "узна",
                "получ",
                "достать",
                "найти",
                "сообщ",
                "пришл",
                "предостав",
                "подскаж",
                "скаж",
            ),
        )
        and (explicit_third_party or described_third_party or seller_private_field)
        and not own_private_field
        and not official_business_contact
    ):
        categories.add("personal_data_risk")

    if (
        _contains_stem(text, ("исправ", "измен", "подправ", "поддел", "нарис", "впис"))
        and _contains_stem(text, ("документ", "чек", "квитанц", "договор", "акт"))
        and _contains_stem(text, ("сумм", "цифр", "провер", "замет", "неверн", "меньш", "друг"))
    ):
        categories.add("document_fraud")

    if not reporting and (
        _contains_stem(text, ("взятк", "вознагражден", "отблагодар", "занест", "доплат"))
        and _contains_stem(text, ("сотрудник", "менеджер", "специалист", "администратор"))
        and _contains_stem(text, ("деньг", "наличн", "отдельно", "неофициальн", "ускор", "приоритет", "решен"))
    ):
        categories.add("bribery_risk")

    violent_intent = _contains_stem(
        text,
        ("разнес", "разгром", "подожг", "удар", "сломаю", "силой застав"),
    ) or bool(re.search(r"\bизоб(?:ью|ьешь|ьет|ьем|ьете|ьют|ить)\b", text))
    if not reporting and (
        violent_intent
        and _contains_stem(text, ("я ", "приед", "буду", "хочу", "как ", "офис", "сотрудник", "продавц", "вас"))
    ):
        categories.add("threat_risk")

    if not reporting and (
        _contains_stem(text, ("в обход", "мимо площад", "неофициальн", "сделать вид", "скрыть сделк"))
        and _contains_stem(text, ("оплат", "платеж", "сделк", "договор", "продавц", "площадк"))
    ):
        categories.add("fraud_risk")
    return categories


def _refusal_answer(categories: set[str]) -> str:
    if "threat_risk" in categories:
        return THREAT_REFUSAL_ANSWER
    if categories == {"personal_data_risk"}:
        return PRIVACY_REFUSAL_ANSWER
    return REFUSAL_ANSWER


def _is_negated_claim(text: str, start: int) -> bool:
    prefix = text[max(0, start - 80):start]
    local_prefix = prefix[-35:]
    return any(marker in prefix for marker in NEGATED_CLAIM_MARKERS) or any(
        marker in local_prefix for marker in CONDITIONAL_CLAIM_MARKERS
    )


def _contains_unnegated_phrase(text: str, phrase: str) -> bool:
    start = 0
    while True:
        index = text.find(phrase, start)
        if index < 0:
            return False
        if not _is_negated_claim(text, index):
            return True
        start = index + len(phrase)


def _matches_unnegated_regex(text: str, pattern: str) -> bool:
    return any(
        not _is_negated_claim(text, match.start())
        for match in re.finditer(pattern, text, flags=re.IGNORECASE)
    )


def pre_check(message: str) -> SafetyCheckResult:
    text = normalize_text(message)
    categories = {category for pattern, category in PROHIBITED_PATTERNS if normalize_text(pattern) in text}
    categories.update(
        category
        for pattern, category in PROHIBITED_REGEX_PATTERNS
        if re.search(pattern, text, flags=re.IGNORECASE)
    )
    categories.update(_configured_categories(text))
    categories.update(_concept_categories(text))
    if categories:
        sorted_categories = sorted(categories)
        return SafetyCheckResult(
            allowed=False,
            categories=sorted_categories,
            answer_override=_refusal_answer(categories),
            needs_review=True,
        )

    if any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in ABUSIVE_LANGUAGE_PATTERNS):
        return SafetyCheckResult(
            allowed=True,
            categories=["abusive_language"],
            answer_prefix=CALM_TONE_PREFIX,
            needs_review=False,
        )
    return SafetyCheckResult()


def post_check(answer: str) -> SafetyCheckResult:
    text = normalize_text(answer)
    categories = {
        category
        for pattern, category in FORBIDDEN_ANSWER_PATTERNS
        if _contains_unnegated_phrase(text, normalize_text(pattern))
    }
    categories.update(
        category
        for pattern, category in FORBIDDEN_ANSWER_REGEX_PATTERNS
        if _matches_unnegated_regex(text, pattern)
    )
    configured = load_matching_config().get("forbidden_phrases", [])
    if isinstance(configured, list) and any(
        _contains_unnegated_phrase(text, normalize_text(str(phrase)))
        for phrase in configured
    ):
        categories.add("business_forbidden_promise")
    if categories:
        return SafetyCheckResult(
            allowed=False,
            categories=sorted(categories),
            answer_override=(
                "По этому вопросу нужна проверка сотрудником MIGTORG. "
                "Я не буду подтверждать данные или обещать результат без системной проверки."
            ),
            needs_review=True,
        )
    return SafetyCheckResult()
