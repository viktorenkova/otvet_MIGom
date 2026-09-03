from dataclasses import dataclass
import re

from backend.app.config import Settings
from backend.app.bot.answer_contracts import fact_context, get_answer_contract, verify_answer
from backend.app.bot.knowledge_search import KnowledgeArticle, load_fallbacks
from backend.app.bot.pii_redaction import redact_for_external_llm
from backend.app.integrations.llm_provider import build_llm_provider
from backend.app.models.llm import LLMRequest, LLMResult
from backend.app.models.user_context import UserRole


DEFAULT_ANSWERS: dict[str, str] = {
    "platform": (
        "MIGTORG — онлайн-аукционная площадка: продавцы размещают лоты, а покупатели делают ставки. "
        "Площадка организует торги, фиксирует ставки и сопровождает процесс, но не является владельцем лота "
        "и не принимает решение о продаже вместо продавца."
    ),
    "registration": (
        "Для регистрации откройте форму на сайте MIGTORG, укажите контактные данные "
        "и следуйте подсказкам. Если код или письмо не приходят, проверьте правильность "
        "телефона или email и папку спама."
    ),
    "tariffs": (
        "Для участия в торгах нужен активный тариф. Тариф дает доступ к торгам, "
        "а баланс используется для денежных операций в личном кабинете. Если тариф "
        "оплачен, но доступ не появился, нужно создать обращение для проверки."
    ),
    "bidding": (
        "Для участия в торгах обычно нужно зарегистрироваться, подключить активный тариф "
        "и выбрать лот для ставки. Победа в торгах не гарантирует автоматическую передачу: "
        "продавец должен согласовать результат."
    ),
    "lot": (
        "По общим вопросам о лоте можно ориентироваться на карточку лота: там обычно "
        "указаны описание, фото, документы и условия. Конкретный адрес, VIN или статус "
        "я не подтверждаю без интеграции с системой."
    ),
    "payment": (
        "Платеж нужно проверять по данным операции. Без доступа к платежной системе я не "
        "могу подтвердить зачисление, но помогу создать обращение для сотрудника."
    ),
    "inspection": (
        "Осмотр помогает проверить состояние автомобиля до дальнейших действий. Если осмотр "
        "выявил несоответствие, сохраните фото, акт осмотра и описание проблемы."
    ),
    "transfer": (
        "После победы в торгах продавец должен согласовать результат. Победа не означает автоматическую передачу лота. "
        "Если лот передан, дальше идут практические шаги: реквизиты, документы, счет, оплата и согласование получения. "
        "Документы готовятся по конкретной сделке после передачи. До уведомления о передаче не нужно самостоятельно оплачивать автомобиль."
    ),
    "pickup": (
        "Порядок получения автомобиля зависит от данных конкретного лота и правил стоянки. "
        "Если есть проблема на выдаче, нужно создать обращение с номером лота и описанием ситуации."
    ),
    "refusal": (
        "Если лот уже передан, отказ без подтвержденной причины может повлечь штрафные последствия. "
        "Если есть основания для отказа, подготовьте фото, акт осмотра или документы для проверки."
    ),
    "penalty": (
        "Штрафы и спорные ситуации проверяются сотрудниками. Я могу объяснить общий порядок, "
        "но не могу отменить штраф или признать отказ обоснованным без проверки."
    ),
    "refund": (
        "Возврат всегда требует проверки условий и конкретной ситуации. Я не могу обещать возврат, "
        "но помогу создать обращение для сотрудника."
    ),
    "support": "Я помогу создать обращение для сотрудников MIGTORG и передать описание ситуации.",
    "unknown": (
        "Не нашел точный ответ. Напишите проще — например: “лот”, “оплата”, “штраф”, “тариф”. "
        "Или выберите частую ситуацию ниже."
    ),
}


@dataclass(frozen=True)
class GeneratedAnswer:
    answer: str
    llm_result: LLMResult | None = None
    used_fact_ids: tuple[str, ...] = ()
    verification_passed: bool = True
    verification_reason: str = ""
    llm_candidate: str = ""


def _redact_for_llm(text: str) -> str:
    redacted = redact_for_external_llm(text)
    if _is_employee_connection_request(text):
        redacted = _redact_requested_employee_name(redacted, "[имя сотрудника]")
    return redacted


def _is_employee_connection_request(text: str) -> bool:
    normalized = text.casefold().replace("ё", "е")
    if re.search(r"\b(?:продавц\w*|страхов\w*)\b", normalized):
        return False
    asks_to_connect = bool(
        re.search(
            r"\b(?:соедин\w*|переключ\w*|позов\w*|приглас\w*|"
            r"дай(?:те)?|позвон\w*|связ\w*)\b",
            normalized,
        )
    )
    mentions_staff = bool(
        re.search(r"\b(?:сотрудник\w*|менеджер\w*|оператор\w*|специалист\w*)\b", normalized)
    )
    names_person = bool(
        re.search(r"\b(?:с|со)\s+(?!вами\b|мной\b|нами\b)[а-я]{3,}\b", normalized)
        or re.search(r"\b(?:свяж\w*|позвон\w*|перезвон\w*|ответ\w*|позов\w*|дай(?:те)?)\s+[а-я]{3,}\b", normalized)
        or re.search(r"\b(?:сотрудник\w*|менеджер\w*|оператор\w*|специалист\w*)\s+[а-я]{3,}\b", normalized)
    )
    return asks_to_connect and (mentions_staff or names_person)


def _requested_employee_name_roots(message: str) -> list[str]:
    if not _is_employee_connection_request(message):
        return []
    generic_words = {
        "вами",
        "мной",
        "нами",
        "сотрудником",
        "сотрудницей",
        "менеджером",
        "оператором",
        "специалистом",
        "поддержкой",
    }
    roots: list[str] = []
    name_patterns = (
        r"\b(?:с|со)\s+([А-ЯЁа-яё]{3,})",
        r"\b(?:свяж\w*|позвон\w*|перезвон\w*|ответ\w*|позов\w*|дай(?:те)?)\s+([А-ЯЁа-яё]{3,})",
        r"\b(?:сотрудник\w*|менеджер\w*|оператор\w*|специалист\w*)\s+([А-ЯЁа-яё]{3,})",
    )
    matches = (match for pattern in name_patterns for match in re.finditer(pattern, message, flags=re.IGNORECASE))
    for match in matches:
        word = match.group(1).casefold().replace("ё", "е")
        if word in generic_words:
            continue
        for suffix in ("иями", "ями", "ами", "ому", "ему", "ого", "его", "ой", "ей", "ом", "ем", "ам", "ям", "ах", "ях", "у", "ю", "а", "я", "е", "и", "ы"):
            if word.endswith(suffix) and len(word) - len(suffix) >= 3:
                word = word[: -len(suffix)]
                break
        roots.append(word)
    return list(dict.fromkeys(roots))


def _redact_requested_employee_name(text: str, replacement: str = "конкретным сотрудником") -> str:
    result = text
    for root in _requested_employee_name_roots(text):
        result = re.sub(rf"\b{re.escape(root)}[а-яё]*\b", replacement, result, flags=re.IGNORECASE)
    return result


def _redact_employee_names_from_answer(answer: str, message: str) -> str:
    result = answer
    for root in _requested_employee_name_roots(message):
        result = re.sub(rf"\b{re.escape(root)}[а-яё]*\b", "конкретного сотрудника", result, flags=re.IGNORECASE)
    return result


def _echoes_requested_employee_name(answer: str, message: str) -> bool:
    return any(
        re.search(rf"\b{re.escape(root)}[а-яё]*\b", answer, flags=re.IGNORECASE)
        for root in _requested_employee_name_roots(message)
    )


def _deduplicate_answer(text: str) -> str:
    parts = re.split(r"(?<=[.!?])\s+|[\r\n]+", re.sub(r"\s+", " ", text).strip())
    seen: set[str] = set()
    kept: list[str] = []
    for part in parts:
        sentence = part.strip()
        if not sentence:
            continue
        key = re.sub(r"[^a-zа-яё0-9]+", "", sentence.casefold())
        if key in seen:
            continue
        seen.add(key)
        kept.append(sentence)
    return " ".join(kept)


def _finalize_answer(text: str, message: str) -> str:
    return _deduplicate_answer(_redact_employee_names_from_answer(text, message))


def _article_answer(article: KnowledgeArticle | None) -> str | None:
    if article is None:
        return None
    if article.user_answer:
        return article.user_answer
    marker = "## Ответ"
    if marker not in article.content:
        return None
    answer = article.content.split(marker, 1)[1].strip()
    if "## " in answer:
        answer = answer.split("## ", 1)[0].strip()
    return answer or None


def _fallback_answer(intent: str) -> str:
    fallbacks = load_fallbacks()
    return fallbacks.get(intent) or DEFAULT_ANSWERS.get(intent, DEFAULT_ANSWERS["unknown"])


def _is_owner_vehicle_question(message_lower: str) -> bool:
    return any(phrase in message_lower for phrase in ("мой автомобиль", "моя машина", "мою машину", "мое авто", "моё авто")) and any(
        phrase in message_lower for phrase in ("торг", "лот", "площадк", "выстав")
    )


def generate_answer(
    message: str,
    intent: str,
    role: UserRole,
    article: KnowledgeArticle | None,
    needs_ticket: bool,
    ticket_id: str | None = None,
    suggested_fields: list[str] | None = None,
    settings: Settings | None = None,
    session_id: str = "",
    safety_flags: list[str] | None = None,
    llm_spend_usd: float = 0.0,
    llm_daily_spend_usd: float = 0.0,
    llm_monthly_spend_usd: float | None = None,
    route_confidence: str = "high",
    llm_allowed: bool = True,
) -> GeneratedAnswer:
    contract = get_answer_contract(article.scenario) if article and article.scenario else None
    base = contract.approved_template if contract else (_article_answer(article) or _fallback_answer(intent))
    message_lower = message.lower()
    if article and article.scenario == "tariff.connect" and "премиум" in message_lower:
        base = (
            "Войдите в личный кабинет, откройте раздел «Тарифы», выберите «Премиум» и завершите оплату и активацию. "
            "Одного пополнения баланса недостаточно — тариф нужно выбрать отдельно. "
            "После оплаты проверьте статус тарифа; если доступ не появился, создайте обращение по платежу."
        )
    elif article and article.scenario == "refund.application" and any(
        word in message_lower for word in ("шаблон", "форма", "образец")
    ):
        base = (
            "Шаблон заявления на возврат депозита приложен ниже. "
            "Укажите номер и дату договора, сумму, данные и банковские реквизиты получателя, подпишите заявление и направьте его на info@migtorg.com. "
            "Не указывайте полный номер карты, CVC/CVV, пароль или код из SMS."
        )
    elif article and article.scenario == "bid.price_terms" and re.fullmatch(
        r"\s*(?:(?:что\s+(?:такое|значит)\s+)?ставка|ставка\s+(?:это|что))\s*[?!.]*\s*",
        message_lower,
    ):
        base = (
            "Ставка — это ценовое предложение участника купить конкретный лот за указанную сумму. "
            "Подтверждайте её только после проверки карточки и условий: победная ставка означает готовность к покупке, если продавец подтвердит передачу лота."
        )
    elif article and article.scenario in {"support.contact", "support.callback"} and _is_employee_connection_request(message):
        base = (
            "Поддержка работает по переписке; я не соединяю пользователей напрямую с конкретными сотрудниками по имени. "
            "Создайте письменное обращение: кратко опишите вопрос и добавьте номер лота или платежа, если он относится к ситуации. "
            "Ответ придёт по указанному вами официальному контакту."
        )
    base_lower = base.lower()
    if (
        intent in {"bidding", "lot", "transfer"}
        and ("точно передад" in message_lower or "гарант" in message_lower)
        and "не могу" not in base_lower
    ):
        base = (
            "Я не могу обещать или подтверждать передачу лота без проверки по системе: финальное решение принимает продавец. "
            + base
        )
        base_lower = base.lower()
    if (
        intent == "lot"
        and _is_owner_vehicle_question(message_lower)
        and not (
            article
            and article.scenario in {"owner_vehicle_on_auction", "insurer_owner_vehicle_listing"}
        )
    ):
        base = (
            "Понимаю ваше беспокойство. MIGTORG является площадкой, где продавцы размещают лоты для торгов, "
            "но не становится владельцем автомобиля и не принимает решение о размещении вместо продавца. "
            "Если вопрос касается именно вашего автомобиля, основания размещения или персональных данных, "
            "лучше направить обращение с номером лота и контактами для проверки."
        )
        base_lower = base.lower()
    if (
        intent == "tariffs"
        and any(phrase in message_lower for phrase in ("какой тариф выбрать", "какой тариф мне", "какой тариф подойдет", "какой тариф подходит"))
        and not (article and article.scenario == "tariff_selection_general")
    ):
        base = (
            "Если нужна одна покупка или вы хотите сначала попробовать площадку, обычно рассматривают Разовый тариф. "
            "Если планируете регулярно участвовать в торгах, анализировать много лотов и работать на перепродажу, чаще подходит Премиум. "
            + base
        )
        base_lower = base.lower()
    if (
        intent == "payment"
        and needs_ticket
        and not (article and article.scenario == "accounting_documents")
        and "не могу подтверд" not in base_lower
        and "не подтвержда" not in base_lower
    ):
        base += " Без проверки в системе я не могу подтвердить зачисление или статус платежа."
        base_lower = base.lower()
    if (
        intent in {"bidding", "lot", "transfer"}
        and "гарант" in message_lower
        and "не гарант" not in base_lower
    ):
        base = (
            "Победа в торгах не гарантирует передачу лота: после торгов продавец должен "
            "согласовать дальнейшие действия по конкретной сделке. "
            + base
        )
        base_lower = base.lower()
    if (
        intent == "lot"
        and needs_ticket
        and any(word in message_lower for word in ("адрес", "vin", "статус", "документ"))
        and "не подтвержда" not in base_lower
    ):
        base += " По конкретному лоту я не подтверждаю адрес, VIN, документы или статус без проверки в системе."
        base_lower = base.lower()
    if (
        role == "authorized"
        and intent in {"payment", "tariffs", "lot", "transfer", "pickup", "refund", "penalty"}
        and not (
            article
            and article.slug
            in {
                "kb-014-демо-режим-после-регистрации",
                "manual-review-2026-07-11-q-015-что-дает-демо-режим",
                "manual-review-2026-07-11-q-016-можно-ли-делать-ставки-в-демо-режиме",
                "manual-review-2026-07-11-q-017-можно-ли-видеть-результаты-торгов-в-демо-режиме",
                "manual-review-2026-07-11-q-021-как-посмотреть-цену-за-лот-если-у-меня-демо-режим",
                "manual-review-2026-07-11-q-027-пополнил-кошелек-но-тариф-не-включился-что-делать",
                "site-doc-028-demo-mode-upgrade",
            }
        )
    ):
        base += (
            " Вы можете также проверить соответствующий раздел личного кабинета, но без интеграции "
            "я не вижу ваши реальные данные."
        )
    if needs_ticket and ticket_id:
        base += f" Обращение создано. Номер: {ticket_id}."

    base = _finalize_answer(base, message)

    if settings is None or not settings.llm_enabled:
        verification = verify_answer(base, base, contract)
        return GeneratedAnswer(
            answer=verification.answer,
            used_fact_ids=verification.used_fact_ids,
            verification_passed=verification.passed,
            verification_reason=verification.reason,
        )
    if not llm_allowed or route_confidence != "high" or article is None or contract is None:
        verification = verify_answer(base, base, contract)
        reason = (
            "llm_ineligible:explicit"
            if not llm_allowed
            else f"llm_ineligible:confidence_{route_confidence}"
            if route_confidence != "high"
            else "llm_ineligible:no_article"
            if article is None
            else "llm_ineligible:no_contract"
        )
        return GeneratedAnswer(
            answer=base,
            used_fact_ids=verification.used_fact_ids,
            verification_passed=True,
            verification_reason=reason,
        )
    if needs_ticket:
        verification = verify_answer(base, base, contract)
        return GeneratedAnswer(
            answer=verification.answer,
            used_fact_ids=verification.used_fact_ids,
            verification_passed=verification.passed,
            verification_reason=verification.reason,
        )
    monthly_spend = llm_spend_usd if llm_monthly_spend_usd is None else llm_monthly_spend_usd
    if llm_daily_spend_usd >= settings.llm_daily_budget_usd:
        verification = verify_answer(base, base, contract)
        return GeneratedAnswer(
            answer=verification.answer,
            used_fact_ids=verification.used_fact_ids,
            verification_passed=verification.passed,
            verification_reason="llm_budget_daily_exhausted",
        )
    if monthly_spend >= settings.active_llm_monthly_budget_usd:
        verification = verify_answer(base, base, contract)
        return GeneratedAnswer(
            answer=verification.answer,
            used_fact_ids=verification.used_fact_ids,
            verification_passed=verification.passed,
            verification_reason="llm_budget_monthly_exhausted",
        )

    approved_facts = fact_context(contract)
    prompt = (
        "Дайте прямой ответ пользователю на основе контекста ниже. Используйте 2–4 коротких предложения. "
        "Оставьте только сведения, которые отвечают на заданный вопрос. Не повторяйте мысли и не добавляйте новых фактов. "
        "Не называйте и не повторяйте имена сотрудников, даже если имя было в вопросе.\n\n"
        f"Вопрос пользователя: {_redact_for_llm(message)}\n\n"
        f"Тема: {intent}\n"
        f"Роль пользователя: {role}\n\n"
        f"Сценарий базы знаний: {article.scenario if article else 'fallback'}\n"
        f"Утверждённый шаблон ответа:\n{contract.approved_template}\n\n"
        "Допустимые атомарные факты (ID нужны для проверки, не показывайте их пользователю):\n"
        f"{approved_facts}"
    )
    provider = build_llm_provider(settings)
    result = provider.generate(
        LLMRequest(
            prompt=prompt,
            fallback_text=base,
            provider=settings.llm_provider,
            model=settings.llm_primary_model,
            fallback_model=settings.llm_fallback_model,
            task_type="answer_generation",
            session_id=session_id,
            user_role=role,
            escalation_required=needs_ticket,
            safety_flags=safety_flags or [],
        )
    )
    candidate = (
        _finalize_answer(result.text, message)
        if result.success and not _echoes_requested_employee_name(result.text, message)
        else base
    )
    max_reasonable_length = min(1200, max(600, int(len(base) * 1.5)))
    if not candidate or len(candidate) > max_reasonable_length:
        candidate = base
    verification = verify_answer(candidate, base, contract)
    result.text = verification.answer
    result.verification_accepted = bool(result.success and verification.passed)
    result.verification_reason = verification.reason
    result.fallback_used = bool(not result.success or not verification.passed)
    return GeneratedAnswer(
        answer=verification.answer,
        llm_result=result,
        used_fact_ids=verification.used_fact_ids,
        verification_passed=verification.passed,
        verification_reason=verification.reason,
        llm_candidate=candidate,
    )
