from dataclasses import dataclass
import re

from backend.app.config import Settings
from backend.app.bot.knowledge_search import KnowledgeArticle, load_fallbacks
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


def _redact_for_llm(text: str) -> str:
    redacted = re.sub(
        r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}",
        "[email]",
        text,
    )
    redacted = re.sub(
        r"(?:\+7|8)[\s\-()]*\d{3}[\s\-()]*\d{3}[\s\-]*\d{2}[\s\-]*\d{2}",
        "[телефон]",
        redacted,
    )
    redacted = re.sub(
        r"(?i)(?<![A-Z0-9])[A-HJ-NPR-Z0-9]{17}(?![A-Z0-9])",
        "[VIN]",
        redacted,
    )
    return re.sub(r"\b\d{7,}\b", "[идентификатор]", redacted)


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
) -> GeneratedAnswer:
    base = _article_answer(article) or _fallback_answer(intent)
    message_lower = message.lower()
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

    if settings is None or not settings.llm_enabled:
        return GeneratedAnswer(answer=base)
    if needs_ticket:
        return GeneratedAnswer(answer=base)
    if llm_spend_usd >= settings.active_llm_budget_usd:
        return GeneratedAnswer(answer=base)

    prompt = (
        "Сформулируйте ответ пользователю на основе контекста ниже. "
        "Не добавляйте новых фактов и не решайте спорные вопросы.\n\n"
        f"Вопрос пользователя: {_redact_for_llm(message)}\n\n"
        f"Тема: {intent}\n"
        f"Роль пользователя: {role}\n\n"
        f"Сценарий базы знаний: {article.scenario if article else 'fallback'}\n"
        f"Контекст базы знаний:\n{base}"
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
    return GeneratedAnswer(answer=result.text if result.success else base, llm_result=result)
