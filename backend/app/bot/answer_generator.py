from dataclasses import asdict, dataclass
import re

from backend.app.bot.runtime_templates import runtime_templates, legacy_template
from backend.app.config import Settings
from backend.app.bot.answer_contracts import fact_context, get_answer_contract, verify_answer
from backend.app.bot.knowledge_search import KnowledgeArticle, load_fallbacks
from backend.app.bot.pii_redaction import redact_for_external_llm
from backend.app.integrations.llm_provider import build_llm_provider
from backend.app.models.llm import LLMRequest, LLMResult
from backend.app.models.user_context import UserRole


DEFAULT_ANSWERS: dict[str, str] = runtime_templates()["defaults"]


@dataclass(frozen=True)
class GeneratedAnswer:
    answer: str
    llm_result: LLMResult | None = None
    used_fact_ids: tuple[str, ...] = ()
    verification_passed: bool = True
    verification_reason: str = ""
    llm_candidate: str = ""
    document_policy: str = "keep"


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
    from backend.app.bot.scenario_policy import article_allowed
    if article and not article_allowed(article, role):
        return GeneratedAnswer(answer="Для этого раздела необходимо подтвердить вход в MIGTORG.",
                               used_fact_ids=(), verification_passed=False,
                               verification_reason="policy:scenario_access_denied")
    from backend.app.bot.knowledge_gaps import matching_gap
    gap = matching_gap(message, article.scenario) if article and article.scenario else None
    if gap:
        return GeneratedAnswer(answer=gap["safe_answer"], used_fact_ids=(),
                               verification_passed=True,
                               verification_reason="knowledge_gap:" + gap["gap_id"])
    contract = get_answer_contract(article.scenario) if article and article.scenario else None
    if settings and settings.answer_assembly_enabled and contract:
        from backend.app.bot.answer_assembly import build_answer_plan, verify_plan_text
        from backend.app.bot.architecture_decision import decision_context
        try:
            plan = build_answer_plan(message, article.scenario, role)
            expected = build_answer_plan(message, article.scenario, role)
        except (OSError, ValueError, KeyError, TypeError):
            plan = expected = None
        if plan is not None and expected is not None and verify_plan_text(plan.text, plan, expected):
            trace = decision_context.get()
            if trace is not None:
                trace["answer_plan"] = asdict(plan)
                trace["answer_plan"]["verification"] = "exact_published_fragments"
            return GeneratedAnswer(answer=plan.text, used_fact_ids=plan.required_fact_ids,
                verification_passed=True, verification_reason="assembly:" + plan.profile,
                document_policy=plan.documents)
        trace = decision_context.get()
        if trace is not None:
            trace["answer_assembly_error"] = "source_unavailable"
            trace["service_text"] = "assembly.source_unavailable"
        return GeneratedAnswer(answer="Не удалось проверить источники ответа. Уточните вопрос или создайте письменное обращение.",
                               verification_passed=False, verification_reason="assembly:source_unavailable", document_policy="omit")
    base = contract.approved_template if contract else (_article_answer(article) or _fallback_answer(intent))
    message_lower = message.lower()
    if article and article.scenario == "tariff.connect" and "премиум" in message_lower:
        base = (
            legacy_template("legacy_exception_01")
        )
    elif article and article.scenario == "refund.application" and any(
        word in message_lower for word in ("шаблон", "форма", "образец")
    ):
        base = (
            legacy_template("legacy_exception_02")
        )
    elif article and article.scenario == "bid.price_terms" and re.fullmatch(
        '\\s*(?:(?:что\\s+(?:такое|значит)\\s+)?ставка|ставка\\s+(?:это|что))\\s*[?!.]*\\s*',
        message_lower,
    ):
        base = (
            legacy_template("legacy_exception_04")
        )
    elif article and article.scenario in {"support.contact", "support.callback"} and _is_employee_connection_request(message):
        base = (
            legacy_template("legacy_exception_05")
        )
    base_lower = base.lower()
    if (
        intent in {"bidding", "lot", "transfer"}
        and ("точно передад" in message_lower or "гарант" in message_lower)
        and "не могу" not in base_lower
    ):
        base = (
            legacy_template("legacy_exception_06")
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
            legacy_template("legacy_exception_07")
        )
        base_lower = base.lower()
    if (
        intent == "tariffs"
        and any(phrase in message_lower for phrase in ("какой тариф выбрать", "какой тариф мне", "какой тариф подойдет", "какой тариф подходит"))
        and not (article and article.scenario == "tariff_selection_general")
    ):
        base = (
            legacy_template("legacy_exception_08")
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
        base += legacy_template("legacy_exception_09")
        base_lower = base.lower()
    if (
        intent in {"bidding", "lot", "transfer"}
        and "гарант" in message_lower
        and "не гарант" not in base_lower
    ):
        base = (
            legacy_template("legacy_exception_10")
            + base
        )
        base_lower = base.lower()
    if (
        intent == "lot"
        and needs_ticket
        and any(word in message_lower for word in ("адрес", "vin", "статус", "документ"))
        and "не подтвержда" not in base_lower
    ):
        base += legacy_template("legacy_exception_11")
        base_lower = base.lower()
    base = _finalize_answer(base, message)

    # Validate Python exceptions too; unapproved text cannot approve itself.
    if contract:
        base = verify_answer(base, contract.approved_template, contract).answer

    if settings is None or not settings.llm_enabled or settings.architecture_experiment:
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
            verification_passed=verification.passed,
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
    from backend.app.bot.architecture_decision import decision_context
    from backend.app.bot.processing_budget import remaining
    from backend.app.integrations.llm_provider import estimate_cost
    ctx = decision_context.get()
    if ctx:
        timeout = min(4.0, remaining()-0.2)
        if timeout <= 0:
            return GeneratedAnswer(base, verification_reason="llm_deadline_exhausted")
        # Reserve before a concurrent request can consume the same remaining budget.
        amount = estimate_cost(settings, len(prompt.encode("utf-8"))+2048, settings.llm_max_output_tokens)
        if amount > 0:
            reservation = ctx["logger"].reserve_llm_budget(amount, settings.llm_daily_budget_usd, settings.active_llm_monthly_budget_usd)
            if not reservation:
                return GeneratedAnswer(base, verification_reason="llm_budget_reserved_elsewhere")
            ctx["answer_budget_reservation"] = reservation
        elif settings.llm_provider != "mock":
            return GeneratedAnswer(base, verification_reason="llm_pricing_unavailable")
        settings = settings.model_copy(update={"llm_request_timeout_seconds": timeout,
            "llm_total_timeout_seconds": timeout, "llm_fallback_model": None})
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
