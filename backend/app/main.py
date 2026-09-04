from contextlib import asynccontextmanager
import hashlib
import json
import re
import secrets
from time import perf_counter

from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from backend.app.bot.answer_generator import generate_answer
from backend.app.bot.dialog_logger import DialogLogger
from backend.app.bot.escalation import needs_ticket, suggested_fields_for
from backend.app.bot.intent_classifier import classify_intent
from backend.app.bot.knowledge_search import (
    KnowledgeSearchResult,
    get_article_by_id,
    has_ambiguous_phrase_rule,
    is_ticket_creation_request,
    search_knowledge_match,
    structured_route_name,
    warm_knowledge_indexes,
)
from backend.app.bot.role_resolver import resolve_role
from backend.app.bot.scenario_engine import extract_query_facets, find_scenario_action, get_scenario
from backend.app.bot.safety_guard import post_check, pre_check
from backend.app.bot.text_processing import analyze_text, best_intent_pattern, load_matching_config, tokenize
from backend.app.bot.topic_router import route_topic
from backend.app.bot.ticket_builder import build_ticket
from backend.app.config import get_settings
from backend.app.build_manifest import build_runtime_manifest
from backend.app.delivery.email_provider import EmailTicketProvider
from backend.app.delivery.local_provider import LocalDatabaseTicketProvider
from backend.app.integrations.langfuse_client import LangfuseClient
from backend.app.integrations.status_provider import build_status_provider
from backend.app.bot.trusted_context import TrustedContextError, verify_trusted_context_token
from backend.app.models.chat import ChatAction, ChatFeedbackRequest, ChatRequest, ChatResponse, LegacyChatRequest, LegacyChatResponse
from backend.app.models.safety import SafetyEvent
from backend.app.models.ticket import Ticket, TicketCreateRequest

settings = get_settings()
logger = DialogLogger(settings.database_path)
local_ticket_provider = LocalDatabaseTicketProvider(logger)
email_ticket_provider = EmailTicketProvider(settings)


def _in_llm_rollout(session_id: str, percentage: int) -> bool:
    if percentage <= 0:
        return False
    if percentage >= 100:
        return True
    bucket = int(hashlib.sha256(session_id.encode("utf-8")).hexdigest()[:8], 16) % 100
    return bucket < percentage


langfuse_client = LangfuseClient(settings)
status_provider = build_status_provider(settings)
MAX_CONTEXT_FOLLOWUPS = 2
REFUND_APPLICATION_TEMPLATE = {
    "label": "Шаблон_заявления_на_возврат_депозита.docx",
    "url": "/static/templates/Шаблон_заявления_на_возврат_депозита.docx",
}


@asynccontextmanager
async def lifespan(_: FastAPI):
    route_paths = ", ".join(str(getattr(route, "path", "")) for route in app.routes)
    print(f"MIGTORG_DEPLOY_VERSION={settings.deploy_version}")
    print(f"MIGTORG_APP_FILE={__file__}")
    print(f"MIGTORG_WIDGET_ROOT={settings.widget_root}")
    print(f"MIGTORG_ROUTES={route_paths}")
    warm_knowledge_indexes()
    yield


app = FastAPI(title="MIGTORG Chatbot MVP1", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=settings.static_root), name="static")
app.mount("/widget", StaticFiles(directory=settings.widget_root, html=True), name="widget")
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allowed_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def prevent_stale_widget_assets(request, call_next):
    response = await call_next(request)
    if request.url.path == "/widget" or request.url.path.startswith("/widget/"):
        response.headers["Cache-Control"] = "no-store, max-age=0"
    return response


@app.get("/", include_in_schema=False)
def demo_widget() -> RedirectResponse:
    return RedirectResponse(url="/widget/")


@app.get("/widget", include_in_schema=False)
def demo_widget_without_slash() -> RedirectResponse:
    return RedirectResponse(url="/widget/")


def _save_and_deliver_ticket(ticket: Ticket) -> Ticket:
    saved = local_ticket_provider.deliver(ticket)
    try:
        delivered = email_ticket_provider.deliver(saved)
    except Exception:
        logger.record_ticket_delivery_failure(saved.id)
        saved.status = "delivery_failed"
        return saved
    if delivered.status != saved.status:
        logger.update_ticket_status(delivered.id, delivered.status)
    return delivered


def _log_safety(
    session_id: str,
    user_id: str | None,
    message: str,
    categories: list[str],
    answer: str,
    needs_review: bool,
    ticket_created: bool,
) -> None:
    for category in categories:
        logger.log_safety_event(
            SafetyEvent(
                session_id=session_id,
                user_id=user_id,
                message=message,
                category=category,
                answer=answer,
                needs_review=needs_review,
                ticket_created=ticket_created,
            )
        )


def _persist_turn(
    *,
    started_at: float,
    request: ChatRequest,
    analysis,
    role: str,
    intent: str,
    answer: str,
    article_id: str | None,
    score: int,
    confidence: str,
    matched_features: list[str],
    action: str,
    needs_ticket: bool,
    ticket_id: str | None,
    ticket_created: bool,
    fallback_reason: str,
    safety_categories: list[str],
) -> str:
    facets = extract_query_facets(request.message)
    _, quality_event_id = logger.log_turn(
        session_id=request.session_id,
        role=role,
        original_message=request.message,
        normalized_message=analysis.normalized,
        corrected_message=analysis.corrected,
        detected_entities=analysis.entities,
        intent=intent,
        answer=answer,
        article_id=article_id,
        score=score,
        confidence=confidence,
        matched_features=matched_features,
        action=action,
        latency_ms=round((perf_counter() - started_at) * 1000),
        needs_ticket=needs_ticket,
        ticket_id=ticket_id,
        ticket_created=ticket_created,
        page_type=str(request.context.page_type or ""),
        fallback_reason=fallback_reason,
        safety_categories=safety_categories,
        scenario_id=(article_id if get_scenario(article_id or "") else None),
        resolution=(
            "out_of_scope"
            if fallback_reason == "out_of_scope"
            else "clarified"
            if action == "clarify"
            else "escalated"
            if needs_ticket
            else "status"
            if action == "fetch_status"
            else "answered"
        ),
        query_facets={
            "objects": sorted(facets.objects),
            "operations": sorted(facets.operations),
            "states": sorted(facets.states),
            "stage": facets.stage,
        },
    )
    return str(quality_event_id)


def _scenario_actions(scenario_id: str | None) -> list[ChatAction]:
    scenario = get_scenario(scenario_id or "")
    if not scenario:
        return []
    return [
        ChatAction(
            id=str(item.get("id") or ""),
            type=str(item.get("type") or "answer"),
            label=str(item.get("label") or ""),
            scenario_id=str(item.get("scenario_id") or scenario.scenario_id),
            payload=dict(item.get("payload", {})),
            requires_auth=bool(item.get("requires_auth", False)),
            requires_confirmation=bool(item.get("requires_confirmation", False)),
        )
        for item in scenario.actions
        if item.get("id") and item.get("label")
    ]


def _response_scenario_actions(message: str, scenario_id: str | None) -> list[ChatAction]:
    return _scenario_actions(scenario_id)


def _apply_trusted_context(request) -> str:
    token = str(request.trusted_context_token or "")
    if not token:
        request.context.trusted = False
        request.context.trusted_scopes = []
        return ""
    try:
        trusted = verify_trusted_context_token(
            token,
            settings.trusted_context_secret,
            issuer=settings.trusted_context_issuer,
        )
    except TrustedContextError as exc:
        request.context.trusted = False
        request.context.trusted_scopes = []
        return str(exc)
    request.context.trusted = True
    request.context.trusted_scopes = list(trusted.scopes)
    request.context.user_id = trusted.user_id
    request.context.user_email = trusted.email
    request.context.user_phone = trusted.phone
    request.context.is_authorized = True
    request.context.role = "authorized"
    return ""


def _clarification_candidate_intents(state: dict) -> set[str]:
    return {
        str(option.get("intent") or "unknown")
        for option in state.get("options", [])
        if str(option.get("intent") or "unknown") != "unknown"
    }


def _is_standalone_new_topic(message: str, state: dict) -> bool:
    if has_ambiguous_phrase_rule(message):
        return True

    current_intent = classify_intent(message)
    if structured_route_name(message, current_intent):
        return True

    route = route_topic(message)
    candidates = _clarification_candidate_intents(state)
    if route.decisive and candidates and route.intent not in candidates:
        return True
    return bool(
        current_intent != "unknown"
        and candidates
        and current_intent not in candidates
        and len(tokenize(message)) >= 5
    )


def _context_dict(request: ChatRequest) -> dict:
    return request.context.model_dump(mode="json", exclude_none=True)


def _restore_clarification_context(request: ChatRequest, state: dict | None) -> None:
    if not state:
        return
    saved_context = state.get("context", {})
    if not isinstance(saved_context, dict):
        return
    explicitly_sent = request.context.model_fields_set
    for field_name in request.context.__class__.model_fields:
        if field_name == "session_id" or field_name in explicitly_sent:
            continue
        if field_name in saved_context:
            setattr(request.context, field_name, saved_context[field_name])


def _process_chat_message(request: ChatRequest) -> ChatResponse:
    from backend.app.bot.architecture_decision import decision_context
    dialogue = (decision_context.get() or {}).get("dialogue")
    started_at = perf_counter()
    request.context.session_id = request.session_id
    clarification_state = logger.get_pending_clarification_state(request.session_id)
    if dialogue:
        # Structured state owns continuations; do not concatenate old turns again.
        clarification_state = None
    ticket_category_flow = bool(
        clarification_state
        and is_ticket_creation_request(str(clarification_state.get("original_message") or ""))
    )
    if dialogue and dialogue.get("category_choice"):
        ticket_category_flow = True
    _restore_clarification_context(request, clarification_state)
    trusted_context_error = _apply_trusted_context(request)
    role = "authorized" if request.context.trusted else "guest"
    analysis = analyze_text(request.message, request.context)

    safety_before = pre_check(request.message)
    if not safety_before.allowed:
        answer = safety_before.answer_override or ""
        safety_features = [f"safety:{category}" for category in safety_before.categories]
        _log_safety(
            request.session_id,
            request.context.user_id,
            request.message,
            safety_before.categories,
            answer,
            safety_before.needs_review,
            False,
        )
        message_id = _persist_turn(
            started_at=started_at,
            request=request,
            analysis=analysis,
            role=role,
            intent="safety",
            answer=answer,
            article_id=None,
            score=-100,
            confidence="high",
            matched_features=safety_features,
            action="safety_refusal",
            needs_ticket=False,
            ticket_id=None,
            ticket_created=False,
            fallback_reason="safety_conflict",
            safety_categories=safety_before.categories,
        )
        return ChatResponse(
            session_id=request.session_id,
            message_id=message_id,
            answer=answer,
            intent="safety",
            resolution="out_of_scope",
            role=role,
            needs_ticket=False,
            safety_categories=safety_before.categories,
            safety_flags=safety_before.categories,
            model_used=settings.llm_primary_model if settings.llm_enabled else "mock",
            confidence_level="high",
            action="safety_refusal",
        )

    selected_scenario_article = None
    selected_action = find_scenario_action(str(request.selected_action_id or ""))
    if selected_action:
        action_scenario, action_config = selected_action
        action_type = str(action_config.get("type") or "answer")
        target_scenario_id = str(action_config.get("scenario_id") or action_scenario.scenario_id)
        if action_type == "clarify":
            selected_scenario_article = get_article_by_id(target_scenario_id, role)
        elif action_type == "fetch_status":
            kind = str(action_config.get("payload", {}).get("kind") or "lot")
            required_scope = f"status:{kind}:read"
            has_scope = "status:read" in request.context.trusted_scopes or required_scope in request.context.trusted_scopes
            reference_id = str(request.context.user_id or "") if kind == "tariff" else str(request.context.lot_id or "")
            if not request.context.trusted or trusted_context_error or not has_scope:
                answer = (
                    "Для проверки персонального статуса войдите в MIGTORG заново. "
                    "Данные из браузера без защищённого подтверждения я не использую."
                )
                response_actions = [
                    item for item in _scenario_actions(action_scenario.scenario_id)
                    if item.type == "open_ticket"
                ]
                message_id = _persist_turn(
                    started_at=started_at,
                    request=request,
                    analysis=analysis,
                    role=role,
                    intent=action_scenario.intent,
                    answer=answer,
                    article_id=action_scenario.scenario_id,
                    score=250,
                    confidence="high",
                    matched_features=["structured_action", "trusted_context_required"],
                    action="clarify",
                    needs_ticket=False,
                    ticket_id=None,
                    ticket_created=False,
                    fallback_reason="trusted_context_required",
                    safety_categories=[],
                )
                return ChatResponse(
                    session_id=request.session_id,
                    message_id=message_id,
                    answer=answer,
                    intent=action_scenario.intent,
                    scenario_id=action_scenario.scenario_id,
                    resolution="clarified",
                    role=role,
                    needs_ticket=False,
                    actions=response_actions,
                    used_context=[],
                    confidence_level="high",
                    action="clarify",
                )
            if not reference_id:
                answer = "Укажите номер лота, чтобы я мог проверить статус в системе."
                message_id = _persist_turn(
                    started_at=started_at,
                    request=request,
                    analysis=analysis,
                    role=role,
                    intent=action_scenario.intent,
                    answer=answer,
                    article_id=action_scenario.scenario_id,
                    score=250,
                    confidence="high",
                    matched_features=["structured_action", "missing_reference_id"],
                    action="clarify",
                    needs_ticket=False,
                    ticket_id=None,
                    ticket_created=False,
                    fallback_reason="missing_reference_id",
                    safety_categories=[],
                )
                return ChatResponse(
                    session_id=request.session_id,
                    message_id=message_id,
                    answer=answer,
                    intent=action_scenario.intent,
                    scenario_id=action_scenario.scenario_id,
                    resolution="clarified",
                    role=role,
                    needs_ticket=False,
                    actions=_scenario_actions(action_scenario.scenario_id),
                    used_context=["trusted_user"],
                    confidence_level="high",
                    action="clarify",
                )
            status = status_provider.fetch(
                kind,
                str(request.context.user_id or ""),
                reference_id,
                str(request.trusted_context_token or ""),
            )
            if status.success:
                answer = f"Статус: {status.status}. {status.description}".strip()
                resolution = "status"
                fallback_reason = ""
            else:
                answer = (
                    "Сейчас не удалось получить подтверждённый статус из системы MIGTORG. "
                    "Я не буду предполагать результат — создайте обращение для ручной проверки."
                )
                resolution = "escalated"
                fallback_reason = status.error_code
            message_id = _persist_turn(
                started_at=started_at,
                request=request,
                analysis=analysis,
                role=role,
                intent=action_scenario.intent,
                answer=answer,
                article_id=action_scenario.scenario_id,
                score=300,
                confidence="high",
                matched_features=["structured_action", f"status_kind:{kind}"],
                action="fetch_status",
                needs_ticket=not status.success,
                ticket_id=None,
                ticket_created=False,
                fallback_reason=fallback_reason,
                safety_categories=[],
            )
            return ChatResponse(
                session_id=request.session_id,
                message_id=message_id,
                answer=answer,
                intent=action_scenario.intent,
                scenario_id=action_scenario.scenario_id,
                resolution=resolution,
                role=role,
                needs_ticket=not status.success,
                actions=_scenario_actions(action_scenario.scenario_id),
                used_context=["trusted_user", "lot_id"],
                data_freshness=status.freshness,
                confidence_level="high",
                action="fetch_status",
            )

    clarification_choice = (
        logger.consume_clarification_choice(request.session_id, request.message)
        if clarification_state
        else None
    )
    if dialogue and dialogue.get("category_choice"):
        clarification_choice = dialogue["category_choice"]
    if (
        clarification_choice
        and not clarification_choice.get("article_id")
        and str(clarification_choice.get("label") or "").strip().casefold() == "другая тема"
    ):
        answer = "Опишите вопрос своими словами — я заново подберу подходящую тему."
        message_id = _persist_turn(
            started_at=started_at,
            request=request,
            analysis=analysis,
            role=role,
            intent="unknown",
            answer=answer,
            article_id=None,
            score=0,
            confidence="medium",
            matched_features=["clarification_choice:other"],
            action="clarify",
            needs_ticket=False,
            ticket_id=None,
            ticket_created=False,
            fallback_reason="clarification_other",
            safety_categories=safety_before.categories,
        )
        return ChatResponse(
            session_id=request.session_id,
            message_id=message_id,
            answer=answer,
            intent="unknown",
            resolution="clarified",
            role=role,
            needs_ticket=False,
            safety_categories=safety_before.categories,
            safety_flags=safety_before.categories,
            confidence_level="medium",
            action="clarify",
            model_used="mock",
        )

    effective_message = request.message
    forced_intent = ""
    context_followup = False
    context_limit_reached = False
    previous_attempts = int(clarification_state.get("attempts", 0)) if clarification_state else 0
    if clarification_choice and not clarification_choice.get("article_id"):
        forced_intent = str(clarification_choice.get("intent") or "")
        selected_label = str(clarification_choice.get("label") or request.message)
        effective_message = selected_label
        context_followup = bool(forced_intent and forced_intent != "unknown")
    elif clarification_state and not clarification_choice:
        original_message = str(clarification_state.get("original_message") or "").strip()
        if previous_attempts >= MAX_CONTEXT_FOLLOWUPS:
            context_limit_reached = True
            logger.clear_pending_clarification(request.session_id)
        elif _is_standalone_new_topic(request.message, clarification_state):
            logger.clear_pending_clarification(request.session_id)
        elif original_message:
            current_intent = classify_intent(request.message)
            candidate_intents = _clarification_candidate_intents(clarification_state)
            if current_intent != "unknown" and current_intent in candidate_intents:
                forced_intent = current_intent
            effective_message = f"{original_message}. Уточнение пользователя: {request.message}"
            context_followup = True
            logger.clear_pending_clarification(request.session_id)

    if dialogue:
        effective_message = dialogue["search_message"]
        context_followup = dialogue["transition"] in {"continue", "correct", "repeat"}
    if effective_message != request.message:
        analysis = analyze_text(effective_message, request.context)

    if ticket_category_flow and clarification_choice:
        intent = forced_intent or "support"
        answer = (
            "Категория выбрана. Нажмите «Создать обращение» ниже — форма откроется уже с этой темой. "
            "Коротко опишите ситуацию и добавьте контакт для ответа."
        )
        suggested_fields = suggested_fields_for(intent, has_contact=bool(request.contact))
        message_id = _persist_turn(
            started_at=started_at,
            request=request,
            analysis=analysis,
            role=role,
            intent=intent,
            answer=answer,
            article_id=None,
            score=150,
            confidence="high",
            matched_features=["ticket_category_selected"],
            action="create_ticket",
            needs_ticket=True,
            ticket_id=None,
            ticket_created=False,
            fallback_reason="",
            safety_categories=safety_before.categories,
        )
        return ChatResponse(
            session_id=request.session_id,
            message_id=message_id,
            answer=answer,
            intent=intent,
            resolution="escalated",
            role=role,
            needs_ticket=True,
            suggested_fields=suggested_fields,
            ticket_required_fields=suggested_fields,
            safety_categories=safety_before.categories,
            safety_flags=safety_before.categories,
            confidence_level="high",
            action="create_ticket",
            model_used="mock",
        )

    selected_article = selected_scenario_article
    if dialogue and dialogue.get("selected_scenario_id"):
        selected_article = get_article_by_id(dialogue["selected_scenario_id"], role)
    if clarification_choice:
        selected_article = get_article_by_id(str(clarification_choice.get("article_id", "")), role)

    if selected_article:
        intent = selected_article.intent
        pattern_match = None
        search_result = KnowledgeSearchResult(
            article=selected_article,
            score=150,
            confidence="high",
            matched_features=["clarification_choice"],
        )
    else:
        intent = forced_intent or classify_intent(effective_message)
        pattern_match = best_intent_pattern(effective_message, request.context, analysis=analysis)
        search_result = search_knowledge_match(
            effective_message,
            intent,
            role,
            request.context,
            analysis=analysis,
            pattern_match=pattern_match,
            skip_topic_ambiguity=bool(forced_intent),
        )
    article = search_result.article
    if article and article.intent != "unknown":
        intent = article.intent
    confidence = search_result.confidence
    matched_features = list(search_result.matched_features)
    if context_followup:
        matched_features.append("context_followup")
    matched_features.extend(f"safety:{category}" for category in safety_before.categories)
    if (
        pattern_match
        and article is not None
        and settings.routing_architecture == "control"
        and pattern_match.intent == intent
        and not search_result.clarifying_options
    ):
        confidence = "high" if pattern_match.confidence_level == "high" else confidence
        matched_features.extend(
            f"pattern:{pattern_match.pattern_id}:{feature.group}:{feature.match_type}"
            for feature in pattern_match.matched_features
        )
    matched_features = list(dict.fromkeys(matched_features))
    if confidence in {"low", "medium"}:
        configured_fallbacks = load_matching_config().get("fallback_menu", [])
        fallback_labels = [
            str(item.get("label"))
            for item in configured_fallbacks
            if isinstance(item, dict) and item.get("label")
        ][:4]
        clarifying_options = search_result.clarifying_options or (
            pattern_match.clarifying_options if pattern_match else fallback_labels
        )
        clarification_intents = {
            option_intent
            for option_intent in search_result.clarifying_intents
            if option_intent != "unknown"
        }
        if article is None and len(clarification_intents) > 1:
            intent = "unknown"
        if search_result.fallback_reason == "out_of_scope":
            answer = (
                "Я отвечаю только на вопросы о работе MIGTORG — лотах, торгах, тарифах, оплате и документах. "
                "Не удалось определить связь вопроса с площадкой. Уточните, пожалуйста, ваш вопрос."
            )
        else:
            answer = search_result.clarifying_question or (
                "Не нашел точный ответ. Напишите проще — например: “лот”, “оплата”, “штраф”, “тариф”. "
                "Или выберите частую ситуацию ниже."
            )
        if safety_before.answer_prefix:
            answer = safety_before.answer_prefix + answer
        if safety_before.categories:
            _log_safety(
                request.session_id,
                request.context.user_id,
                request.message,
                safety_before.categories,
                answer,
                safety_before.needs_review,
                False,
            )
        if clarifying_options and not context_limit_reached:
            pending_options = []
            for index, label in enumerate(clarifying_options):
                article_id = (
                    search_result.clarifying_article_ids[index]
                    if index < len(search_result.clarifying_article_ids)
                    else ""
                )
                option_article = get_article_by_id(article_id, role) if article_id else None
                option_intent = (
                    search_result.clarifying_intents[index]
                    if index < len(search_result.clarifying_intents)
                    else (option_article.intent if option_article else classify_intent(label))
                )
                if label.casefold() == "другая тема":
                    option_intent = "unknown"
                pending_options.append(
                    {
                        "label": label,
                        "article_id": article_id,
                        "intent": option_intent,
                    }
                )
            logger.save_pending_clarification(
                request.session_id,
                pending_options,
                original_message=effective_message,
                context=_context_dict(request),
                attempts=(previous_attempts + 1) if context_followup else 1,
            )
        message_id = _persist_turn(
            started_at=started_at,
            request=request,
            analysis=analysis,
            role=role,
            intent=intent,
            answer=answer,
            article_id=article.slug if article else None,
            score=search_result.score,
            confidence=confidence,
            matched_features=matched_features,
            action="clarify",
            needs_ticket=False,
            ticket_id=None,
            ticket_created=False,
            fallback_reason=search_result.fallback_reason,
            safety_categories=safety_before.categories,
        )
        return ChatResponse(
            session_id=request.session_id,
            message_id=message_id,
            answer=answer,
            intent=intent,
            resolution=("out_of_scope" if search_result.fallback_reason == "out_of_scope" else "clarified"),
            role=role,
            needs_ticket=False,
            safety_categories=safety_before.categories,
            safety_flags=safety_before.categories,
            clarifying_options=clarifying_options,
            confidence_level=confidence,
            action="clarify",
            model_used="mock",
        )

    should_create_ticket = needs_ticket(
        effective_message,
        intent,
        request.context,
        analysis=analysis,
    ) or bool(article and article.needs_ticket)
    if article and article.scenario == "insurer_owner_vehicle_listing":
        should_create_ticket = False
    if article and article.scenario == "refund.eligibility":
        # Eligibility is an informational policy answer. A ticket is offered
        # only after the user chooses a concrete refund operation.
        should_create_ticket = False
    if article and article.scenario == "lot.location" and not request.context.lot_id:
        # General location guidance must not become an individual case merely
        # because the query contains the word "address".
        should_create_ticket = False
    message_lower = effective_message.lower()
    template_request_words = ("форма", "шаблон", "заявлен", "заполнить", "документ")
    is_template_request = any(word in message_lower for word in template_request_words)
    refusal_issue_words = ("не соответств", "поврежд", "расхожд", "дефект", "не совпад", "разбит", "отсутств")
    has_refusal_issue_details = any(word in message_lower for word in refusal_issue_words)
    is_deposit_refund_template_request = bool(
        article
        and article.scenario in {"deposit_refund_template", "refund.application"}
        and is_template_request
    )
    is_motivated_refusal_document_request = bool(
        article
        and article.scenario == "motivated_refusal_template"
        and (is_template_request or "нужен мотивированный отказ" in message_lower)
    )
    is_template_only_answer = bool(
        article
        and not has_refusal_issue_details
        and (
            is_deposit_refund_template_request
            or is_motivated_refusal_document_request
            or (
                intent == "refusal"
                and article.template
                and article.action == "show_document_and_offer_ticket"
                and is_template_request
            )
        )
    )
    if is_template_only_answer and not request.consent_to_ticket:
        should_create_ticket = False
    if intent == "lot" and request.context.lot_id:
        should_create_ticket = True
    extracted_contact = request.contact or request.context.user_email or request.context.user_phone
    if not extracted_contact and analysis.entities["email"]:
        extracted_contact = analysis.entities["email"][0]
    if not extracted_contact and analysis.entities["phone"]:
        extracted_contact = analysis.entities["phone"][0]
    contact_available = bool(extracted_contact)
    suggested_fields = suggested_fields_for(intent, has_contact=contact_available) if should_create_ticket else []
    if should_create_ticket and article and article.required_fields:
        suggested_fields = list(dict.fromkeys([*suggested_fields, *article.required_fields]))
        if contact_available:
            suggested_fields = [field for field in suggested_fields if field != "contact"]
    can_create_ticket = should_create_ticket and request.consent_to_ticket and contact_available
    ticket: Ticket | None = None
    if can_create_ticket:
        history = logger.get_history(request.session_id)
        ticket = build_ticket(
            effective_message,
            intent,
            role,
            request.context,
            extracted_contact,
            history,
            request.attachments,
        )
        ticket = _save_and_deliver_ticket(ticket)

    action_matrix = load_matching_config().get("intent_action_matrix", {})
    matrix_action = action_matrix.get(intent, "answer") if isinstance(action_matrix, dict) else "answer"
    action = "create_ticket" if should_create_ticket else str(matrix_action)
    if should_create_ticket and article and article.action == "show_document_and_offer_ticket":
        action = article.action
    elif is_template_only_answer:
        action = "show_document"
    elif intent == "unknown" and not article:
        action = "clarify"
    elif not should_create_ticket and article and article.action == "answer" and (article.scenario or article.answer_type == "general"):
        action = "answer"
    elif not should_create_ticket and article and article.action and article.action != "answer":
        action = article.action
    response_actions = _response_scenario_actions(effective_message, article.scenario if article else None)
    scenario_clarifying_options = (
        [item.label for item in response_actions if item.type == "clarify"]
        if action == "clarify"
        else []
    )

    generated = generate_answer(
        effective_message,
        intent,
        role,
        article,
        should_create_ticket,
        ticket.id if ticket else None,
        suggested_fields,
        settings=settings,
        session_id=request.session_id,
        safety_flags=safety_before.categories,
        llm_daily_spend_usd=logger.get_llm_spend(settings.llm_environment, days=1),
        llm_monthly_spend_usd=logger.get_llm_spend(settings.llm_environment, days=31),
        route_confidence=confidence,
        llm_allowed=(
            not bool(scenario_clarifying_options)
            and _in_llm_rollout(request.session_id, settings.llm_rollout_percentage)
        ),
    )
    matched_features = list(dict.fromkeys([
        *matched_features,
        *(f"answer_fact:{fact_id}" for fact_id in generated.used_fact_ids),
        f"answer_verifier:{generated.verification_reason or 'not_applicable'}",
    ]))
    answer = generated.answer
    if safety_before.answer_prefix:
        answer = safety_before.answer_prefix + answer
    safety_after = post_check(answer)
    safety_categories = list(dict.fromkeys([*safety_before.categories, *safety_after.categories]))
    template_links = [article.template] if article and article.template and safety_after.allowed else []
    if is_deposit_refund_template_request and safety_after.allowed and not template_links:
        template_links = [REFUND_APPLICATION_TEMPLATE]
    attachments = [link["url"] for link in template_links]
    if not safety_after.allowed:
        answer = safety_after.answer_override or answer
        _log_safety(
            request.session_id,
            request.context.user_id,
            request.message,
            safety_after.categories,
            answer,
            safety_after.needs_review,
            bool(ticket),
        )
    if safety_before.categories:
        _log_safety(
            request.session_id,
            request.context.user_id,
            request.message,
            safety_before.categories,
            answer,
            safety_before.needs_review,
            bool(ticket),
        )

    if generated.llm_result:
        all_safety_flags = list(dict.fromkeys([*safety_before.categories, *safety_after.categories]))
        logger.log_llm_request(
            generated.llm_result,
            request.session_id,
            role,
            should_create_ticket,
            all_safety_flags,
        )
        langfuse_client.capture_llm_result(
            generated.llm_result,
            request.session_id,
            role,
            should_create_ticket,
            all_safety_flags,
        )

    message_id = _persist_turn(
        started_at=started_at,
        request=request,
        analysis=analysis,
        role=role,
        intent=intent,
        answer=answer,
        article_id=article.slug if article else None,
        score=search_result.score,
        confidence=confidence,
        matched_features=matched_features,
        action=action,
        needs_ticket=should_create_ticket,
        ticket_id=ticket.id if ticket else None,
        ticket_created=bool(ticket),
        fallback_reason=search_result.fallback_reason,
        safety_categories=safety_categories,
    )
    return ChatResponse(
        session_id=request.session_id,
        message_id=message_id,
        answer=answer,
        intent=intent,
        scenario_id=(article.scenario if article and get_scenario(article.scenario) else None),
        resolution=("escalated" if should_create_ticket else "clarified" if action == "clarify" else "answered"),
        role=role,
        needs_ticket=should_create_ticket,
        ticket_id=ticket.id if ticket else None,
        safety_categories=safety_categories,
        safety_flags=safety_categories,
        suggested_fields=suggested_fields,
        ticket_required_fields=suggested_fields,
        attachments=attachments,
        template_links=template_links,
        actions=response_actions,
        used_context=[
            field
            for field, present in (
                ("page_type", bool(request.context.page_type)),
                ("lot_id", bool(request.context.lot_id)),
                ("trusted_user", bool(request.context.trusted)),
            )
            if present
        ],
        model_used=generated.llm_result.model if generated.llm_result else "mock",
        confidence_level=confidence,
        clarifying_options=scenario_clarifying_options,
        action=action,
    )


@app.get("/health")
@app.get("/api/health")
def health() -> dict:
    build_manifest = build_runtime_manifest(settings)
    return {
        "status": "ok",
        "app_name": settings.app_name,
        "environment": settings.app_environment,
        "deploy_version": settings.deploy_version,
        "widget_ready": (settings.widget_root / "index.html").is_file()
        and (settings.widget_root / "widget.js").is_file()
        and (settings.widget_root / "style.css").is_file(),
        "knowledge_mode": build_manifest["knowledge_mode"],
        "build_manifest": build_manifest,
    }


@app.get("/api/internal/quality-report")
def quality_report(
    days: int = 30,
    x_quality_report_token: str | None = Header(default=None),
) -> dict:
    expected_token = settings.quality_report_token
    if not expected_token:
        raise HTTPException(status_code=404, detail="Not Found")
    if not x_quality_report_token or not secrets.compare_digest(x_quality_report_token, expected_token):
        raise HTTPException(status_code=403, detail="Invalid quality report token")
    report = logger.get_quality_report(days=days, include_examples=False)
    daily_spend = logger.get_llm_spend(settings.llm_environment, days=1)
    monthly_spend = logger.get_llm_spend(settings.llm_environment, days=31)
    warning_ratio = settings.llm_budget_warning_pct / 100
    report["llm"]["budget"] = {
        "environment": settings.llm_environment,
        "daily_spend_usd": round(daily_spend, 6),
        "daily_limit_usd": settings.llm_daily_budget_usd,
        "daily_warning": daily_spend >= settings.llm_daily_budget_usd * warning_ratio,
        "monthly_spend_usd": round(monthly_spend, 6),
        "monthly_limit_usd": settings.active_llm_monthly_budget_usd,
        "monthly_warning": monthly_spend >= settings.active_llm_monthly_budget_usd * warning_ratio,
    }
    return report


@app.post("/api/internal/tickets/retry-due")
def retry_due_tickets(
    x_quality_report_token: str | None = Header(default=None),
) -> dict:
    expected_token = settings.quality_report_token
    if not expected_token:
        raise HTTPException(status_code=404, detail="Not Found")
    if not x_quality_report_token or not secrets.compare_digest(x_quality_report_token, expected_token):
        raise HTTPException(status_code=403, detail="Invalid quality report token")
    if not settings.ticket_email_enabled:
        raise HTTPException(status_code=409, detail="Email delivery is disabled")
    sent: list[str] = []
    failed: list[str] = []
    for row in logger.get_due_delivery_tickets():
        ticket = Ticket(
            id=str(row["id"]),
            status="new",
            topic=str(row["topic"]),
            description=str(row["description"]),
            contact=row["contact"],
            role=str(row["role"]),
            user_id=row["user_id"],
            lot_id=row["lot_id"],
            payment_id=row["payment_id"],
            session_id=str(row["session_id"]),
            page_type=row["page_type"],
            dialog_history=json.loads(str(row["dialog_history"] or "[]")),
            attachments=json.loads(str(row["attachments"] or "[]")),
            category=row["category"],
            priority=str(row["priority"] or "normal"),
            scenario_id=row["scenario_id"],
            source_message_id=row["source_message_id"],
            collected_fields=json.loads(str(row["collected_fields"] or "{}")),
            delivery_attempts=int(row["delivery_attempts"] or 0),
            created_at=str(row["created_at"]),
        )
        try:
            delivered = email_ticket_provider.deliver(ticket)
            logger.update_ticket_status(ticket.id, delivered.status)
            sent.append(ticket.id)
        except Exception:
            logger.record_ticket_delivery_failure(ticket.id)
            failed.append(ticket.id)
    return {"sent": sent, "failed": failed}


@app.get("/api/internal/review-queue")
def review_queue(
    days: int = 30,
    x_quality_report_token: str | None = Header(default=None),
) -> dict:
    expected_token = settings.quality_report_token
    if not expected_token:
        raise HTTPException(status_code=404, detail="Not Found")
    if not x_quality_report_token or not secrets.compare_digest(x_quality_report_token, expected_token):
        raise HTTPException(status_code=403, detail="Invalid quality report token")
    items = logger.get_review_queue(days=days, include_dev_sessions=False)
    return {"period_days": days, "item_count": len(items), "items": items}


@app.post("/api/chat/message", response_model=ChatResponse)
def chat_message(request: ChatRequest) -> ChatResponse:
    return process_chat_message(request)


@app.post("/api/chat/ticket")
def create_ticket(request: TicketCreateRequest) -> dict[str, str]:
    request.context.session_id = request.session_id
    _apply_trusted_context(request)
    role = "authorized" if request.context.trusted else "guest"
    history = logger.get_history(request.session_id)
    contact = request.contact or request.context.user_email or request.context.user_phone
    if not contact:
        raise HTTPException(status_code=400, detail="contact is required to create ticket")
    ticket = Ticket(
        topic=request.topic,
        description=request.description,
        contact=contact,
        role=role,
        user_id=request.context.user_id,
        lot_id=request.lot_id or request.context.lot_id,
        payment_id=request.payment_id,
        session_id=request.session_id,
        page_type=request.context.page_type,
        dialog_history=history,
        attachments=request.attachments,
        category=("callback" if request.request_callback else request.category),
        priority=request.priority,
        scenario_id=request.scenario_id,
        source_message_id=request.source_message_id,
        collected_fields={
            key: value
            for key, value in {
                "preferred_callback_time": request.preferred_callback_time,
            }.items()
            if value
        },
    )
    ticket = _save_and_deliver_ticket(ticket)
    logger.mark_ticket_created(request.session_id)
    return {"ticket_id": ticket.id, "status": ticket.status}


@app.get("/api/chat/history/{session_id}")
def chat_history(session_id: str) -> dict[str, list[dict]]:
    return {"messages": logger.get_history(session_id)}


@app.post("/api/chat/feedback")
def chat_feedback(request: ChatFeedbackRequest) -> dict[str, str]:
    logger.save_feedback(request.session_id, request.rating, request.comment, request.message_id)
    return {"status": "ok"}


@app.post("/chat", response_model=LegacyChatResponse)
def legacy_chat(request: LegacyChatRequest) -> LegacyChatResponse:
    response = process_chat_message(ChatRequest(message=request.message))
    answer = response.answer
    if response.needs_ticket and "оператор" not in answer.lower():
        answer += " При необходимости обращение передадут оператору."
    return LegacyChatResponse(answer=answer, needs_escalation=response.needs_ticket)


def _process_bound_chat_message(request: ChatRequest, dialogue_turn=None, lease_token=None) -> ChatResponse:
    """Bind issued actions to this session before any action or search runs."""
    import time
    from backend.app.bot.architecture_decision import decision_context
    from backend.app.bot.scenario_policy import action_allowed, scenario_allowed
    from backend.app.bot.knowledge_gaps import matching_gap
    started = perf_counter()
    _apply_trusted_context(request)
    role = "authorized" if request.context.trusted else "guest"
    previous = logger.get_response_state(request.session_id)
    trace = {"session_id": request.session_id, "logger": logger,
             "deadline": time.monotonic() + 5.0,
             "minimal_state": {"previous_scenario_id": previous["response"].get("scenario_id")} if previous else {},
             "previous_message_id": previous["message_id"] if previous else None}
    if dialogue_turn:
        from backend.app.bot.pii_redaction import redact_for_external_llm
        trace["dialogue"] = dialogue_turn.model_dump(exclude={"state"})
        trace["minimal_state"] = {
            "previous_scenario_id": dialogue_turn.state.active_scenario_id,
            "goal": redact_for_external_llm(dialogue_turn.understanding.goal),
            "objects": dialogue_turn.understanding.objects,
            "operations": dialogue_turn.understanding.operations,
            "entity_fields": sorted(dialogue_turn.understanding.entities),
        }
    token = decision_context.set(trace)
    try:
        valid_action = True
        if request.selected_action_id:
            issued = next((a for a in (previous["actions"] if previous else [])
                           if a["id"] == request.selected_action_id), None)
            definition = find_scenario_action(request.selected_action_id)
            valid_action = bool(issued and definition
                and scenario_allowed(definition[0], role)
                and action_allowed(ChatAction.model_validate(issued), role)
                and (not request.conversation_turn_id or request.conversation_turn_id == previous["message_id"]))
        if not valid_action:
            answer = "Это действие больше не относится к текущему ответу. Опишите, пожалуйста, что нужно проверить."
            message_id = _persist_turn(started_at=started, request=request,
                analysis=analyze_text(request.message, request.context), role=role, intent="unknown",
                answer=answer, article_id=None, score=0, confidence="low",
                matched_features=["action_not_issued_for_current_turn"], action="clarify",
                needs_ticket=False, ticket_id=None, ticket_created=False,
                fallback_reason="action_not_issued_for_current_turn", safety_categories=[])
            response = ChatResponse(session_id=request.session_id, message_id=message_id,
                answer=answer, intent="unknown", role=role, needs_ticket=False,
                resolution="clarified", action="clarify", confidence_level="low")
            trace["decision"] = {"reason": "action_not_issued_for_current_turn"}
        elif dialogue_turn and dialogue_turn.service_reply and pre_check(request.message).allowed:
            response = _dialogue_service_response(request, dialogue_turn, role, started)
            trace["service_text"] = "dialogue." + dialogue_turn.service_reply
        else:
            response = _process_chat_message(request)
        response.actions = [a for a in response.actions if action_allowed(a, role)]
        gap = matching_gap(request.message, response.scenario_id or "")
        if gap and response.resolution == "answered":
            response.resolution = "clarified"
            response.action = "clarify"
            response.confidence_level = "medium"
            trace["knowledge_gap"] = gap["gap_id"]
        if response.ticket_id:
            response.action_result = {"ticket_id": response.ticket_id, "created": True,
                                      "delivery": "unconfirmed"}
            response.answer += f" Обращение создано. Номер: {response.ticket_id}."
            trace["service_text"] = "runtime.ticket_created:confirmed_sqlite_record"
        trace["used_context"] = response.used_context
        if dialogue_turn:
            response.used_context = list(dict.fromkeys([*response.used_context, *dialogue_turn.used_context]))
            response.pending_requests = list(dialogue_turn.state.pending_requests)
            if response.pending_requests and response.resolution in {"answered", "escalated"}:
                response.answer += " Сохранил остальные вопросы; напишите «следующий вопрос», чтобы продолжить."
                trace["queue_service_text"] = "dialogue.pending_questions"
            trace["used_context"] = response.used_context
        trace["result"] = {"scenario_id": response.scenario_id, "resolution": response.resolution,
                           "ticket_offered": response.needs_ticket,
                           "ticket_created": bool(response.ticket_id), "ticket_delivered": None}
        trace["elapsed_ms"] = round((perf_counter() - started) * 1000, 2)
        logger.save_response_state(response, {k: v for k, v in trace.items() if k not in {"logger", "deadline", "session_id"}},
                                   dialogue_turn=dialogue_turn, lease_token=lease_token)
        return response
    finally:
        decision_context.reset(token)


def _dialogue_service_response(request, turn, role, started):
    manual = turn.service_reply == "manual_help"
    answer = ("Не удалось уточнить ситуацию. Можно создать письменное обращение для ручной проверки. "
              "Описание вопроса и уже сообщённые данные сохранены в истории; обращение ещё не отправлено."
              if manual else "Укажите правильный номер лота. Предыдущий номер больше не использую.")
    message_id = _persist_turn(started_at=started, request=request,
        analysis=analyze_text(request.message, request.context), role=role, intent=turn.understanding.intent,
        answer=answer, article_id=None, score=0, confidence="low", matched_features=["dialogue:" + turn.service_reply],
        action="create_ticket" if manual else "clarify", needs_ticket=manual, ticket_id=None,
        ticket_created=False, fallback_reason="dialogue:" + turn.service_reply, safety_categories=[])
    return ChatResponse(session_id=request.session_id, message_id=message_id, answer=answer,
        intent=turn.understanding.intent, role=role, needs_ticket=manual,
        resolution="escalated" if manual else "clarified", confidence_level="low",
        action="create_ticket" if manual else "clarify",
        suggested_fields=["contact", "description"] if manual else ["lot_id"],
        actions=[a for a in _scenario_actions("support.contact") if a.type == "open_ticket"] if manual else [])


def process_chat_message(request: ChatRequest) -> ChatResponse:
    if not settings.dialogue_state_enabled:
        return _process_bound_chat_message(request)
    from backend.app.bot.dialogue_understanding import prepare_turn
    from backend.app.models.dialogue import DialogueState
    _apply_trusted_context(request)
    role = "authorized" if request.context.trusted else "guest"
    lease = logger.acquire_dialogue_turn(request.session_id)
    if not lease:
        raise HTTPException(status_code=409, detail="dialogue_turn_in_progress")
    try:
        previous = logger.get_response_state(request.session_id)
        if ((request.conversation_turn_id and (not previous or request.conversation_turn_id != previous["message_id"]))
            or (request.state_version is not None and request.state_version != (previous["version"] if previous else 0))):
            raise HTTPException(status_code=409, detail="stale_conversation_turn")
        state = logger.load_dialogue_state(request.session_id)
        subject = "authorized:" + str(request.context.user_id) if request.context.trusted else "guest"
        if state.subject != subject:
            state = DialogueState(subject=subject)
        if previous and state.version != previous["version"]:
            # A flag-off turn invalidates older experimental context on re-enable.
            state = DialogueState(subject=subject)
        turn = prepare_turn(request.message, state, role, request.context.lot_id)
        if request.selected_action_id and previous and any(a["id"] == request.selected_action_id for a in previous["actions"]):
            # The bound handler still checks access and the exact previous turn.
            lot_id = request.context.lot_id or state.active.entities.get("lot_id")
            if lot_id:
                turn.understanding.entities.setdefault("lot_id", lot_id)
        # Only untrusted user-supplied business identifiers are restored here.
        # Identity, scopes, contact details and tokens always come from this request.
        request.context.lot_id = turn.understanding.entities.get("lot_id")
        if turn.resume_action_id and not request.selected_action_id:
            request.selected_action_id = turn.resume_action_id
        logger.clear_pending_clarification(request.session_id)
        return _process_bound_chat_message(request, dialogue_turn=turn, lease_token=lease)
    except RuntimeError as exc:
        if str(exc) == "dialogue_lease_lost":
            raise HTTPException(status_code=409, detail="dialogue_turn_expired") from exc
        raise
    finally:
        logger.release_dialogue_turn(request.session_id, lease)
