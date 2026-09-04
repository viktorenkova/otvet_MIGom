"""Conservative local understanding and state transitions, independent of routing.

Vocabulary comes from the project's taxonomy; no held-out query lookup table.
An inferred context never grants access and never supplies a business status.
"""
from functools import lru_cache
import json
from pathlib import Path
import re

from backend.app.bot.intent_classifier import classify_intent
from backend.app.bot.scenario_engine import extract_query_facets, get_scenario
from backend.app.bot.scenario_policy import scenario_allowed
from backend.app.bot.text_processing import normalize_matching_text, tokenize
from backend.app.models.dialogue import DialogueState, DialogueTurn, QueryUnderstanding


CALLBACK = re.compile(r"\b(?:перезвоните(?:\s+мне)?|прошу\s+перезвонить|позвоните\s+мне|свяжитесь\s+со\s+мной)\b[^?!;]*", re.I)
NEXT = re.compile(r"^(?:да[, ]+)?(?:следующий|второй|другой сохран[её]нный)\s+вопрос[.!?]*$", re.I)
SWITCH = re.compile(r"^(?:теперь|другой вопрос|новая тема|не про .+?[, ]+а про)\b[:, ]*", re.I)
FOLLOWUP = re.compile(r"^(?:а\s|и\s|а?$)|\b(?:его|е[её]|он|она|это|этого|дальше|потом|сколько ждать|не понял|не знаю)\b", re.I)
LOT = re.compile(r"\b(?:лот(?:а|у|ом|е)?|lot)\s*(?:№|#)?\s*([\w-]*\d[\w-]*)\b", re.I)
CORRECTION = re.compile(r"\bне\s+(?:лот\s*)?(?:№|#)?\s*(\d[\w-]*)\s*[,;]?\s*а\s+(?:лот\s*)?(?:№|#)?\s*(\d[\w-]*)", re.I)
NONCONTENT = re.compile(r"^(?:да|нет|не знаю|не понял[аи]?|непонятно|не помогло|то же самое|так же)[.!? ]*$", re.I)
STOP = {"как", "что", "где", "когда", "нужно", "хочу", "мне", "мой", "мои", "это", "про", "для", "или", "при", "после", "можно", "только", "вопрос", "migtorg", "мигторг"}


@lru_cache(maxsize=1)
def taxonomy():
    path = Path(__file__).resolve().parents[3] / "configs/retrieval_taxonomy_terms.json"
    return json.loads(path.read_text(encoding="utf-8"))


def roots(text):
    return {w[:5] for w in tokenize(normalize_matching_text(text)) if len(w) >= 4 and w not in STOP}


def understand(message: str) -> QueryUnderstanding:
    text = message.strip()
    callbacks = [m.group(0).strip(" ,.") for m in CALLBACK.finditer(text)
                 if not re.search(r"\bне\s*$", text[:m.start()])]
    primary = text
    if callbacks:
        for callback in callbacks:
            primary = primary.replace(callback, "").strip(" ,.;")
    if not primary:
        primary, callbacks = text, []
    parts = [p.strip() for p in re.split(r"(?<=\?)\s+(?=\w)|;|\b(?:и ещё|и еще|а ещё|а еще)\b", primary) if p.strip()]
    intents = [classify_intent(part) for part in parts]
    explicit_split = bool(re.search(r";|\b(?:и ещё|и еще|а ещё|а еще)\b", primary))
    independent = len(parts) > 1 and (explicit_split or ("unknown" not in intents and len(set(intents)) > 1))
    extra = (parts[1:] if independent else []) + callbacks
    primary = parts[0] if independent else primary
    background = []
    # Keep event history as context, but do not let it replace an explicit goal.
    contrast = re.split(r",?\s+но\s+", primary, maxsplit=1, flags=re.I)
    if len(contrast) == 2 and classify_intent(contrast[1]) != "unknown":
        background, primary = [contrast[0]], contrast[1]
    facets = extract_query_facets(primary)
    entities = {}
    lot_matches = LOT.findall(primary)
    if lot_matches:
        entities["lot_id"] = lot_matches[-1]
    correction = CORRECTION.search(primary)
    if correction:
        entities["lot_id"] = correction.group(2)
    tariff = re.search(r"\b(премиум|премиальн\w*|разов\w*)\b", primary, re.I)
    if tariff:
        entities["tariff_type"] = "Премиум" if tariff[1].casefold().startswith("преми") else "Разовый"
    actor = "seller" if re.search(r"\bя\s+продавец\b", primary, re.I) else "buyer" if re.search(r"\bя\s+покупатель\b", primary, re.I) else None
    unknown = []
    if not facets.objects and classify_intent(primary) == "unknown":
        unknown.append("object")
    return QueryUnderstanding(goal=primary, intent=classify_intent(primary),
        objects=sorted(facets.objects), operations=sorted(facets.operations), states=sorted(facets.states),
        stage=facets.stage, actor=actor, entities=entities, secondary_requests=extra,
        background_events=background, negations=re.findall(r"\bне\s+[\w-]+", primary, re.I), unknown_fields=unknown)


def resolve_choice(text: str, candidates: list[dict], role: str) -> str | None:
    """Resolve an explicit distinguishing term among the previously offered choices."""
    options = [(c, get_scenario(c.get("article_id", ""))) for c in candidates]
    options = [(c, s) for c, s in options if scenario_allowed(s, role)]
    exact = [s.scenario_id for c, s in options if normalize_matching_text(text) == normalize_matching_text(c["label"])]
    if len(exact) == 1:
        return exact[0]
    if len(tokenize(text)) > 12 or re.search(r"\bне\b", text, re.I):
        return None
    words = roots(text)
    title_roots = [(s.scenario_id, roots(c["label"])) for c, s in options]
    hits = []
    for sid, tokens in title_roots:
        others = set().union(*(other for other_id, other in title_roots if other_id != sid))
        if words & (tokens - others):
            hits.append(sid)
    return hits[0] if len(hits) == 1 else None


def object_hint(state: DialogueState) -> str:
    scenario = get_scenario(state.active_scenario_id or "")
    objects = list(state.active.objects) or (list(scenario.objects) if scenario else [])
    prefix = (state.active_scenario_id or "").split(".")[0]
    if prefix in taxonomy()["objects"] and prefix in objects:
        objects = [prefix]
    # Prefer the user's concrete item over incidental auction/site background.
    for preferred in ("refund", "tariff", "contract", "document", "bid", "lot", "office", "vehicle"):
        if preferred in objects:
            return taxonomy()["objects"].get(preferred, [preferred])[0]
    return taxonomy()["objects"].get(objects[0], [""])[0] if objects else ""


def prepare_turn(message: str, old: DialogueState, role: str, context_lot: str | None = None) -> DialogueTurn:
    state = old.model_copy(deep=True)
    current = understand(message)
    transition = "new" if state.status == "idle" else "switch"
    selected = None
    category_choice = None
    resume = None
    reply = None
    used = []
    if NEXT.fullmatch(message.strip()) and state.pending_requests:
        current = understand(state.pending_requests.pop(0))
        transition = "next"
    elif state.status != "idle":
        selected = resolve_choice(message, state.expected_candidates, role)
        if state.expected_candidates:
            from backend.app.bot.knowledge_search import is_ticket_creation_request
            if is_ticket_creation_request(state.active.goal):
                category_choice = next((c for c in state.expected_candidates if not c.get("article_id")
                    and normalize_matching_text(c["label"]) == normalize_matching_text(message)), None)
        correction = CORRECTION.search(message)
        rejected_id = re.search(r"\b(?:лот\s+не\s+\d+|не\s+мой\s+лот)\b", message, re.I)
        bare_id = re.fullmatch(r"\s*(?:№|#)?\s*(\d[\w-]{1,30})\s*[.!]?", message)
        if bare_id and state.expected_field == "lot_id":
            current.entities["lot_id"] = bare_id[1]
        changed_id = "lot_id" in current.entities
        has_new_goal = current.intent != "unknown" and current.intent != state.active.intent
        entity_only = bool(correction or rejected_id or bare_id or LOT.fullmatch(message.strip(" .!?")))
        continuation = (selected is not None or category_choice is not None or (changed_id and (entity_only or not has_new_goal)) or bool(rejected_id)
            or (bool(FOLLOWUP.search(message)) and not (has_new_goal and current.objects))
            or (state.status == "clarifying" and not has_new_goal))
        if SWITCH.match(message):
            continuation = False
            current = understand(SWITCH.sub("", message, count=1))
        if normalize_matching_text(message) == normalize_matching_text(state.active.goal):
            transition = "repeat"
            continuation = True
        elif continuation:
            transition = "correct" if correction or rejected_id or (changed_id and state.active.entities.get("lot_id") != current.entities["lot_id"]) else "continue"
        if continuation:
            used = ["active_task", "entities"]
            prior_entities = dict(state.active.entities)
            if rejected_id:
                prior_entities.pop("lot_id", None)
                current.unknown_fields.append("lot_id")
                reply = "clarify_entity"
            current.entities = {**prior_entities, **current.entities}
            if current.intent == "unknown":
                current.intent = state.active.intent
            if not current.objects:
                current.objects = list(state.active.objects)
            if current.objects and "object" in current.unknown_fields:
                current.unknown_fields.remove("object")
            current.actor = current.actor or state.active.actor
            current.stage = current.stage or state.active.stage
            if entity_only:
                current.goal = state.active.goal
                prior_id = state.active.entities.get("lot_id")
                if prior_id and (rejected_id or changed_id):
                    current.goal = re.sub(r"(?<!\w)" + re.escape(prior_id) + r"(?!\w)",
                                          current.entities.get("lot_id", ""), current.goal).strip()
                current.operations = list(state.active.operations)
            if state.expected_field == "lot_id" and current.entities.get("lot_id") and state.pending_action_id:
                resume = state.pending_action_id
            if state.status == "clarifying" and not selected and (NONCONTENT.fullmatch(message.strip()) or state.clarification_attempts >= 2):
                reply = "manual_help"
                current.background_events.append(current.goal)
                current.goal = state.active.goal
    if transition in {"new", "switch", "next"}:
        state.expected_candidates = []
        state.expected_field = None
        state.pending_action_id = None
        state.clarification_attempts = 0
        state.active_scenario_id = None
        if context_lot and ("lot" in current.objects or "vehicle" in current.objects):
            current.entities.setdefault("lot_id", context_lot)
    query = current.goal
    if transition in {"continue", "correct"} and not selected:
        hint = object_hint(old)
        if resume:
            query = current.goal
        elif hint and not extract_query_facets(query).objects:
            query = re.sub(r"\b(?:его|е[её]|он|она|это|этого)\b", hint, query, flags=re.I)
            if hint.casefold() not in query.casefold():
                query += " " + hint
        if current.entities.get("tariff_type") and "тариф" in query and current.entities["tariff_type"].casefold() not in query.casefold():
            query += " " + current.entities["tariff_type"]
    for item in current.secondary_requests:
        if item not in state.pending_requests:
            state.pending_requests.append(item)
    state.active = current
    return DialogueTurn(understanding=current, state=state, transition=transition, search_message=query,
        selected_scenario_id=selected, category_choice=category_choice,
        resume_action_id=resume, service_reply=reply, used_context=used)


def finish_turn(turn: DialogueTurn, response, clarification: dict | None, trace: dict) -> DialogueState:
    state = turn.state
    if response.intent == "safety":
        return DialogueState(subject=state.subject, previous_message_id=response.message_id,
                             previous_action=response.action)
    state.active_scenario_id = response.scenario_id or state.active_scenario_id
    state.previous_message_id = response.message_id
    state.previous_action = response.action
    state.expected_candidates = list((clarification or {}).get("options", []))
    state.expected_field = None
    state.pending_action_id = None
    features = trace.get("decision", {}).get("pipeline_features", [])
    if "missing_reference_id" in features:
        state.expected_field = "lot_id"
        state.pending_action_id = next((a.id for a in response.actions if a.type == "fetch_status"), None)
    if turn.service_reply == "clarify_entity":
        state.expected_field = "lot_id"
    if response.scenario_id:
        scenario = get_scenario(response.scenario_id)
        if scenario:
            state.active.intent = scenario.intent
            if not state.active.objects:
                state.active.objects = list(scenario.objects)
    if turn.service_reply == "manual_help":
        state.status = "manual_help"
        state.expected_candidates = []
    elif response.resolution == "clarified":
        state.status = "clarifying"
        state.clarification_attempts += 1
    else:
        state.status = "answered"
        state.clarification_attempts = 0
    return state
