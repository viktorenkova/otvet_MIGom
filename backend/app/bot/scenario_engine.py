from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
import json
from pathlib import Path
import re
from typing import Any

from backend.app.bot.text_processing import correct_typos, load_matching_config, normalize_text, tokenize
from backend.app.config import get_settings


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    title: str
    intent: str
    domain: str
    roles: tuple[str, ...]
    stage: str
    objects: tuple[str, ...]
    operations: tuple[str, ...]
    states: tuple[str, ...]
    positive_examples: tuple[str, ...]
    negative_examples: tuple[str, ...]
    required_context: tuple[str, ...]
    allowed_context: tuple[str, ...]
    facts: tuple[str, ...]
    fact_records: tuple[dict[str, Any], ...]
    short_answer: str
    detailed_answer: str
    next_step: str
    actions: tuple[dict[str, Any], ...]
    escalation: dict[str, Any]
    source: str
    source_version: str
    reviewed_at: str
    review_owner: str
    expert: str
    review_interval_days: int
    answer_policy: dict[str, Any]
    search_document: str
    retrieval_taxonomy_terms: tuple[dict[str, Any], ...]
    atomic_unit_ids: tuple[str, ...]
    knowledge_gap_ids: tuple[str, ...]
    status: str = "active"
    legacy_ids: tuple[str, ...] = ()

    @property
    def answer(self) -> str:
        parts = [self.short_answer.strip(), self.detailed_answer.strip(), self.next_step.strip()]
        return " ".join(part for part in parts if part)


@dataclass(frozen=True)
class QueryFacets:
    objects: frozenset[str] = frozenset()
    operations: frozenset[str] = frozenset()
    states: frozenset[str] = frozenset()
    stage: str = ""


@dataclass(frozen=True)
class ScenarioDecision:
    scenario: Scenario | None
    score: int
    confidence: str
    facets: QueryFacets
    matched_features: tuple[str, ...] = ()
    clarifying_question: str = ""
    candidates: tuple[Scenario, ...] = ()


OBJECT_PATTERNS = {
    "account": r"\b(?:аккаунт\w*|кабинет\w*|регистрац\w*|вход\w*)\b",
    "tariff": r"\b(?:тариф\w*|доступ\w*|подписк\w*)\b",
    "bid": r"\bставк\w*\b",
    "auction": r"\b(?:торг\w*|аукцион\w*|котиров\w*)\b",
    "lot": r"\bлот\w*\b",
    "contract": r"\b(?:договор\w*|дкп)\b",
    "payment": r"\b(?:платеж\w*|платёж\w*|оплат\w*|деньг\w*)\b",
    "balance": r"\b(?:баланс\w*|кошел[её]к\w*)\b",
    "commission": r"\b(?:комисси\w*|комса)\b",
    "refund": r"\b(?:возврат\w*|вернут\w*|вернуть|вывод\w*\s+ден\w*|вывести\s+ден\w*)\b",
    "deposit": r"\b(?:депозит\w*|обеспечительн\w*\s+плат[её]ж\w*)\b",
    "invoice": r"\b(?:счет\w*|счёт\w*|реквизит\w*|плат[её]жн\w*\s+поручен\w*)\b",
    "document": r"\b(?:документ\w*|доки|счет\w*|счёт\w*)\b",
    "vehicle": r"\b(?:авто|автомобил\w*|машин\w*|тачк\w*)\b",
    "seller": r"\b(?:продавц\w*|продавец|продаж\w*)\b",
    "support": r"\b(?:поддержк\w*|оператор\w*|менеджер\w*|сотрудник\w*)\b",
    "site": r"\b(?:сайт\w*|страниц\w*|карточк\w*|интерфейс\w*)\b",
}

OPERATION_PATTERNS = {
    "connect": r"\b(?:подключ\w*|начат\w*|стать|зарегистр\w*)\b",
    "participate": r"\b(?:участв\w*|торгова\w*|сделат\w*\s+ставк\w*)\b",
    "check": r"\b(?:провер\w*|посмотр\w*|узнат\w*|статус\w*)\b",
    "receive": r"\b(?:получ\w*|забрат\w*|взят\w*)\b",
    "publish": r"\b(?:выстав\w*|размест\w*|вылож\w*|продат\w*)\b",
    "contact": r"\b(?:позвон\w*|перезвон\w*|связ\w*|контакт\w*)\b",
    "buy": r"\b(?:купит\w*|покупа\w*|приобрест\w*)\b",
    "pay": r"\b(?:оплат\w*|заплат\w*|перечисл\w*|пополн\w*|списал\w*)\b",
    "refund": r"\b(?:вернут\w*|вернуть|возврат\w*|вывест\w*|вывод\w*)\b",
    "inspect": r"\b(?:осмотр\w*|осмотрет\w*|посмотрет\w*|проверит\w*)\b",
    "recover": r"\b(?:восстанов\w*|войти|зайти|парол\w*|код\w*)\b",
    "troubleshoot": r"\b(?:ошибк\w*|не\s+работ\w*|завис\w*|не\s+открыва\w*)\b",
}

STATE_PATTERNS = {
    "not_visible": r"\b(?:не\s+вид\w*|не\s+отображ\w*|пропал\w*)\b",
    "missing": r"\b(?:не\s+пришел\w*|не\s+пришёл\w*|не\s+получ\w*|нет)\b",
    "completed": r"\b(?:заверш\w*|законч\w*|окончен\w*)\b",
    "waiting": r"\b(?:жду|ожида\w*|долго)\b",
    "blocked": r"\b(?:заблокир\w*|блокировк\w*|бан)\b",
    "no_response": r"\b(?:не\s+отвеч\w*|тишин\w*|молчит|игнор\w*)\b",
    "unavailable": r"\b(?:не\s+получа\w*|не\s+мог\w*|недоступ\w*|не\s+приход\w*)\b",
    "error": r"\b(?:ошибк\w*|некоррект\w*|завис\w*|сломал\w*)\b",
}


def _normal(text: str) -> str:
    return correct_typos(normalize_text(text))


def _is_named_seller_silence(text: str) -> bool:
    organizations = load_matching_config().get("domain_organizations", [])
    has_organization = bool(
        isinstance(organizations, list)
        and any(
            re.search(rf"(?<!\w){re.escape(_normal(str(name)))}(?!\w)", text)
            for name in organizations
            if _normal(str(name))
        )
    ) or bool(re.search(r"\bстрахов\w*\b", text))
    has_silence = bool(
        re.search(
            r"\b(?:молчит|тишин\w*|игнор\w*|гасит\w*|гасится|пропал\w*|"
            r"не\s+отвеч\w*|не\s+выходит\w*(?:\s+на\s+связь)?)\b",
            text,
        )
    )
    return has_organization and has_silence


def extract_query_facets(message: str) -> QueryFacets:
    text = _normal(message)
    objects = frozenset(name for name, pattern in OBJECT_PATTERNS.items() if re.search(pattern, text))
    operations = frozenset(name for name, pattern in OPERATION_PATTERNS.items() if re.search(pattern, text))
    states = frozenset(name for name, pattern in STATE_PATTERNS.items() if re.search(pattern, text))
    stage = ""
    if re.search(r"\b(?:выигр\w*|побед\w*)\b", text):
        stage = "after_win"
    elif re.search(r"\b(?:до|перед)\s+(?:торг\w*|ставк\w*)\b", text):
        stage = "before_bid"
    elif "registration" in objects or "account" in objects:
        stage = "registration"
    return QueryFacets(objects, operations, states, stage)


def _scenario_from_dict(raw: dict[str, Any]) -> Scenario:
    return Scenario(
        scenario_id=str(raw["scenario_id"]),
        title=str(raw["title"]),
        intent=str(raw.get("intent") or "unknown"),
        domain=str(raw.get("domain") or raw.get("intent") or "unknown"),
        roles=tuple(str(item) for item in raw.get("roles", ["guest", "authorized"])),
        stage=str(raw.get("stage") or ""),
        objects=tuple(str(item) for item in raw.get("objects", [])),
        operations=tuple(str(item) for item in raw.get("operations", [])),
        states=tuple(str(item) for item in raw.get("states", [])),
        positive_examples=tuple(str(item) for item in raw.get("positive_examples", [])),
        negative_examples=tuple(str(item) for item in raw.get("negative_examples", [])),
        required_context=tuple(str(item) for item in raw.get("required_context", [])),
        allowed_context=tuple(str(item) for item in raw.get("allowed_context", [])),
        facts=tuple(str(item) for item in raw.get("facts", [])),
        fact_records=tuple(dict(item) for item in raw.get("fact_records", []) if isinstance(item, dict)),
        short_answer=str(raw.get("short_answer") or ""),
        detailed_answer=str(raw.get("detailed_answer") or ""),
        next_step=str(raw.get("next_step") or ""),
        actions=tuple(dict(item) for item in raw.get("actions", []) if isinstance(item, dict)),
        escalation=dict(raw.get("escalation", {})),
        source=str(raw.get("source") or ""),
        source_version=str(raw.get("source_version") or "legacy-v2"),
        reviewed_at=str(raw.get("reviewed_at") or ""),
        review_owner=str(raw.get("review_owner") or ""),
        expert=str(raw.get("expert") or ""),
        review_interval_days=int(raw.get("review_interval_days") or 30),
        answer_policy=dict(raw.get("answer_policy", {})),
        search_document=str(raw.get("search_document") or ""),
        retrieval_taxonomy_terms=tuple(
            dict(item) for item in raw.get("retrieval_taxonomy_terms", []) if isinstance(item, dict)
        ),
        atomic_unit_ids=tuple(str(item) for item in raw.get("atomic_unit_ids", [])),
        knowledge_gap_ids=tuple(str(item) for item in raw.get("knowledge_gap_ids", [])),
        status=str(raw.get("status") or "active"),
        legacy_ids=tuple(str(item) for item in raw.get("legacy_ids", [])),
    )


@lru_cache(maxsize=1)
def load_scenarios() -> tuple[Scenario, ...]:
    root = get_settings().knowledge_root
    v31_path = root / "v3_1" / "scenarios.json"
    path = v31_path if v31_path.exists() else root / "v2" / "scenarios.json"
    if not path.exists():
        return ()
    raw = json.loads(path.read_text(encoding="utf-8"))
    schema_version = str(raw.get("schema_version") or "")
    if path == v31_path and schema_version != "3.1":
        raise ValueError("knowledge/v3_1/scenarios.json must use schema_version=3.1")
    if path != v31_path and schema_version != "2":
        raise ValueError("knowledge/v2/scenarios.json must use schema_version=2")
    return tuple(
        _scenario_from_dict(item)
        for item in raw.get("records", [])
        if isinstance(item, dict) and str(item.get("status") or "active") == "active"
    )


def get_scenario(scenario_id: str) -> Scenario | None:
    return next((item for item in load_scenarios() if item.scenario_id == scenario_id), None)


def find_scenario_action(action_id: str) -> tuple[Scenario, dict[str, Any]] | None:
    for scenario in load_scenarios():
        for action in scenario.actions:
            if str(action.get("id") or "") == action_id:
                return scenario, dict(action)
    return None


def _example_score(message: str, example: str) -> int:
    text = _normal(message)
    target = _normal(example)
    if text == target:
        return 220
    if target and target in text:
        return 165
    message_terms = {term for term in tokenize(text) if len(term) > 2}
    target_terms = {term for term in tokenize(target) if len(term) > 2}
    if not target_terms:
        return 0
    overlap = len(message_terms & target_terms)
    coverage = overlap / len(target_terms)
    if coverage >= 0.8 and overlap >= 2:
        return 110 + overlap * 5
    if coverage >= 0.5 and overlap >= 2:
        return 55 + overlap * 5
    return 0


def match_scenario(message: str, role: str) -> ScenarioDecision:
    text = _normal(message)
    facets = extract_query_facets(message)
    if _is_named_seller_silence(text):
        scenario = get_scenario("transfer.seller_no_response")
        if scenario:
            return ScenarioDecision(
                scenario,
                240,
                "high",
                facets,
                ("scenario_route:named_seller_silence",),
            )
    asks_for_visit_or_address = bool(
        re.fullmatch(
            r"(?:как|можно|хочу|куда|нужно|собираюсь)\s+"
            r"(?:(?:мне|нам)\s+)?(?:к\s+вам\s+)?(?:в\s+офис\s+)?"
            r"(?:попасть|приехать|доехать|добраться|ехать)(?:\s+(?:к\s+вам\s+)?в\s+офис)?"
            r"|как\s+до\s+вас\s+добраться"
            r"|(?:где(?:\s+находится)?|какой|подскажите)\s+(?:(?:у\s+вас|ваш|вашего)\s+)?(?:адрес|офис)"
            r"|(?:где|какой)\s+(?:адрес|офис)\s+(?:migtorg|мигторг)",
            text,
        )
    )
    if asks_for_visit_or_address:
        candidate_ids = ("support.office_visit", "inspection.arrange", "pickup.access_issuer")
        scenarios_by_id = {item.scenario_id: item for item in load_scenarios()}
        candidates = tuple(scenarios_by_id[item] for item in candidate_ids if item in scenarios_by_id)
        return ScenarioDecision(
            None,
            90,
            "medium",
            facets,
            ("scenario_ambiguity:visit_purpose",),
            "Уточните цель визита: вам нужен офис MIGTORG, осмотр автомобиля или место получения выигранного автомобиля?",
            candidates,
        )
    if re.fullmatch(r"регистрац\w*\s+(?:и|или)\s+вход", text):
        candidate_ids = ("account.registration", "account.login_problem")
        scenarios_by_id = {item.scenario_id: item for item in load_scenarios()}
        candidates = tuple(scenarios_by_id[item] for item in candidate_ids if item in scenarios_by_id)
        return ScenarioDecision(
            None,
            90,
            "medium",
            facets,
            ("scenario_ambiguity:registration_or_login",),
            "Что именно вам нужно: зарегистрироваться или решить проблему со входом?",
            candidates,
        )
    if re.fullmatch(
        r"(?:как\s+)?(?:(?:вообще|мне)\s+)?(?:начать\s+)?(?:работать|пользоваться)"
        r"(?:\s+(?:(?:с\s+)?(?:migtorg|мигторг|площадкой|сайтом)|на\s+(?:вашей\s+)?площадке))?",
        text,
    ):
        candidate_ids = ("platform.about", "buyer.get_started", "seller.get_started")
        scenarios_by_id = {item.scenario_id: item for item in load_scenarios()}
        candidates = tuple(scenarios_by_id[item] for item in candidate_ids if item in scenarios_by_id)
        return ScenarioDecision(
            None,
            90,
            "medium",
            facets,
            ("scenario_ambiguity:how_to_use",),
            "Уточните, пожалуйста: хотите узнать, как устроена площадка, начать покупать или стать продавцом?",
            candidates,
        )
    if re.fullmatch(
        r"(?:шаблон|форма|образец)(?:\s+заявлен\w*)?(?:\s+на)?\s+возврат\w*"
        r"|(?:пришл\w*|нужен|нужна)\s+(?:шаблон|форма)(?:\s+заявлен\w*)?(?:\s+на)?\s+возврат\w*",
        text,
    ):
        scenario = get_scenario("refund.application")
        if scenario:
            return ScenarioDecision(
                scenario,
                230,
                "high",
                facets,
                ("scenario_route:refund_template",),
            )
    if re.fullmatch(
        r"(?:(?:как|можно|хочу|нужно)\s+)?(?:вернуть|вывести|получить\s+обратно)(?:\s+(?:мои|свои))?\s+(?:деньги|средства)"
        r"|(?:хочу|нужен|оформить)\s+возврат"
        r"|возврат(?:\s+(?:денежн\w+\s+средств|денег|средств))?",
        text,
    ):
        candidate_ids = (
            "refund.eligibility",
            "refund.application",
            "refund.destination",
            "refund.timing_status",
        )
        candidates = tuple(item for item in load_scenarios() if item.scenario_id in candidate_ids)
        return ScenarioDecision(
            None,
            85,
            "medium",
            facets,
            ("scenario_ambiguity:generic_refund",),
            "Что именно вы хотите вернуть?",
            candidates,
        )
    if re.fullmatch(r"(?:как\s*)?подключ\w*", text):
        candidate_ids = ("buyer.get_started", "seller.get_started", "tariff.connect")
        candidates = tuple(item for item in load_scenarios() if item.scenario_id in candidate_ids)
        return ScenarioDecision(
            None,
            80,
            "medium",
            facets,
            ("scenario_ambiguity:connect",),
            "Что именно вы хотите подключить?",
            candidates,
        )

    generic_lot_question = bool(
        not facets.stage
        and not facets.operations
        and not facets.states
        and re.fullmatch(
            r"(?:(?:у\s+меня|есть)\s+)?(?:вопрос|информация|подскажите)"
            r"(?:\s+(?:по|о|об))?(?:\s+(?:моему|конкретному|этому))?\s+лот\w*"
            r"(?:\s+(?:номер|№)?\s*[a-zа-я0-9-]+)?",
            text,
        )
    )
    if generic_lot_question:
        candidate_ids = (
            "win.next_steps",
            "auction.status",
            "contract.receive",
            "pickup.receive_lot",
        )
        candidates = tuple(item for item in load_scenarios() if item.scenario_id in candidate_ids)
        return ScenarioDecision(
            None,
            90,
            "medium",
            facets,
            ("scenario_ambiguity:generic_lot",),
            "Что именно нужно узнать по лоту?",
            candidates,
        )

    ranked: list[tuple[int, Scenario, tuple[str, ...]]] = []
    for scenario in load_scenarios():
        if role not in scenario.roles and "all" not in scenario.roles:
            continue
        if any(_normal(example) in text for example in scenario.negative_examples if _normal(example)):
            continue
        example_score = max((_example_score(message, item) for item in scenario.positive_examples), default=0)
        object_hits = facets.objects.intersection(scenario.objects)
        operation_hits = facets.operations.intersection(scenario.operations)
        state_hits = facets.states.intersection(scenario.states)
        score = example_score + len(object_hits) * 22 + len(operation_hits) * 28 + len(state_hits) * 22
        features = []
        if example_score:
            features.append("scenario_example")
        features.extend(f"object:{item}" for item in sorted(object_hits))
        features.extend(f"operation:{item}" for item in sorted(operation_hits))
        features.extend(f"state:{item}" for item in sorted(state_hits))
        if scenario.stage and facets.stage == scenario.stage:
            score += 20
            features.append(f"stage:{scenario.stage}")
        if scenario.scenario_id == "support.callback" and (
            facets.stage or facets.objects.difference({"support"})
        ):
            score -= 120
            features.append("supplemental_action:callback")
        if score:
            ranked.append((score, scenario, tuple(features)))

    ranked.sort(key=lambda item: item[0], reverse=True)
    if not ranked:
        return ScenarioDecision(None, 0, "low", facets)
    score, best, features = ranked[0]
    margin = score - (ranked[1][0] if len(ranked) > 1 else 0)
    if score >= 165 and margin >= 18:
        return ScenarioDecision(best, score, "high", facets, (*features, f"scenario_margin:{margin}"))
    if score >= 95:
        candidates = tuple(item[1] for item in ranked[:3])
        return ScenarioDecision(
            None,
            score,
            "medium",
            facets,
            (*features, f"scenario_margin:{margin}"),
            "Уточните, пожалуйста, какой вариант ближе к вашему вопросу:",
            candidates,
        )
    return ScenarioDecision(None, score, "low", facets, features)


def clear_scenario_cache() -> None:
    load_scenarios.cache_clear()
