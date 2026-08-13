from dataclasses import dataclass, field
from functools import lru_cache
import json
from pathlib import Path
import re
from typing import Any, Protocol

from backend.app.config import get_settings
from backend.app.bot.semantic_search import MultilingualHybridSemanticIndex
from backend.app.bot.scenario_engine import clear_scenario_cache, load_scenarios, match_scenario
from backend.app.bot.topic_router import route_topic
from backend.app.bot.text_processing import (
    IntentPatternMatch,
    TextAnalysis,
    analyze_text,
    best_intent_pattern,
    confidence_level,
    correct_typos,
    load_matching_config,
    normalize_text,
    phrase_matches,
    tokenize,
)
from backend.app.models.user_context import UserRole


MATCHING_CONFIG_PATH = Path("configs/no_llm_matching_config.json")

STOPWORDS = {
    "меня",
    "мне",
    "мой",
    "моя",
    "мое",
    "моё",
    "мои",
    "ваш",
    "ваша",
    "ваше",
    "ваши",
    "сайт",
    "да",
    "нет",
    "это",
    "как",
    "что",
    "где",
    "если",
    "или",
    "для",
    "при",
    "нужен",
    "нужна",
    "нужно",
    "хочу",
    "есть",
    "после",
    "почему",
    "можно",
    "вас",
    "нам",
    "вам",
}

DISCRIMINATIVE_TERM_STEMS = (
    "тариф",
    "платеж",
    "оплат",
    "ставк",
    "торг",
    "лот",
    "vin",
    "вин",
    "осмотр",
    "передач",
    "стоянк",
    "дкп",
    "возврат",
    "депозит",
    "штраф",
    "отказ",
    "регистрац",
    "парол",
    "аккаунт",
    "кабинет",
    "поддержк",
    "специалист",
    "сотрудник",
)

# A semantic match is useful only after the message is known to be about
# MIGTORG. Without this gate, unrelated short phrases can accidentally look
# similar to a knowledge-base article and receive a confident answer.
DOMAIN_TERM_STEMS = (
    *DISCRIMINATIVE_TERM_STEMS,
    "migtorg",
    "мигторг",
    "аукцион",
    "автомоб",
    "машин",
    "имуществ",
    "док",
    "договор",
    "счет",
    "счёт",
    "акт",
    "бухгалтер",
    "закрыва",
    "переда",
    "получ",
    "выдач",
    "забрат",
    "выкуп",
    "продав",
    "покупател",
    "сделк",
    "реквизит",
    "баланс",
    "доступ",
    "демо",
    "площадк",
    "каталог",
    "карточк",
    "подписк",
    "комисс",
    "кешб",
    "бонус",
    "аккредитац",
    "личн",
    "профил",
    "войти",
    "вход",
    "парол",
    "осмотр",
    "стоянк",
    "эвакуатор",
    "авто",
    "участ",
    "страхов",
)

DOMAIN_SERVICE_PHRASES = (
    "кто вы",
    "что это за сайт",
    "что вы умеете",
    "чем можете помочь",
    "как с вами связаться",
    "как связаться",
    "куда обратиться",
    "не могу дозвониться",
    "где офис",
)

TICKET_REQUEST_PATTERN = re.compile(
    r"^(?:(?:хочу|нужно|надо|можно|помогите)\s+)?"
    r"(?:(?:создать|составить|оформить|подать|написать)\s+)?"
    r"(?:обращение|претензию|претензия|жалобу|жалоба)"
    r"(?:\s+(?:в поддержку|сотрудникам|оператору))?$"
)

INTENT_TERM_STEMS: dict[str, tuple[str, ...]] = {
    "registration": ("регистрац", "парол", "аккаунт", "кабинет"),
    "tariffs": ("тариф",),
    "payment": ("платеж", "оплат"),
    "bidding": ("ставк", "торг"),
    "lot": ("лот", "vin"),
    "inspection": ("осмотр",),
    "transfer": ("передач", "дкп"),
    "pickup": ("стоянк",),
    "refusal": ("отказ",),
    "penalty": ("штраф",),
    "refund": ("возврат", "депозит"),
    "support": ("поддержк", "специалист", "сотрудник"),
}


TECHNICAL_MARKERS = (
    "бот должен",
    "бот не должен",
    "первая линия",
    "пользователь должен",
    "попросите пользователя",
    "передайте вопрос",
    "корректный ответ",
    "действие бота",
)


@dataclass(frozen=True)
class KnowledgeArticle:
    title: str
    slug: str
    section: str
    content: str
    intent: str = "unknown"
    user_answer: str | None = None
    problem: str = ""
    internal_action: str = ""
    needs_ticket: bool = False
    safety_flags: list[str] = field(default_factory=list)
    template: dict[str, str] | None = None
    channel: str = ""
    scenario: str = ""
    user_phrases: list[str] = field(default_factory=list)
    negative_phrases: list[str] = field(default_factory=list)
    priority: int = 0
    fallback_allowed: bool = True
    required_fields: list[str] = field(default_factory=list)
    answer_type: str = ""
    action: str = ""
    trigger_phrases: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    negative_keywords: list[str] = field(default_factory=list)
    page_type_boost: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class PreparedArticle:
    normalized_title: str
    normalized_problem: str
    normalized_haystack: str
    haystack_words: frozenset[str]
    normalized_negative_phrases: tuple[str, ...]


@dataclass(frozen=True)
class KnowledgeSearchResult:
    article: KnowledgeArticle | None
    score: int
    confidence: str
    matched_features: list[str] = field(default_factory=list)
    fallback_reason: str = ""
    clarifying_question: str = ""
    clarifying_options: list[str] = field(default_factory=list)
    semantic_similarity: float = 0.0
    semantic_margin: float = 0.0
    clarifying_article_ids: list[str] = field(default_factory=list)
    clarifying_intents: list[str] = field(default_factory=list)


class KnowledgeSearchProvider(Protocol):
    def search(
        self,
        message: str,
        intent: str,
        role: UserRole,
        context: Any | None = None,
        analysis: TextAnalysis | None = None,
        pattern_match: IntentPatternMatch | None = None,
    ) -> KnowledgeSearchResult:
        ...


def _semantic_query(message: str, analysis: TextAnalysis | None = None) -> str:
    """Expand only high-confidence domain shorthand before vector search."""
    text = _normalize(message)
    expanded = analysis.synonym_normalized if analysis else text
    organizations = _load_matching_config().get("domain_organizations", [])
    organization_mentioned = bool(
        isinstance(organizations, list)
        and any(_normalize(str(name)) in text for name in organizations)
    )
    contact_delay = bool(
        re.search(r"\b(?:молчит|тишин\w*|игнор\w*|гасит\w*|гасится|пропал\w*|не\s+отвеч\w*|не\s+выходит\w*)\b", text)
    )
    if organization_mentioned and contact_delay:
        expanded += " страховая продавец не выходит на связь не отвечает передача лота документы"
    if re.fullmatch(r"как\s+участв\w*", text):
        expanded += " как участвовать в торгах сделать ставку начать торговаться"
    return expanded


class TfidfSemanticSearchProvider:
    def search(
        self,
        message: str,
        intent: str,
        role: UserRole,
        context: Any | None = None,
        analysis: TextAnalysis | None = None,
        pattern_match: IntentPatternMatch | None = None,
    ) -> KnowledgeSearchResult:
        config = _semantic_config()
        if not bool(config.get("enabled", False)):
            return KnowledgeSearchResult(None, 0, "low", fallback_reason="semantic_search_disabled")

        allowed = {"public", "guest"} if role == "guest" else {"public", "guest", "authorized"}
        candidates = [
            article
            for article in load_articles()
            if article.section in allowed
            and article.intent != "prohibited"
            and article.fallback_allowed
            and not _has_negative_phrase(article, message)
        ]
        match = _semantic_index().search(
            _semantic_query(message, analysis),
            {article.slug for article in candidates},
            intent,
        )
        if not match:
            return KnowledgeSearchResult(None, 0, "low", fallback_reason="semantic_no_candidates")

        article = next((item for item in candidates if item.slug == match.article_id), None)
        if not article:
            return KnowledgeSearchResult(None, 0, "low", fallback_reason="semantic_article_unavailable")

        answer_similarity = float(config.get(
            "dense_answer_similarity" if match.dense_available else "answer_similarity",
            0.30,
        ))
        answer_margin = float(config.get(
            "dense_answer_margin" if match.dense_available else "answer_margin",
            0.04,
        ))
        clarify_similarity = float(config.get(
            "dense_clarify_similarity" if match.dense_available else "clarify_similarity",
            0.20,
        ))
        clarify_margin = float(config.get(
            "dense_clarify_margin" if match.dense_available else "clarify_margin",
            0.03,
        ))
        features = [
            f"semantic_hybrid:{match.similarity:.3f}",
            f"semantic_margin:{match.margin:.3f}",
            f"semantic_lexical:{match.lexical_similarity:.3f}",
        ]
        if match.dense_available:
            features.append(f"semantic_dense:{match.dense_similarity:.3f}")
        domainless_dense_floor = float(config.get("domainless_dense_answer_similarity", 0.84))
        may_answer = _has_domain_signal(message) or (
            match.dense_available and match.dense_similarity >= domainless_dense_floor
        )
        if may_answer and match.similarity >= answer_similarity and match.margin >= answer_margin:
            return KnowledgeSearchResult(
                article=article,
                score=80 + round(match.similarity * 20),
                confidence="high",
                matched_features=features,
                semantic_similarity=match.similarity,
                semantic_margin=match.margin,
            )
        if _has_domain_signal(message) and match.similarity >= clarify_similarity and match.margin >= clarify_margin:
            candidate_articles = [
                candidate
                for article_id in match.candidate_article_ids
                if (candidate := next((item for item in candidates if item.slug == article_id), None))
            ]
            menu = _load_matching_config().get("fallback_menu", [])
            intent_labels = {
                str(item.get("value")): str(item.get("label"))
                for item in menu
                if isinstance(item, dict) and item.get("value") and item.get("label")
            } if isinstance(menu, list) else {}
            configured_labels = config.get("topic_labels", {})
            if isinstance(configured_labels, dict):
                intent_labels.update({str(key): str(value) for key, value in configured_labels.items()})
            option_labels = [
                intent_labels.get(candidate.intent, candidate.title)
                for candidate in candidate_articles
            ]
            option_ids = [candidate.slug for candidate in candidate_articles]
            option_labels.append("Другая тема")
            option_ids.append("")
            return KnowledgeSearchResult(
                article=article,
                score=50 + round(match.similarity * 20),
                confidence="medium",
                matched_features=features,
                clarifying_question="Выберите тему, которая ближе к вашему вопросу:",
                clarifying_options=option_labels,
                semantic_similarity=match.similarity,
                semantic_margin=match.margin,
                clarifying_article_ids=option_ids,
                clarifying_intents=[candidate.intent for candidate in candidate_articles] + ["unknown"],
            )
        return KnowledgeSearchResult(
            None,
            round(match.similarity * 100),
            "low",
            features,
            "semantic_below_threshold",
            semantic_similarity=match.similarity,
            semantic_margin=match.margin,
        )


def _normalize(text: str) -> str:
    return correct_typos(normalize_text(text))


def _tokens(text: str) -> list[str]:
    return [word for word in tokenize(text) if len(word) > 2 and word not in STOPWORDS]


def _has_discriminative_term(terms: set[str]) -> bool:
    return any(term.startswith(stem) for term in terms for stem in DISCRIMINATIVE_TERM_STEMS)


def _has_intent_term(terms: set[str], intent: str) -> bool:
    stems = INTENT_TERM_STEMS.get(intent, ())
    return any(term.startswith(stem) for term in terms for stem in stems)


def _has_domain_signal(message: str) -> bool:
    normalized_message = _normalize(message)
    if any(phrase in normalized_message for phrase in DOMAIN_SERVICE_PHRASES):
        return True
    terms = set(tokenize(normalized_message))
    if any(term.startswith(stem) for term in terms for stem in DOMAIN_TERM_STEMS):
        return True
    organizations = _load_matching_config().get("domain_organizations", [])
    return bool(
        isinstance(organizations, list)
        and any(_normalize(str(name)) in normalized_message for name in organizations)
    )


def _title_from_content(path: Path, content: str) -> str:
    for line in content.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return path.stem.replace("-", " ").title()


def _load_scenario_config(root: Path) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    path = root / "normalized" / "scenario_overrides.json"
    if not path.exists():
        return {}, {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    return dict(raw.get("articles", {})), dict(raw.get("fallbacks", {}))


def _load_additional_records(root: Path) -> list[dict[str, Any]]:
    path = root / "normalized" / "scenario_overrides.json"
    if not path.exists():
        return []
    raw = json.loads(path.read_text(encoding="utf-8"))
    records = raw.get("additional_articles", [])
    return [dict(record) for record in records if isinstance(record, dict)] if isinstance(records, list) else []


def _suppressed_legacy_ids(root: Path) -> set[str]:
    suppressed = {
        legacy_id
        for scenario in load_scenarios()
        for legacy_id in scenario.legacy_ids
    }
    inventory_path = root / "v2" / "legacy_inventory.json"
    if not inventory_path.exists():
        return suppressed
    raw = json.loads(inventory_path.read_text(encoding="utf-8"))
    suppressed_statuses = {"migrated_to_v2", "merged_into_v2", "deactivated"}
    suppressed.update(
        str(item.get("legacy_id") or "")
        for item in raw.get("records", [])
        if isinstance(item, dict) and str(item.get("status") or "") in suppressed_statuses
    )
    return {item for item in suppressed if item}


def _load_matching_config() -> dict[str, Any]:
    return load_matching_config()


def _semantic_config() -> dict[str, Any]:
    raw = _load_matching_config().get("semantic_matching", {})
    return raw if isinstance(raw, dict) else {}


@lru_cache(maxsize=1)
def _ambiguous_phrase_rules() -> dict[str, dict[str, Any]]:
    raw = _load_matching_config().get("ambiguous_phrases", {})
    if not isinstance(raw, dict):
        return {}
    return {
        _normalize(str(phrase)): rule
        for phrase, rule in raw.items()
        if _normalize(str(phrase)) and isinstance(rule, dict)
    }


def _ambiguous_phrase_rule(message: str) -> dict[str, Any] | None:
    rules = _ambiguous_phrase_rules()
    text = _normalize(message)
    if is_ticket_creation_request(message):
        return rules.get("создать обращение")
    exact_rule = rules.get(text)
    if exact_rule:
        return exact_rule

    if re.fullmatch(r"(?:где|куда|что с) (?:мои )?(?:деньги|деньгами|бабки)", text):
        return rules.get("где деньги")
    if "штраф" in text and "депозит" in text:
        return rules.get("вопрос по штрафу или депозиту")

    has_bid_cancellation = bool(
        re.search(r"\bставк\w*\b", text)
        and re.search(r"\b(?:отмен\w*|удал\w*|снят\w*|сним\w*|аннулир\w*)\b", text)
    )
    if has_bid_cancellation and not _bid_cancellation_route(message):
        return rules.get("отменить ставку")
    return None


def is_ticket_creation_request(message: str) -> bool:
    return bool(TICKET_REQUEST_PATTERN.fullmatch(_normalize(message)))


def has_ambiguous_phrase_rule(message: str) -> bool:
    return _ambiguous_phrase_rule(message) is not None


@lru_cache(maxsize=1)
def _canonical_phrase_rules() -> dict[str, str]:
    raw = _load_matching_config().get("canonical_phrases", {})
    if not isinstance(raw, dict):
        return {}
    legacy_targets: dict[str, list[Any]] = {}
    for scenario in load_scenarios():
        for legacy_id in scenario.legacy_ids:
            legacy_targets.setdefault(legacy_id, []).append(scenario)

    def resolve_target(phrase: str, article_id: str) -> str:
        candidates = legacy_targets.get(article_id, [])
        if not candidates:
            return article_id
        if len(candidates) == 1:
            return candidates[0].scenario_id
        phrase_terms = {item for item in tokenize(correct_typos(phrase)) if item not in STOPWORDS}

        def relevance(scenario) -> tuple[float, int]:
            best = 0.0
            best_overlap = 0
            for example in scenario.positive_examples:
                example_terms = {item for item in tokenize(correct_typos(example)) if item not in STOPWORDS}
                overlap = len(phrase_terms & example_terms)
                coverage = overlap / max(1, len(phrase_terms | example_terms))
                if (coverage, overlap) > (best, best_overlap):
                    best, best_overlap = coverage, overlap
            return best, best_overlap

        return max(candidates, key=relevance).scenario_id

    return {
        _normalize(str(phrase)): resolve_target(str(phrase), str(article_id))
        for phrase, article_id in raw.items()
        if _normalize(str(phrase)) and str(article_id)
    }


@lru_cache(maxsize=1)
def _canonical_token_rules() -> dict[tuple[str, ...], str]:
    """Order-insensitive canonical matches for short telegraphic questions."""
    grouped: dict[tuple[str, ...], set[str]] = {}
    for phrase, article_id in _canonical_phrase_rules().items():
        key = tuple(sorted(tokenize(correct_typos(phrase))))
        if 2 <= len(key) <= 7:
            grouped.setdefault(key, set()).add(article_id)
    return {
        key: next(iter(article_ids))
        for key, article_ids in grouped.items()
        if len(article_ids) == 1
    }


def _phrase_is_active(
    article_id: str,
    phrase: str,
    ambiguous_phrases: set[str],
    canonical_phrases: dict[str, str],
) -> bool:
    normalized_phrase = _normalize(phrase)
    if not normalized_phrase or normalized_phrase in ambiguous_phrases:
        return False
    canonical_article = canonical_phrases.get(normalized_phrase)
    return canonical_article is None or canonical_article == article_id


def load_fallbacks() -> dict[str, str]:
    root = get_settings().knowledge_root
    _, fallbacks = _load_scenario_config(root)
    return fallbacks


@lru_cache(maxsize=1)
def load_articles() -> list[KnowledgeArticle]:
    root = get_settings().knowledge_root
    if not root.exists():
        return []

    normalized_path = root / "normalized" / "migtorg_knowledge_base.json"
    if normalized_path.exists():
        return [*_load_v2_articles(), *_load_normalized_articles(root, normalized_path)]

    articles: list[KnowledgeArticle] = []
    for path in root.glob("*/*.md"):
        section = path.parent.name
        if section == "internal_rules":
            continue
        content = path.read_text(encoding="utf-8")
        articles.append(
            KnowledgeArticle(
                title=_title_from_content(path, content),
                slug=path.stem,
                section=section,
                content=content,
                user_phrases=[_title_from_content(path, content)],
            )
        )
    return articles


def _load_v2_articles() -> list[KnowledgeArticle]:
    articles: list[KnowledgeArticle] = []
    for scenario in load_scenarios():
        has_ticket_action = any(str(item.get("type") or "") == "open_ticket" for item in scenario.actions)
        clarification_only = bool(
            len(scenario.actions) >= 2
            and all(str(item.get("type") or "") == "clarify" for item in scenario.actions)
        )
        operational_issue = bool(
            set(scenario.states) & {"error", "unavailable", "not_visible", "missing", "blocked", "no_response"}
        )
        content = (
            f"# {scenario.title}\n\n"
            f"## Утвержденные факты\n{' '.join(scenario.facts)}\n\n"
            f"## Ответ\n{scenario.answer}\n"
        )
        articles.append(
            KnowledgeArticle(
                title=scenario.title,
                slug=scenario.scenario_id,
                section="guest",
                content=content,
                intent=scenario.intent,
                user_answer=scenario.answer,
                problem="; ".join(scenario.positive_examples),
                internal_action="",
                needs_ticket=has_ticket_action and operational_issue,
                scenario=scenario.scenario_id,
                user_phrases=list(scenario.positive_examples),
                negative_phrases=list(scenario.negative_examples),
                priority=300,
                fallback_allowed=True,
                required_fields=[str(item) for item in scenario.escalation.get("required_fields", [])],
                answer_type="scenario",
                action="clarify" if clarification_only else "answer",
                trigger_phrases=list(scenario.positive_examples),
                keywords=[*scenario.objects, *scenario.operations, *scenario.states],
                negative_keywords=list(scenario.negative_examples),
            )
        )
    return articles


def _phrases_from_problem(problem: str) -> list[str]:
    quoted = re.findall(r"[«\"]([^»\"]+)[»\"]", problem)
    if quoted:
        return [phrase.strip() for phrase in quoted if phrase.strip()]
    return [problem.strip()] if problem.strip() else []


def _polish_user_answer(answer: str) -> str:
    text = re.sub(r"\s+", " ", answer).strip()
    replacements = {
        "По учебной логике": "По общему порядку",
        "по учебной логике": "по общему порядку",
        "по учебной инструкции": "по инструкции площадки",
        "в учебных материалах указан": "указан",
        "и WhatsApp/MAX/Telegram +7 (926) 511-43-99": "",
        "WhatsApp/MAX/Telegram +7 (926) 511-43-99": "",
        "Попросите ссылку или номер лота, скриншот, браузер/устройство и время проблемы. Если проблема повторяется, передайте в техподдержку.": "Для проверки проблемы укажите ссылку или номер лота, скриншот, браузер или устройство и время, когда ошибка повторилась. Я помогу подготовить обращение в техническую поддержку.",
        "Бот не заказывает": "Я не заказываю",
        "бот не заказывает": "я не заказываю",
        "Бот не подтверждает": "Я не подтверждаю",
        "бот не подтверждает": "я не подтверждаю",
        "Бот не должен обещать": "Я не могу обещать",
        "бот не должен обещать": "я не могу обещать",
        "Пользователю нужно": "Вам нужно",
        "пользователю нужно": "вам нужно",
        "Пользователь должен": "Вам нужно",
        "пользователь должен": "вам нужно",
        "Передать вопрос ответственному специалисту": "Я помогу передать вопрос ответственному специалисту",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    sentences = []
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        lower = sentence.lower().strip()
        if not lower:
            continue
        if any(lower.startswith(marker) for marker in TECHNICAL_MARKERS):
            continue
        sentences.append(sentence.strip())
    return " ".join(sentences).strip() or text


def _load_normalized_articles(root: Path, path: Path) -> list[KnowledgeArticle]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    overrides, _ = _load_scenario_config(root)
    ambiguous_phrases = set(_ambiguous_phrase_rules())
    canonical_phrases = _canonical_phrase_rules()
    suppressed_legacy_ids = _suppressed_legacy_ids(root)
    articles: list[KnowledgeArticle] = []
    for record in [*raw.get("records", []), *_load_additional_records(root)]:
        status = str(record.get("status") or "active").casefold()
        if status != "active":
            continue
        article_id = str(record.get("id", ""))
        if article_id in suppressed_legacy_ids:
            continue
        override = overrides.get(article_id, {})
        topic = str(record.get("topic", ""))
        problem = str(record.get("problem", ""))
        user_answer = str(override.get("user_answer") or record.get("user_answer", ""))
        content = f"# {topic}\n\n## Проблема\n{problem}\n\n## Ответ\n{user_answer}\n"
        phrases = [
            *[str(item) for item in override.get("user_phrases", [])],
            *[str(item) for item in record.get("user_phrases", [])],
            *[str(item) for item in record.get("trigger_phrases", [])],
            *_phrases_from_problem(problem),
            topic,
        ]
        active_phrases = [
            phrase
            for phrase in phrases
            if _phrase_is_active(article_id, phrase, ambiguous_phrases, canonical_phrases)
        ]
        active_triggers = [
            str(item)
            for item in record.get("trigger_phrases", [])
            if _phrase_is_active(article_id, str(item), ambiguous_phrases, canonical_phrases)
        ]
        action = str(record.get("action", ""))
        needs_ticket = bool(override.get("needs_ticket", record.get("needs_ticket", False))) or action in {
            "create_ticket",
            "show_document_and_offer_ticket",
        }
        articles.append(
            KnowledgeArticle(
                title=str(record.get("title") or topic),
                slug=article_id,
                section=str(override.get("visibility") or record.get("visibility", "guest")),
                content=content,
                intent=str(override.get("intent") or record.get("intent", "unknown")),
                user_answer=_polish_user_answer(user_answer),
                problem=problem,
                internal_action=str(record.get("internal_action", "")),
                needs_ticket=needs_ticket,
                safety_flags=list(record.get("safety_flags", [])),
                template=record.get("template"),
                channel=str(record.get("channel", "")),
                scenario=str(override.get("scenario", "")),
                user_phrases=list(dict.fromkeys(active_phrases)),
                negative_phrases=[str(item) for item in override.get("negative_phrases", [])],
                priority=int(override.get("priority", record.get("priority", 0))),
                fallback_allowed=bool(override.get("fallback_allowed", True)),
                required_fields=[
                    str(item) for item in (override.get("required_fields") or record.get("required_fields", []))
                ],
                answer_type=str(record.get("answer_type", "")),
                action=action,
                trigger_phrases=active_triggers,
                keywords=[str(item) for item in record.get("keywords", [])],
                negative_keywords=[str(item) for item in record.get("negative_keywords", [])],
                page_type_boost=[str(item) for item in record.get("page_type_boost", [])],
            )
        )
    return articles


@lru_cache(maxsize=1)
def _prepared_articles() -> dict[str, PreparedArticle]:
    prepared: dict[str, PreparedArticle] = {}
    for article in load_articles():
        normalized_haystack = _normalize(
            f"{article.slug} {article.title} {article.problem} {article.content} "
            f"{' '.join(article.user_phrases)} {' '.join(article.trigger_phrases)} {' '.join(article.keywords)}"
        )
        normalized_negative_phrases = tuple(
            normalized
            for phrase in [*article.negative_phrases, *article.negative_keywords]
            if (normalized := _normalize(phrase))
        )
        prepared[article.slug] = PreparedArticle(
            normalized_title=_normalize(article.title),
            normalized_problem=_normalize(article.problem),
            normalized_haystack=normalized_haystack,
            haystack_words=frozenset(_tokens(normalized_haystack)),
            normalized_negative_phrases=normalized_negative_phrases,
        )
    return prepared


@lru_cache(maxsize=1)
def _semantic_index() -> MultilingualHybridSemanticIndex:
    return MultilingualHybridSemanticIndex(load_articles(), _semantic_config())


def clear_knowledge_cache() -> None:
    """Reload knowledge and phrase routing rules after an explicit content update."""
    _prepared_articles.cache_clear()
    _semantic_index.cache_clear()
    load_articles.cache_clear()
    _ambiguous_phrase_rules.cache_clear()
    _canonical_phrase_rules.cache_clear()
    _canonical_token_rules.cache_clear()
    _synonym_token_groups.cache_clear()
    clear_scenario_cache()


def warm_knowledge_indexes() -> None:
    load_articles()
    _prepared_articles()
    if bool(_semantic_config().get("enabled", False)):
        _semantic_index()


def get_article_by_id(article_id: str, role: UserRole) -> KnowledgeArticle | None:
    allowed = {"public", "guest"} if role == "guest" else {"public", "guest", "authorized"}
    return next(
        (
            article
            for article in load_articles()
            if article.slug == article_id and article.section in allowed
        ),
        None,
    )


def _phrase_score(message: str, phrase: str) -> int:
    normalized_message = _normalize(message)
    normalized_phrase = _normalize(phrase)
    if not normalized_phrase:
        return 0
    if normalized_phrase in normalized_message:
        return 150
    matched, match_type = phrase_matches(message, phrase)
    if matched:
        if len(tokenize(normalized_phrase)) == 1 and match_type in {"synonym", "fuzzy"}:
            return 28
        return 90 if match_type in {"synonym", "fuzzy"} else 120
    phrase_tokens = set(_tokens(normalized_phrase))
    if not phrase_tokens:
        return 0
    message_tokens = set(_tokens(normalized_message))
    overlap = len(phrase_tokens & message_tokens)
    if overlap == len(phrase_tokens) and len(phrase_tokens) >= 2:
        return 90
    if overlap >= 2:
        return 28 + overlap * 7
    return 0


def _has_negative_phrase(
    article: KnowledgeArticle,
    message: str,
    normalized_message: str | None = None,
) -> bool:
    normalized_message = normalized_message or _normalize(message)
    prepared = _prepared_articles().get(article.slug)
    if prepared:
        return any(phrase in normalized_message for phrase in prepared.normalized_negative_phrases)
    negative_items = [*article.negative_phrases, *article.negative_keywords]
    return any(_normalize(phrase) in normalized_message for phrase in negative_items if _normalize(phrase))


def _context_value(context: Any | None, field: str) -> str:
    if context is None:
        return ""
    return str(getattr(context, field, "") or "")


def _has_any_phrase(message: str, phrases: list[str]) -> bool:
    normalized_message = _normalize(message)
    return any(_normalize(phrase) in normalized_message for phrase in phrases if _normalize(phrase))


def _is_vehicle_ownership_query(message: str) -> bool:
    text = _normalize(message)
    vehicle = r"(?:авто|автомобил\w*|машин\w*|тачк\w*|транспортн\w*\s+средств\w*|тс)"
    return bool(
        re.search(rf"\b(?:все\s+)?{vehicle}\s+ваш\w*\b", text)
        or re.search(rf"\bваш\w*\s+{vehicle}\b", text)
        or re.search(rf"\bчьи\s+{vehicle}\b", text)
        or re.search(rf"\b{vehicle}\s+чьи\b", text)
        or re.search(rf"\bкому\s+принадлеж\w*\s+{vehicle}\b", text)
        or (re.search(rf"\b{vehicle}\b", text) and re.search(r"\b(?:принадлеж\w*|владел\w*|собственност\w*)\b", text))
    )


def _insurer_listed_owner_vehicle_route(message: str) -> str:
    text = _normalize(message)
    vehicle = r"(?:авто|автомобил\w*|машин\w*|тачк\w*|транспортн\w*\s+средств\w*|тс)"
    has_insurer = bool(re.search(r"\b(?:страхов\w*|ск)\b", text))
    has_listing = bool(re.search(r"\b(?:выстав\w*|размест\w*|опубликов\w*)\b", text))
    has_owned_vehicle = bool(
        re.search(
            rf"\b(?:мой|моя|мое|моё|мою|мои|моего|моей)\s+{vehicle}\b",
            text,
        )
    )
    if has_insurer and has_listing and has_owned_vehicle:
        return "insurer_listed_owner_vehicle"
    return ""


def _vehicle_inspection_purpose_route(message: str) -> str:
    text = _normalize(message)
    has_purpose = bool(
        re.search(r"\b(?:зачем|для\s+чего|почему)\b", text)
        or re.search(r"\bчто\s+да[её]т\b", text)
    )
    has_inspection = bool(
        re.search(r"\b(?:смотр\w*|осмотр\w*|провер\w*)\b", text)
    )
    has_vehicle = bool(
        re.search(r"\b(?:авто|автомобил\w*|машин\w*|тачк\w*|тс)\b", text)
    )
    if has_purpose and has_inspection and (has_vehicle or "осмотр" in text):
        return "vehicle_inspection_purpose"
    return ""


def _bid_cancellation_route(message: str) -> str:
    text = _normalize(message)
    has_bid = bool(re.search(r"\bставк\w*\b", text))
    has_cancellation = bool(
        re.search(r"\b(?:отмен\w*|удал\w*|снят\w*|сним\w*|аннулир\w*)\b", text)
    )
    if not (has_bid and has_cancellation):
        return ""

    auction_completed = bool(
        re.search(
            r"\b(?:торг\w*|аукцион\w*)\b.*\b(?:заверш\w*|окончен\w*|законч\w*|состоял\w*|прошл\w*)\b",
            text,
        )
        or re.search(
            r"\b(?:заверш\w*|окончен\w*|законч\w*|состоял\w*|прошл\w*)\b.*\b(?:торг\w*|аукцион\w*)\b",
            text,
        )
        or re.search(r"\bпосле\s+(?:торг\w*|аукцион\w*)\b", text)
    )
    if auction_completed:
        return "bid_cancellation_completed"
    if re.search(r"\bзакрыт\w*\s+(?:торг\w*|аукцион\w*)\b", text) or re.search(
        r"\b(?:торг\w*|аукцион\w*)\s+закрыт\w*\b", text
    ):
        return "bid_cancellation_closed_active"
    if re.search(r"\bоткрыт\w*\s+(?:торг\w*|аукцион\w*)\b", text) or re.search(
        r"\b(?:торг\w*|аукцион\w*)\s+открыт\w*\b", text
    ):
        return "bid_cancellation_open_active"
    return ""


def _bid_adjustment_route(message: str) -> str:
    text = _normalize(message)
    has_bid = bool(re.search(r"\bставк\w*\b", text))
    if not has_bid:
        return ""

    has_change = bool(re.search(r"\b(?:измен\w*|поменя\w*|поменять|редактир\w*)\b", text))
    has_reduce = bool(re.search(r"\b(?:уменьш\w*|сниз\w*|меньш\w*)\b", text))
    if not (has_change or has_reduce):
        return ""

    auction_completed = bool(
        re.search(
            r"\b(?:торг\w*|аукцион\w*)\b.*\b(?:заверш\w*|окончен\w*|законч\w*|состоял\w*|прошл\w*)\b",
            text,
        )
        or re.search(
            r"\b(?:заверш\w*|окончен\w*|законч\w*|состоял\w*|прошл\w*)\b.*\b(?:торг\w*|аукцион\w*)\b",
            text,
        )
        or re.search(r"\bпосле\s+(?:торг\w*|аукцион\w*)\b", text)
    )
    if auction_completed:
        return "bid_cancellation_completed"
    if has_reduce:
        return "bid_reduction_active"
    return "bid_change_active"


def _auction_format_definition_route(message: str) -> str:
    text = _normalize(message)
    tokens = tokenize(text)
    if len(tokens) > 6:
        return ""

    if re.search(r"\b(?:ставк\w*|отмен\w*|удал\w*|измен\w*|уменьш\w*|сниз\w*)\b", text):
        return ""

    has_open_format = bool(
        re.search(r"\bоткрыт\w*\s+(?:торг\w*|аукцион\w*)\b", text)
        or re.search(r"\b(?:торг\w*|аукцион\w*)\s+открыт\w*\b", text)
    )
    if has_open_format:
        return "open_auction_definition"

    has_closed_format = bool(
        re.search(r"\bзакрыт\w*\s+(?:торг\w*|аукцион\w*)\b", text)
        or re.search(r"\b(?:торг\w*|аукцион\w*)\s+закрыт\w*\b", text)
    )
    if has_closed_format:
        return "closed_auction_definition"

    return ""


def _bid_access_problem_route(message: str) -> str:
    text = _normalize(message)
    has_bid = bool(re.search(r"\bставк\w*\b", text))
    if not has_bid:
        return ""
    if re.search(r"\bнедоступ\w*\b", text):
        return "bid_unavailable_demo"
    if re.search(r"\bзаблокир\w*\b", text):
        return "bid_blocked_tariff"
    return ""


def _bid_outcome_route(message: str) -> str:
    text = _normalize(message)
    has_bid = bool(re.search(r"\bставк\w*\b", text))
    if not has_bid:
        return ""
    if re.search(r"\b(?:перебил\w*|перебит\w*|не\s+выигрыв\w*|не\s+побежд\w*)\b", text):
        return "bid_outbid"
    if re.search(r"\b(?:ради|для)\s+(?:просмотр\w*|аналитик\w*)\b", text) or re.search(
        r"\bпосмотр\w*\s+(?:победн\w*\s+)?ставк\w*\b", text
    ):
        return "bid_for_result"
    return ""


def _inspection_access_problem_route(message: str) -> str:
    text = _normalize(message)
    has_inspection = bool(re.search(r"\b(?:осмотр\w*|площадк\w*)\b", text))
    has_access_problem = bool(
        re.search(r"\b(?:невозмож\w*|нет\s+доступ\w*|не\s+пустил\w*|не\s+попад\w*|не\s+состоял\w*)\b", text)
    )
    return "inspection_access_problem" if has_inspection and has_access_problem else ""


def _lot_refusal_route(message: str) -> str:
    text = _normalize(message)
    has_subject = bool(re.search(r"\b(?:лот\w*|авто|автомобил\w*|машин\w*|тачк\w*|тс)\b", text))
    has_refusal = bool(
        re.search(
            r"\b(?:не\s+заберу|не\s+буду\s+забир\w*|не\s+хочу\s+забир\w*|не\s+хочу\s+выкуп\w*|передумал\w*)\b",
            text,
        )
    )
    return "lot_refusal_general" if has_subject and has_refusal else ""


def _pickup_problem_route(message: str) -> str:
    text = _normalize(message)
    if re.search(r"\bлот\w*\b", text):
        return ""
    has_vehicle = bool(re.search(r"\b(?:авто|автомобил\w*|машин\w*|тачк\w*|тс)\b", text))
    has_pickup_problem = bool(
        re.search(r"\b(?:не\s+могу\s+забрат\w*|не\s+отда\w*|не\s+выда\w*|не\s+пуска\w*)\b", text)
    )
    return "pickup_problem" if has_vehicle and has_pickup_problem else ""


def _goods_bulk_purchase_route(message: str) -> str:
    text = _normalize(message)
    has_purchase = bool(re.search(r"\b(?:куп\w*|покуп\w*|забрат\w*)\b", text))
    has_goods_batch = bool(re.search(r"\b(?:оптом|парт\w*|товар\w*|част\w*)\b", text))
    has_seller_action = bool(re.search(r"\b(?:прод\w*|размест\w*|вылож\w*|выстав\w*)\b", text))
    return "goods_bulk_purchase" if has_purchase and has_goods_batch and not has_seller_action else ""


def _seller_item_placement_route(message: str) -> str:
    text = _normalize(message)
    has_item = bool(re.search(r"\b(?:товар\w*|имуществ\w*|парт\w*|объект\w*)\b", text))
    has_placement = bool(re.search(r"\b(?:размещ\w*|размест\w*|продав\w*|выстав\w*|опубликов\w*)\b", text))
    has_purchase = bool(re.search(r"\b(?:куп\w*|покуп\w*|забрат\w*)\b", text))
    return "seller_item_placement" if has_item and has_placement and not has_purchase else ""


def _winner_ownership_route(message: str) -> str:
    text = _normalize(message)
    has_winner = bool(re.search(r"\b(?:победител\w*|победил\w*|выиграл\w*)\b", text))
    subject = r"(?:лот\w*|авто|автомобил\w*|машин\w*|тачк\w*)"
    mine = r"(?:мой|моя|мое|моё|мои|моим)"
    has_possession = bool(
        re.search(rf"\b{subject}\s+{mine}\b", text)
        or re.search(rf"\b{mine}\s+{subject}\b", text)
    )
    return "winner_ownership_explanation" if has_winner and has_possession else ""


def _site_action_error_route(message: str) -> str:
    text = _normalize(message)
    has_action_target = bool(re.search(r"\b(?:кнопк\w*|ставк\w*)\b", text))
    has_site_error = bool(re.search(r"\b(?:ошибк\w*)\b", text) and re.search(r"\b(?:сайт\w*|страниц\w*)\b", text))
    has_action_error = bool(re.search(r"\b(?:ошибк\w*|не\s+работ\w*|не\s+отправ\w*|не\s+нажим\w*)\b", text))
    if (has_action_target and has_action_error) or has_site_error:
        return "site_action_error"
    return ""


def _document_visit_route(message: str) -> str:
    text = _normalize(message)
    has_visit = bool(re.search(r"\b(?:приед\w*|приехат\w*|заехат\w*)\b", text))
    has_documents = bool(re.search(r"\bдокумент\w*\b", text))
    return "office_documents_visit" if has_visit and has_documents else ""


def _tariff_selection_route(message: str) -> str:
    text = _normalize(message)
    if re.search(r"\b(?:поток\s+лот\w*|много\s+лот\w*|покуп\w*\s+регуляр\w*)\b", text):
        return "premium_tariff_overview"
    if re.search(r"\bне\s+хочу\s+много\s+торг\w*\b", text):
        return "one_purchase_tariff"
    return ""


def _contract_refund_route(message: str) -> str:
    text = _normalize(message)
    has_contract_close = bool(re.search(r"\b(?:расторг\w*|закрыт\w*|закрыть|прекрат\w*)\b", text))
    has_contract = bool(re.search(r"\bдоговор\w*\b", text))
    has_refund_or_write = bool(re.search(r"\b(?:верн\w*|возврат\w*|деньг\w*|депозит\w*|куда\s+писат\w*)\b", text))
    return "deposit_refund_contract" if has_contract and has_contract_close and has_refund_or_write else ""


def _seller_contract_termination_route(message: str) -> str:
    text = _normalize(message)
    has_seller = bool(re.search(r"\b(?:продав\w*|поставщик\w*|собственник\w*)\b", text))
    has_contract_close = bool(re.search(r"\b(?:расторг\w*|прекрат\w*|закрыт\w*|закрыть)\b", text))
    has_contract = bool(re.search(r"\bдоговор\w*\b", text))
    return "seller_contract_termination" if has_seller and has_contract and has_contract_close else ""


def _property_category_from_query(message: str, intent: str) -> str:
    if intent not in {"unknown", "platform"}:
        return ""

    text = _normalize(message)
    patterns = (
        r"^(?:а\s+)?(?:у\s+вас\s+)?(?:есть\s+ли|есть|бывает\s+ли|бывает|бывают\s+ли|бывают|продаются\s+ли|продаются|продаете|найдутся\s+ли|найдутся)\s+(.+)$",
        r"^(?:а\s+)?(.+?)\s+(?:есть|бывает|бывают|продаются|продаете)$",
        r"^(?:а\s+)?(?:где|ищу|хочу\s+купить|можно\s+купить|как\s+найти)\s+(.+)$",
    )
    category = ""
    for pattern in patterns:
        match = re.fullmatch(pattern, text)
        if match:
            category = match.group(1).strip()
            break
    if not category:
        return ""

    category = re.sub(r"^(?:а|ну|скажите|подскажите)\s+", "", category).strip()
    category = re.sub(r"^(?:посмотреть|найти|купить)\s+", "", category).strip()
    category_tokens = tokenize(category)
    if not category_tokens or len(category_tokens) > 7:
        return ""

    vehicle_only = {
        "авто",
        "автомобиль",
        "автомобили",
        "машина",
        "машины",
        "тачка",
        "тачки",
        "тс",
        "транспорт",
        "транспортное средство",
        "транспортные средства",
        "годные остатки",
        "годные остатки транспортных средств",
        "готс",
    }
    category_without_owner = re.sub(
        r"^(?:все\s+)?(?:мой|моя|мое|моё|мою|мои|моего|моей|наш|наша|наше|наши)\s+",
        "",
        category,
    ).strip()
    if category in vehicle_only or category_without_owner in vehicle_only:
        return ""

    service_stems = (
        "деньг",
        "платеж",
        "оплат",
        "возврат",
        "депозит",
        "тариф",
        "премиум",
        "доступ",
        "штраф",
        "комисс",
        "скидк",
        "ставк",
        "торг",
        "лот",
        "регистрац",
        "аккаунт",
        "кабинет",
        "парол",
        "документ",
        "договор",
        "акт",
        "птс",
        "дкп",
        "vin",
        "выдач",
        "передач",
        "стоянк",
        "осмотр",
        "сотрудник",
        "специалист",
        "поддержк",
        "офис",
        "адрес",
        "телефон",
        "контакт",
        "продав",
        "поставщик",
        "баланс",
        "счет",
        "чек",
        "заявлен",
        "шаблон",
        "страхов",
        "доставк",
        "гарант",
        "срок",
        "цен",
        "стоимост",
        "сайт",
        "бот",
        "обучен",
        "интерфейс",
        "результат",
        "итог",
        "статус",
        "уведомлен",
        "истори",
        "правил",
        "отображ",
        "показыв",
        "публик",
    )
    filler_tokens = {"для", "по", "на", "в", "во", "с", "со", "и", "или", "мой", "моя", "мои", "ваш", "ваша", "ваши"}
    meaningful_tokens = [token for token in category_tokens if token not in filler_tokens]
    if meaningful_tokens and all(
        any(token.startswith(stem) for stem in service_stems)
        for token in meaningful_tokens
    ):
        return ""
    return category


def _insurance_section_overview_route(message: str) -> str:
    text = _normalize(message)
    sections = [section for section in ("каско", "осаго") if section in tokenize(text)]
    if len(sections) != 1:
        return ""

    section = sections[0]
    route_name = {
        "каско": "kasko_section_overview",
        "осаго": "osago_section_overview",
    }[section]
    other_tokens = [token for token in tokenize(text) if token != section]
    if not other_tokens:
        return route_name

    overview_stems = (
        "что",
        "это",
        "так",
        "раздел",
        "категор",
        "про",
        "означ",
        "знач",
        "расскаж",
        "объясн",
        "расшифр",
        "покаж",
        "где",
        "найт",
        "посмотр",
        "откр",
        "сайт",
        "мигторг",
        "migtorg",
        "какие",
        "какой",
        "лот",
        "прода",
        "есть",
        "страхов",
        "на",
        "за",
        "в",
    )
    if len(other_tokens) <= 7 and all(
        any(token.startswith(stem) for stem in overview_stems)
        for token in other_tokens
    ):
        return route_name
    return ""


def _technical_failure_route(message: str) -> str:
    text = _normalize(message)
    has_failure = bool(
        re.search(
            r"\b(?:не\s+работ\w*|не\s+откры\w*|не\s+груз\w*|не\s+загруж\w*|"
            r"не\s+прогруж\w*|не\s+реагир\w*|перестал\w*\s+реагир\w*|не\s+ищ\w*|"
            r"слом\w*|завис\w*|висит|туп\w*|лаг\w*|лежит|лег|"
            r"бел\w*\s+экран\w*|пуст\w*\s+экран\w*|"
            r"экран\w*\b.{0,40}\bпуст\w*|пуст\w*\b.{0,40}\bэкран\w*)\b",
            text,
        )
    )
    if not has_failure:
        return ""

    has_property_section = bool(
        re.search(
            r"\b(?:(?:раздел|каталог)\w*\s+)?имуществ\w*\b|\bproperty\b",
            text,
        )
    )
    if has_property_section:
        return "property_section_failure"

    has_search = bool(re.search(r"\bпоиск\w*\b", text))
    has_filter = bool(re.search(r"\b(?:фильтр\w*|сортиров\w*)\b", text))
    if has_search and has_filter:
        return "search_filter_failure"
    if has_search:
        return "search_failure"
    if has_filter:
        return "filter_failure"

    has_site = bool(
        re.search(
            r"\b(?:сайт\w*|площадк\w*|страниц\w*|интерфейс\w*|экран\w*|карточк\w*|"
            r"мобильн\w*|телефон\w*|смартфон\w*|планшет\w*|айфон\w*|iphone\w*|android\w*)\b",
            text,
        )
    )
    broad_failure = bool(
        re.fullmatch(
            r"(?:у\s+меня\s+)?(?:вообще\s+)?(?:ниче|ничего)\s+"
            r"(?:не\s+работ\w*|неработ\w*|слом\w*|завис\w*|висит|лаг\w*)",
            text,
        )
        or re.fullmatch(
            r"(?:у\s+меня\s+)?(?:не\s+работ\w*|неработ\w*|слом\w*|завис\w*|висит|лаг\w*)\s+"
            r"(?:вообще\s+)?(?:ниче|ничего)",
            text,
        )
    )
    if has_site or broad_failure:
        return "site_failure"
    return ""


def _vehicle_transfer_delay_route(message: str) -> str:
    text = _normalize(message)
    has_lot = bool(re.search(r"\bлот\w*\b", text))
    has_lot_or_vehicle = bool(
        re.search(r"\b(?:лот\w*|авто|автомобил\w*|машин\w*|тачк\w*)\b", text)
    )
    has_transfer_delay = bool(
        re.search(r"\bне\s+переда\w*\b", text)
        or (has_lot and re.search(r"\bне\s+отда\w*\b", text))
        or (has_lot and re.search(r"\bне\s+выда\w*\b", text))
        or re.search(r"\bкогда\s+переда\w*\b", text)
        or re.search(r"\b(?:долго|сколько|жду|задерж\w*)\b.*\bпередач\w*\b", text)
        or re.search(r"\bпередач\w*\b.*\b(?:долго|задерж\w*|нет|не\s+подтвержд\w*)\b", text)
    )
    return "vehicle_transfer_delay" if has_lot_or_vehicle and has_transfer_delay else ""


def _named_seller_contact_delay_route(message: str) -> str:
    text = _normalize(message)
    organizations = _load_matching_config().get("domain_organizations", [])
    named_organization = bool(
        isinstance(organizations, list)
        and any(_normalize(str(name)) in text for name in organizations)
    )
    generic_seller = bool(re.search(r"\b(?:страхов\w*|продав\w*|менеджер\w*)\b", text))
    if not named_organization and not generic_seller:
        return ""
    has_silence = bool(
        re.search(
            r"\b(?:молчит|тишин\w*|игнор\w*|гасит\w*|гасится|пропал\w*|"
            r"не\s+отвеч\w*|не\s+выходит\w*(?:\s+на\s+связь)?)\b",
            text,
        )
    )
    return "seller_contact_delay" if has_silence else ""


def _beginner_participation_route(message: str) -> str:
    text = _normalize(message)
    asks_how = bool(re.search(r"\b(?:как|с\s+чего|хочу\s+начат\w*|что\s+нужно)\b", text))
    participation = bool(
        re.search(r"\b(?:участв\w*|аукцион\w*|торг\w*|ставк\w*)\b", text)
        or (
            re.search(r"\b(?:куп\w*|покуп\w*)\b", text)
            and re.search(r"\b(?:авто|автомобил\w*|машин\w*|лот\w*)\b", text)
        )
    )
    return "beginner_participation" if asks_how and participation else ""


def _inspection_before_bidding_route(message: str) -> str:
    text = _normalize(message)
    has_vehicle = bool(re.search(r"\b(?:авто|автомобил\w*|машин\w*|тачк\w*|тс)\b", text))
    has_view = bool(re.search(r"\b(?:осмотр\w*|посмотр\w*|увид\w*|провер\w*)\b", text))
    before = bool(re.search(r"\b(?:до|перед)\s+(?:ставк\w*|торг\w*|аукцион\w*|покупк\w*)\b", text))
    return "inspection_before_bidding" if has_vehicle and has_view and before else ""


def _pickup_location_after_win_route(message: str) -> str:
    text = _normalize(message)
    has_vehicle = bool(re.search(r"\b(?:авто|автомобил\w*|машин\w*|тачк\w*|лот\w*)\b", text))
    has_win_or_pickup = bool(re.search(r"\b(?:выигр\w*|побед\w*|забрат\w*|получ\w*|выдач\w*)\b", text))
    asks_location = bool(re.search(r"\b(?:куда\s+ехат\w*|где\s+забрат\w*|адрес\w*|место\s+выдач\w*)\b", text))
    return "pickup_location_after_win" if has_vehicle and has_win_or_pickup and asks_location else ""


def _deal_documents_delay_route(message: str) -> str:
    text = _normalize(message)
    has_documents = bool(re.search(r"\b(?:договор\w*|дкп|счет\w*|счёт\w*|документ\w*)\b", text))
    missing = bool(re.search(r"\b(?:не\s+прислал\w*|не\s+пришел\w*|не\s+пришёл\w*|не\s+приход\w*|нет|жду)\b", text))
    return "deal_documents_delay" if has_documents and missing else ""


def _payment_timing_route(message: str) -> str:
    text = _normalize(message)
    has_payment = bool(re.search(r"\b(?:когда\s+плат\w*|когда\s+оплат\w*|платить\s+сейчас|срок\w*\s+оплат\w*)\b", text))
    has_lot = bool(re.search(r"\b(?:лот\w*|авто|автомобил\w*|машин\w*|выигр\w*)\b", text))
    return "payment_timing" if has_payment and has_lot else ""


def _auction_status_stale_route(message: str) -> str:
    text = _normalize(message)
    ended = bool(re.search(r"\b(?:торг\w*|аукцион\w*)\b.*\b(?:законч\w*|заверш\w*)\b", text))
    stale = bool(re.search(r"\b(?:статус\w*|не\s+меня\w*|стар\w*|без\s+изменен\w*)\b", text))
    return "auction_status_stale" if ended and stale else ""


def _vin_location_route(message: str) -> str:
    text = _normalize(message)
    asks_location = bool(re.search(r"\b(?:где|как\s+найт\w*|где\s+искать|посмотр\w*)\b", text))
    has_vin = bool(re.search(r"\b(?:vin|вин)\b", text))
    return "vin_location" if asks_location and has_vin else ""


def _vehicle_pickup_route(message: str) -> str:
    text = _normalize(message)
    if re.search(r"\b(?:информац\w*|данн\w*|сведен\w*|документ\w*|доверенност\w*|описан\w*|фото\w*)\b", text):
        return ""
    if re.search(r"\b(?:координат\w*|час\w*|адрес\w*|кто\s+пришл\w*)\b", text):
        return ""
    has_vehicle = bool(re.search(r"\b(?:авто|автомобил\w*|машин\w*|тачк\w*|тс)\b", text))
    has_get_action = bool(re.search(r"\b(?:получ\w*|забра\w*|выдач\w*)\b", text))
    has_seller_action = bool(re.search(r"\b(?:прод\w*|выстав\w*|размест\w*)\b", text))
    return "vehicle_pickup" if has_vehicle and has_get_action and not has_seller_action else ""


def _lot_transfer_request_route(message: str) -> str:
    text = _normalize(message)
    if re.search(r"\b(?:передайте|передать|отдайте|отдать|выдайте|выдать)\s+лот\w*\b", text):
        return "vehicle_transfer_delay"
    return ""


def _third_party_lot_payment_route(message: str) -> str:
    text = _normalize(message)
    has_pay = bool(re.search(r"\b(?:оплат\w*|заплат\w*|платить)\b", text))
    has_other_payer = bool(re.search(r"\b(?:друг\w*|иной|треть\w*)\b", text))
    has_for_me = bool(re.search(r"\b(?:за\s+меня|вместо\s+меня|мой|мо[её]й|моего)\b", text))
    has_tariff = bool(re.search(r"\b(?:тариф\w*|премиум|разов\w*)\b", text))
    return "third_party_lot_payment" if has_pay and has_other_payer and has_for_me and not has_tariff else ""


def _money_status_route(message: str) -> str:
    text = _normalize(message)
    if re.fullmatch(r"(?:где|куда)\s+(?:мои\s+)?(?:деньг\w*|бабк\w*|бабос\w*|лавэ)", text):
        return "money_status_query"
    if re.fullmatch(r"куда\s+делись\s+(?:деньг\w*|бабк\w*)", text):
        return "money_status_query"
    if re.fullmatch(r"что\s+с\s+(?:деньг\w*|бабк\w*|платеж\w*|возврат\w*)", text):
        return "money_status_query"
    return ""


def _short_tariff_route(message: str) -> str:
    text = _normalize(message)
    if re.fullmatch(r"(?:премиум|премиум\s+тариф|premium)", text):
        return "premium_tariff_overview"
    return ""


def _demo_mode_route(message: str) -> str:
    text = _normalize(message)
    has_demo = bool(
        re.search(r"\b(?:демо|демк\w*|demo)\b", text)
        or "ознакомительный режим" in text
    )
    if not has_demo:
        return ""

    vague_failure = bool(
        re.fullmatch(
            r"(?:у\s+меня\s+)?(?:демо(?:\s+режим)?|демк\w*|demo)\s+"
            r"(?:не\s+работ\w*|слом\w*|глюч\w*|не\s+откры\w*)",
            text,
        )
        or re.fullmatch(
            r"(?:у\s+меня\s+)?(?:не\s+работ\w*|слом\w*|глюч\w*|проблем\w*)\s+"
            r"(?:с\s+)?(?:демо(?:\s+режим\w*)?|демк\w*|demo)",
            text,
        )
    )
    if vague_failure:
        return ""

    paid_but_still_demo = bool(
        re.search(
            r"\b(?:оплат\w*|куп\w*|деньг\w*\s+спис\w*|тариф\w*\s+подключ\w*)\b",
            text,
        )
        and (
            re.search(r"\b(?:остал\w*|по\s+прежнему|все\s+еще|до\s+сих\s+пор)\b", text)
            or re.search(r"\b(?:не\s+отключ\w*|не\s+пропал\w*|не\s+сменил\w*)\b", text)
        )
    )
    if paid_but_still_demo:
        return "demo_mode_stuck_after_payment"

    has_upgrade = bool(
        re.search(r"\b(?:перейт\w*|подключ\w*|куп\w*|оплат\w*|выбр\w*)\b", text)
        and re.search(r"\b(?:разов\w*|премиум\w*|платн\w*|автомобильн\w*\s+тариф\w*)\b", text)
        or re.search(r"\b(?:отключ\w*|убрат\w*|выйт\w*)\b.*\bдемо\b", text)
        or re.search(r"\bс\s+демо\b.*\b(?:на\s+)?тариф\w*\b", text)
    )
    if has_upgrade:
        return "demo_mode_upgrade"

    if re.search(r"\b(?:результат\w*|итог\w*|победител\w*)\b", text):
        return "demo_mode_results"

    has_price_visibility = bool(
        re.search(r"\b(?:цен\w*|стоимост\w*)\b", text)
        and re.search(r"\b(?:лот\w*|торг\w*|карточк\w*|вид\w*|показ\w*|отображ\w*)\b", text)
    )
    if has_price_visibility:
        return "demo_mode_price"

    if re.search(r"\b(?:ставк\w*|ставит\w*|участв\w*|торгова\w*)\b", text):
        return "demo_mode_bidding"

    return "demo_mode_overview"


def _concept_definition_route(message: str) -> str:
    text = _normalize(message)
    tokens = tokenize(text)
    if len(tokens) > 8:
        return ""

    has_definition_marker = any(
        marker in text
        for marker in (
            "что такое",
            "что значит",
            "что означает",
            "объясните что",
            "объясни что",
        )
    ) or bool(re.search(r"\b(?:это\s+что|это\s+такое|это)$", text))
    filler_tokens = {
        "что",
        "такое",
        "значит",
        "означает",
        "объясните",
        "объясни",
        "расскажите",
        "расскажи",
        "это",
        "простыми",
        "простых",
        "словами",
        "словах",
        "пожалуйста",
        "и",
        "про",
        "по",
        "о",
        "об",
    }

    def only_concept(*concept_stems: str) -> bool:
        meaningful = [
            token
            for token in tokens
            if token not in filler_tokens
        ]
        return bool(meaningful) and all(
            any(token.startswith(stem) for stem in concept_stems)
            for token in meaningful
        )

    meaningful_tokens = [token for token in tokens if token not in filler_tokens]
    bare_short_concept_term = (
        not has_definition_marker
        and 1 <= len(meaningful_tokens) <= 2
        and any(
            all(token.startswith(stem) for token in meaningful_tokens)
            for stem in ("лот", "ставк", "имуществ", "торг", "аукцион", "котиров")
        )
    )
    if not has_definition_marker and not bare_short_concept_term:
        return ""

    if only_concept("карточк", "лот") and any(token.startswith("карточк") for token in tokens):
        return "lot_card_definition"
    if only_concept("ставк"):
        return "bid_definition"
    if only_concept("торг", "аукцион"):
        return "auction_definition"
    if only_concept("котиров"):
        return "quotation_definition"
    if only_concept("лот"):
        return "lot_definition"
    if only_concept("имуществ"):
        return "property_definition"
    return ""


def _vehicle_category_overview_route(message: str, intent: str) -> str:
    text = _normalize(message)
    tokens = tokenize(text)
    if len(tokens) > 7:
        return ""
    has_vehicle = bool(re.search(r"\b(?:авто|автомобил\w*|машин\w*|тачк\w*)\b", text))
    has_damaged_category = bool(
        re.search(r"\b(?:бит\w*|разбит\w*|поврежден\w*|повреждён\w*|аварийн\w*)\b", text)
        or "после дтп" in text
    )
    if not (has_vehicle and has_damaged_category):
        return ""

    specific_case_markers = (
        "почему",
        "мой",
        "моя",
        "мою",
        "лот",
        "документ",
        "забрат",
        "получит",
        "описан",
        "состоян",
        "откуда",
        "владел",
        "принадлеж",
    )
    if any(marker in text for marker in specific_case_markers):
        return ""
    return "damaged_vehicle_category"


def structured_route_name(message: str, intent: str) -> str:
    for route in (
        _beginner_participation_route(message),
        _inspection_before_bidding_route(message),
        _pickup_location_after_win_route(message),
        _deal_documents_delay_route(message),
        _payment_timing_route(message),
        _auction_status_stale_route(message),
        _vin_location_route(message),
    ):
        if route:
            return route
    seller_contact_delay_route = _named_seller_contact_delay_route(message)
    if seller_contact_delay_route:
        return seller_contact_delay_route
    seller_contract_route = _seller_contract_termination_route(message)
    if seller_contract_route:
        return seller_contract_route
    contract_refund_route = _contract_refund_route(message)
    if contract_refund_route:
        return contract_refund_route
    document_visit_route = _document_visit_route(message)
    if document_visit_route:
        return document_visit_route
    bid_access_problem_route = _bid_access_problem_route(message)
    if bid_access_problem_route:
        return bid_access_problem_route
    bid_outcome_route = _bid_outcome_route(message)
    if bid_outcome_route:
        return bid_outcome_route
    inspection_access_problem_route = _inspection_access_problem_route(message)
    if inspection_access_problem_route:
        return inspection_access_problem_route
    lot_refusal_route = _lot_refusal_route(message)
    if lot_refusal_route:
        return lot_refusal_route
    pickup_problem_route = _pickup_problem_route(message)
    if pickup_problem_route:
        return pickup_problem_route
    goods_bulk_purchase_route = _goods_bulk_purchase_route(message)
    if goods_bulk_purchase_route:
        return goods_bulk_purchase_route
    seller_item_placement_route = _seller_item_placement_route(message)
    if seller_item_placement_route:
        return seller_item_placement_route
    winner_ownership_route = _winner_ownership_route(message)
    if winner_ownership_route:
        return winner_ownership_route
    site_action_error_route = _site_action_error_route(message)
    if site_action_error_route:
        return site_action_error_route
    tariff_selection_route = _tariff_selection_route(message)
    if tariff_selection_route:
        return tariff_selection_route
    tariff_route = _short_tariff_route(message)
    if tariff_route:
        return tariff_route
    money_status_route = _money_status_route(message)
    if money_status_route:
        return money_status_route
    third_party_payment_route = _third_party_lot_payment_route(message)
    if third_party_payment_route:
        return third_party_payment_route
    pickup_route = _vehicle_pickup_route(message)
    if pickup_route:
        return pickup_route
    lot_transfer_request_route = _lot_transfer_request_route(message)
    if lot_transfer_request_route:
        return lot_transfer_request_route
    transfer_route = _vehicle_transfer_delay_route(message)
    if transfer_route:
        return transfer_route
    demo_route = _demo_mode_route(message)
    if demo_route:
        return demo_route
    insurer_listing_route = _insurer_listed_owner_vehicle_route(message)
    if insurer_listing_route:
        return insurer_listing_route
    inspection_purpose_route = _vehicle_inspection_purpose_route(message)
    if inspection_purpose_route:
        return inspection_purpose_route
    bid_cancellation_route = _bid_cancellation_route(message)
    if bid_cancellation_route:
        return bid_cancellation_route
    bid_adjustment_route = _bid_adjustment_route(message)
    if bid_adjustment_route:
        return bid_adjustment_route
    auction_format_route = _auction_format_definition_route(message)
    if auction_format_route:
        return auction_format_route
    if _is_vehicle_ownership_query(message):
        return "vehicle_ownership_query"
    insurance_route = _insurance_section_overview_route(message)
    if insurance_route:
        return insurance_route
    definition_route = _concept_definition_route(message)
    if definition_route:
        return definition_route
    technical_route = _technical_failure_route(message)
    if technical_route:
        return technical_route
    vehicle_category_route = _vehicle_category_overview_route(message, intent)
    if vehicle_category_route:
        return vehicle_category_route
    if _property_category_from_query(message, intent):
        return "property_category_query"
    return ""


def _is_too_broad_unknown(message: str) -> bool:
    normalized_message = _normalize(message)
    broad_problem_phrases = (
        "все сломалось",
        "ничего не работает",
        "не работает",
        "сломалось",
        "проблема",
        "ошибка",
        "не понимаю",
        "непонятн",
        "не получилось",
        "не сработ",
        "не дает продолжить",
        "не дает перейти",
        "помогите разобраться",
        "нужна консультац",
        "дальнейшие действия",
        "двигаться дальше",
        "следующий шаг",
        "что предпринять",
        "странный результат",
        "странно выглядит",
    )
    domain_markers = (
        "тариф",
        "платеж",
        "платёж",
        "лот",
        "ставк",
        "торг",
        "документ",
        "осмотр",
        "стоянк",
        "возврат",
        "штраф",
        "отказ",
        "регистрац",
        "аккаунт",
        "кабинет",
        "профиль",
        "доступ",
        "баланс",
        "карта",
        "автомоб",
        "машин",
        "имущество",
    )
    return any(phrase in normalized_message for phrase in broad_problem_phrases) and not any(
        marker in normalized_message for marker in domain_markers
    )


def _is_context_dependent_without_topic(message: str) -> bool:
    normalized_message = _normalize(message)
    tokens = set(tokenize(normalized_message))
    content_terms = set(_tokens(normalized_message))
    if _has_discriminative_term(content_terms):
        return False

    reference_tokens = {
        "это",
        "этот",
        "эта",
        "эти",
        "этим",
        "этого",
        "этому",
        "там",
        "тут",
        "здесь",
        "так",
        "такой",
        "такое",
    }
    generic_issue_stems = (
        "разобр",
        "подскаж",
        "помо",
        "правильн",
        "неверн",
        "результ",
        "ситуац",
        "проблем",
        "произош",
        "отображ",
        "сдел",
    )
    has_reference = bool(tokens & reference_tokens)
    has_only_generic_issue = any(
        term.startswith(stem)
        for term in content_terms
        for stem in generic_issue_stems
    )
    return has_reference and has_only_generic_issue and len(tokens) <= 9


@lru_cache(maxsize=1)
def _synonym_token_groups() -> tuple[frozenset[str], ...]:
    synonyms = _load_matching_config().get("synonyms", {})
    if not isinstance(synonyms, dict):
        return ()

    groups = []
    for canonical, variants in synonyms.items():
        normalized_group = {_normalize(str(canonical))}
        normalized_group.update(
            normalized
            for item in variants
            if (normalized := _normalize(str(item)))
        )
        groups.append(
            frozenset(token for phrase in normalized_group for token in _tokens(phrase))
        )
    return tuple(groups)


def _synonym_score(words: set[str], article_words: set[str]) -> int:
    score = 0
    for group_tokens in _synonym_token_groups():
        if words & group_tokens and article_words & group_tokens:
            score += 8
    return score


def _score(article: KnowledgeArticle, intent: str, message: str, context: Any | None = None) -> int:
    score, _ = _score_with_features(article, intent, message, context)
    return score


def _score_with_features(
    article: KnowledgeArticle,
    intent: str,
    message: str,
    context: Any | None = None,
    *,
    analysis: TextAnalysis | None = None,
    pattern_match: IntentPatternMatch | None = None,
    pattern_precomputed: bool = False,
    config: dict[str, Any] | None = None,
    normalized_message: str | None = None,
    words: list[str] | None = None,
) -> tuple[int, list[str]]:
    normalized_message = normalized_message or _normalize(message)
    if _has_negative_phrase(article, message, normalized_message):
        return -10_000, ["negative_keyword:-30"]

    config = config or _load_matching_config()
    analysis = analysis or analyze_text(message, context)
    words = words or _tokens(normalized_message)
    word_set = set(words)
    prepared = _prepared_articles().get(article.slug)
    title = prepared.normalized_title if prepared else _normalize(article.title)
    problem = prepared.normalized_problem if prepared else _normalize(article.problem)
    haystack = prepared.normalized_haystack if prepared else _normalize(
        f"{article.slug} {article.title} {article.problem} {article.content} "
        f"{' '.join(article.user_phrases)} {' '.join(article.trigger_phrases)} {' '.join(article.keywords)}"
    )
    haystack_words = set(prepared.haystack_words) if prepared else set(_tokens(haystack))

    score = 0
    matched_features: list[str] = []
    phrase_scores = [_phrase_score(message, phrase) for phrase in article.user_phrases]
    trigger_scores = [_phrase_score(message, phrase) for phrase in article.trigger_phrases]
    best_phrase_score = max(phrase_scores, default=0)
    if best_phrase_score:
        matched_features.append("user_phrase")
    score += best_phrase_score
    best_trigger_score = max(trigger_scores, default=0)
    if best_trigger_score:
        matched_features.append("exact_trigger_phrase:+50")
        score += best_trigger_score + 50
    matched_content_terms = {word for word in words if word in haystack_words}
    token_hits = sum(2 for word in words if word in haystack)
    title_hits = sum(8 for word in words if word in title)
    problem_hits = sum(4 for word in words if word in problem)
    matched_keywords = [keyword for keyword in article.keywords if phrase_matches(message, keyword)[0]]
    matched_keyword_terms = {term for keyword in matched_keywords for term in _tokens(keyword)}
    keyword_hits = len(matched_keywords) * 10
    score += token_hits + title_hits + problem_hits + keyword_hits
    if token_hits or title_hits or problem_hits or keyword_hits:
        matched_features.append("keyword")
        matched_features.append(f"content_terms:{len(matched_content_terms)}")
    synonym_score = _synonym_score(set(analysis.corrected_tokens) | word_set, haystack_words)
    if synonym_score:
        matched_features.append("synonym:+10")
    score += synonym_score

    has_content_evidence = score > 0
    has_strong_content_evidence = bool(
        best_phrase_score >= 90
        or best_trigger_score
        or len(matched_content_terms) >= 2
        or _has_discriminative_term(matched_content_terms | matched_keyword_terms)
        or (intent == article.intent and _has_intent_term(set(words), intent))
        or len(matched_keywords) >= 2
        or any(len(_tokens(keyword)) >= 2 for keyword in matched_keywords)
        or synonym_score >= 16
    )

    if intent != "unknown" and intent == article.intent:
        score += 35
        matched_features.append("intent_match:+35")
    elif intent != "unknown" and article.intent != "unknown":
        score -= 35

    if not pattern_precomputed:
        pattern_match = best_intent_pattern(message, context, analysis=analysis)
    if pattern_match and pattern_match.intent == article.intent:
        reliable_pattern = pattern_match.confidence_level in {"medium", "high"}
        matched_features.extend(
            f"intent_feature:{feature.group}:{feature.match_type}"
            for feature in pattern_match.matched_features
        )
        if reliable_pattern:
            has_content_evidence = True
            has_strong_content_evidence = True
            score += 35
            if pattern_match.action == "create_ticket" and article.needs_ticket:
                score += 10
        else:
            score += 5
            matched_features.append("weak_intent_pattern")

    # Context and article priority may rank relevant candidates, but they must
    # never create a match when the user's text has no connection to the article.
    if not has_content_evidence:
        return 0, []
    if not has_strong_content_evidence:
        matched_features.append("weak_content_evidence")
        return min(score, 49), list(dict.fromkeys(matched_features))

    page_type = _context_value(context, "page_type")
    if page_type and page_type in article.page_type_boost:
        score += 15
        matched_features.append("page_type:+15")
    page_boosts = config.get("page_type_boosts", {})
    page_intents = page_boosts.get(page_type, []) if isinstance(page_boosts, dict) else []
    if article.intent in page_intents:
        score += 15
        matched_features.append("page_type_intent:+15")

    problem_words = config.get("problem_words", [])
    if isinstance(problem_words, list) and _has_any_phrase(message, [str(item) for item in problem_words]):
        if article.needs_ticket or article.action in {"create_ticket", "show_document_and_offer_ticket"} or article.answer_type == "troubleshooting":
            score += 20
            matched_features.append("problem_word:+20")
        else:
            score += 5

    individual_markers = config.get("individual_markers", [])
    if isinstance(individual_markers, list) and _has_any_phrase(message, [str(item) for item in individual_markers]):
        if article.needs_ticket or article.answer_type in {"sensitive", "troubleshooting"}:
            score += 20
            matched_features.append("individual_marker")

    if _context_value(context, "lot_id") and article.intent in {"lot", "transfer", "pickup", "inspection", "refusal", "penalty"}:
        score += 25
        matched_features.append("lot_context")
    if article.intent == "payment" and re.search(r"\b\d{3,}\b", normalized_message):
        score += 20
        matched_features.append("payment_number")

    if article.template and any(word in words for word in ("форма", "шаблон", "заявление", "документ", "документы")):
        score += 35
    if article.needs_ticket and any(word in words for word in ("проверьте", "проверить", "специалист", "сотрудник", "помогите")):
        score += 25
    if article.scenario and any(word in article.scenario for word in words):
        score += 8

    score += min(article.priority, 90)

    return score, list(dict.fromkeys(matched_features))


class RuleBasedSearchProvider:
    def search(
        self,
        message: str,
        intent: str,
        role: UserRole,
        context: Any | None = None,
        analysis: TextAnalysis | None = None,
        pattern_match: IntentPatternMatch | None = None,
        skip_topic_ambiguity: bool = False,
    ) -> KnowledgeSearchResult:
        ambiguous_rule = _ambiguous_phrase_rule(message)
        if ambiguous_rule:
            options = ambiguous_rule.get("options", [])
            article_ids = ambiguous_rule.get("article_ids", [])
            intents = ambiguous_rule.get("intents", [])
            return KnowledgeSearchResult(
                article=None,
                score=0,
                confidence="medium",
                matched_features=["ambiguous_phrase"],
                fallback_reason="ambiguous_phrase",
                clarifying_question=str(ambiguous_rule.get("text", "")),
                clarifying_options=[str(option) for option in options] if isinstance(options, list) else [],
                clarifying_article_ids=(
                    [str(article_id) for article_id in article_ids]
                    if isinstance(article_ids, list)
                    else []
                ),
                clarifying_intents=(
                    [str(option_intent) for option_intent in intents]
                    if isinstance(intents, list)
                    else []
                ),
            )

        normalized_message = _normalize(message)
        canonical_article_id = _canonical_phrase_rules().get(normalized_message)
        if not canonical_article_id:
            canonical_article_id = _canonical_token_rules().get(
                tuple(sorted(tokenize(correct_typos(normalized_message))))
            )
        if canonical_article_id:
            canonical_article = get_article_by_id(canonical_article_id, role)
            if canonical_article and not _has_negative_phrase(canonical_article, message, normalized_message):
                return KnowledgeSearchResult(
                    canonical_article,
                    250,
                    "high",
                    ["canonical_phrase"],
                )

        if not _has_domain_signal(message):
            return KnowledgeSearchResult(
                article=None,
                score=0,
                confidence="low",
                matched_features=["domain_guard:out_of_scope"],
                fallback_reason="out_of_scope",
            )

        structured_route = structured_route_name(message, intent)
        topic_route = route_topic(message)
        if topic_route.ambiguous and not skip_topic_ambiguity and not structured_route:
            labels = _semantic_config().get("topic_labels", {})
            options = [
                str(labels.get(candidate, candidate))
                for candidate in topic_route.candidates
            ]
            return KnowledgeSearchResult(
                article=None,
                score=topic_route.score * 35,
                confidence="medium",
                matched_features=[
                    "topic_router_ambiguous",
                    *[f"topic_candidate:{candidate}" for candidate in topic_route.candidates],
                ],
                fallback_reason="topic_candidates_close",
                clarifying_question="Уточните, пожалуйста, к какой теме ближе ваш вопрос:",
                clarifying_options=[*options, "Другая тема"],
                clarifying_intents=[*topic_route.candidates, "unknown"],
            )
        if not skip_topic_ambiguity and not structured_route and _is_context_dependent_without_topic(message):
            return KnowledgeSearchResult(
                None,
                0,
                "low",
                ["insufficient_context"],
                "insufficient_context",
            )

        if intent == "unknown" and not structured_route and _is_too_broad_unknown(message):
            return KnowledgeSearchResult(None, 0, "low", fallback_reason="too_broad_unknown")

        allowed = {"public", "guest"} if role == "guest" else {"public", "guest", "authorized"}
        candidates = [
            article
            for article in load_articles()
            if article.section in allowed and article.intent != "prohibited"
        ]
        if not candidates:
            return KnowledgeSearchResult(None, 0, "low", fallback_reason="no_candidates")

        structured_routes = _load_matching_config().get("structured_routes", {})
        route_config = structured_routes.get(structured_route, {}) if isinstance(structured_routes, dict) else {}
        structured_article_id = str(route_config.get("article_id", "")) if isinstance(route_config, dict) else ""
        if structured_article_id:
            structured_article = next(
                (article for article in candidates if article.slug == structured_article_id),
                None,
            )
            if structured_article:
                return KnowledgeSearchResult(
                    structured_article,
                    230,
                    "high",
                    [f"structured_route:{structured_route}"],
                )

        analysis = analysis or analyze_text(message, context)
        if pattern_match is None:
            pattern_match = best_intent_pattern(message, context, analysis=analysis)
        words = _tokens(normalized_message)
        config = _load_matching_config()

        exact_pool = (
            candidates
            if intent == "unknown"
            else [article for article in candidates if article.intent in {intent, "unknown"}]
        )
        exact_candidates = [
            article
            for article in exact_pool
            if not _has_negative_phrase(article, message, normalized_message)
            and any(
                normalized_phrase in normalized_message
                for phrase in article.user_phrases
                if (normalized_phrase := _normalize(phrase))
            )
        ]
        if exact_candidates:
            best = sorted(
                exact_candidates,
                key=lambda article: (
                    article.priority,
                    article.needs_ticket,
                    article.template is not None,
                    article.scenario,
                ),
                reverse=True,
            )[0]
            return KnowledgeSearchResult(best, 150, "high", ["exact_user_phrase"])

        def score_articles(articles: list[KnowledgeArticle]) -> list[tuple[KnowledgeArticle, int, list[str]]]:
            return [
                (
                    article,
                    *_score_with_features(
                        article,
                        intent,
                        message,
                        context,
                        analysis=analysis,
                        pattern_match=pattern_match,
                        pattern_precomputed=True,
                        config=config,
                        normalized_message=normalized_message,
                        words=words,
                    ),
                )
                for article in articles
            ]

        if intent != "unknown":
            matching_articles = [article for article in candidates if article.intent in {intent, "unknown"}]
            matching_scored = sorted(score_articles(matching_articles), key=lambda item: item[1], reverse=True)
            matching_scored = [item for item in matching_scored if item[1] >= 18]
            if matching_scored:
                best, best_score, matched_features = matching_scored[0]
                if not best.fallback_allowed and best_score < 110:
                    return KnowledgeSearchResult(None, best_score, confidence_level(best_score), matched_features, "fallback_not_allowed")
                return KnowledgeSearchResult(best, best_score, confidence_level(best_score), matched_features)

            matching_ids = {article.slug for article in matching_articles}
            remaining_articles = [article for article in candidates if article.slug not in matching_ids]
            score_by_id = {
                article.slug: (score, features)
                for article, score, features in [*score_articles(matching_articles), *score_articles(remaining_articles)]
            }
            scored = sorted(
                ((article, *score_by_id[article.slug]) for article in candidates),
                key=lambda item: item[1],
                reverse=True,
            )
        else:
            scored = sorted(score_articles(candidates), key=lambda item: item[1], reverse=True)

        best, best_score, matched_features = scored[0]

        if best_score < 18:
            return KnowledgeSearchResult(None, best_score, confidence_level(best_score), matched_features, "score_below_minimum")
        if intent != "unknown" and best.intent not in {intent, "unknown"} and best_score < 180:
            return KnowledgeSearchResult(None, best_score, confidence_level(best_score), matched_features, "intent_conflict")
        if not best.fallback_allowed and best_score < 110:
            return KnowledgeSearchResult(None, best_score, confidence_level(best_score), matched_features, "fallback_not_allowed")
        return KnowledgeSearchResult(best, best_score, confidence_level(best_score), matched_features)


class HybridSearchProvider:
    def search(
        self,
        message: str,
        intent: str,
        role: UserRole,
        context: Any | None = None,
        analysis: TextAnalysis | None = None,
        pattern_match: IntentPatternMatch | None = None,
        skip_topic_ambiguity: bool = False,
    ) -> KnowledgeSearchResult:
        rule_result = RuleBasedSearchProvider().search(
            message,
            intent,
            role,
            context,
            analysis,
            pattern_match,
            skip_topic_ambiguity,
        )
        config = _semantic_config()
        if not bool(config.get("enabled", False)):
            return rule_result
        if rule_result.fallback_reason in {
            "ambiguous_phrase",
            "insufficient_context",
            "topic_candidates_close",
            "too_broad_unknown",
            "no_candidates",
        }:
            return rule_result

        locked_features = {"canonical_phrase", "exact_user_phrase"}
        if locked_features.intersection(rule_result.matched_features):
            return rule_result
        rule_score_lock = int(config.get("rule_score_lock", 220))
        if rule_result.confidence == "high" and rule_result.score >= rule_score_lock:
            return rule_result

        semantic_result = TfidfSemanticSearchProvider().search(
            message,
            intent,
            role,
            context,
            analysis,
            pattern_match,
        )
        if not semantic_result.article:
            return rule_result

        same_article = bool(
            rule_result.article
            and rule_result.article.slug == semantic_result.article.slug
        )
        if same_article:
            if semantic_result.confidence == "high" and rule_result.confidence != "high":
                semantic_result.matched_features.append("semantic_confirms_rule")
                return semantic_result
            if semantic_result.confidence == "medium" and rule_result.confidence == "low":
                semantic_result.matched_features.append("semantic_confirms_rule")
                return semantic_result
            return KnowledgeSearchResult(
                article=rule_result.article,
                score=rule_result.score,
                confidence=rule_result.confidence,
                matched_features=list(dict.fromkeys([
                    *rule_result.matched_features,
                    *semantic_result.matched_features,
                    "semantic_confirms_rule",
                ])),
                fallback_reason=rule_result.fallback_reason,
                clarifying_question=rule_result.clarifying_question,
                clarifying_options=rule_result.clarifying_options,
                semantic_similarity=semantic_result.semantic_similarity,
                semantic_margin=semantic_result.semantic_margin,
                clarifying_article_ids=rule_result.clarifying_article_ids,
            )

        if semantic_result.confidence == "high":
            semantic_result.matched_features.append("semantic_override")
            return semantic_result

        challenge_rule_max_score = int(config.get("challenge_rule_max_score", 150))
        challenge_margin = float(config.get("challenge_margin", 0.06))
        may_challenge_high_rule = (
            rule_result.confidence == "high"
            and rule_result.score <= challenge_rule_max_score
            and semantic_result.semantic_margin >= challenge_margin
        )
        if semantic_result.confidence == "medium" and (
            rule_result.confidence in {"low", "medium"} or may_challenge_high_rule
        ):
            semantic_result.matched_features.append("semantic_clarification")
            return semantic_result
        return rule_result


def search_knowledge_match(
    message: str,
    intent: str,
    role: UserRole,
    context: Any | None = None,
    analysis: TextAnalysis | None = None,
    pattern_match: IntentPatternMatch | None = None,
    skip_topic_ambiguity: bool = False,
) -> KnowledgeSearchResult:
    runtime_settings = get_settings()
    scenario_decision = match_scenario(message, role) if runtime_settings.knowledge_v2_enabled else None
    if runtime_settings.knowledge_v2_shadow_mode:
        legacy_result = HybridSearchProvider().search(
            message,
            intent,
            role,
            context,
            analysis,
            pattern_match,
            skip_topic_ambiguity,
        )
        if scenario_decision and scenario_decision.scenario:
            legacy_result.matched_features.extend(
                [
                    f"shadow_scenario:{scenario_decision.scenario.scenario_id}",
                    f"shadow_scenario_confidence:{scenario_decision.confidence}",
                ]
            )
        return legacy_result
    if scenario_decision and scenario_decision.scenario and scenario_decision.confidence == "high":
        article = get_article_by_id(scenario_decision.scenario.scenario_id, role)
        if article:
            return KnowledgeSearchResult(
                article=article,
                score=scenario_decision.score,
                confidence="high",
                matched_features=["knowledge_v2", *scenario_decision.matched_features],
            )
    if (
        scenario_decision
        and scenario_decision.confidence == "medium"
        and scenario_decision.candidates
        and _normalize(message) not in _canonical_phrase_rules()
    ):
        candidates = list(scenario_decision.candidates)
        return KnowledgeSearchResult(
            article=None,
            score=scenario_decision.score,
            confidence="medium",
            matched_features=["knowledge_v2", *scenario_decision.matched_features],
            fallback_reason="scenario_candidates_close",
            clarifying_question=scenario_decision.clarifying_question,
            clarifying_options=[item.title for item in candidates] + ["Другая тема"],
            clarifying_article_ids=[item.scenario_id for item in candidates] + [""],
            clarifying_intents=[item.intent for item in candidates] + ["unknown"],
        )
    return HybridSearchProvider().search(
        message,
        intent,
        role,
        context,
        analysis,
        pattern_match,
        skip_topic_ambiguity,
    )


def search_knowledge(message: str, intent: str, role: UserRole, context: Any | None = None) -> KnowledgeArticle | None:
    return search_knowledge_match(message, intent, role, context).article
