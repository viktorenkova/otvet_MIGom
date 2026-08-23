from __future__ import annotations

from dataclasses import dataclass
from collections import Counter
from difflib import SequenceMatcher
from functools import lru_cache
import re
from typing import Iterable

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from backend.app.bot.scenario_engine import QueryFacets, Scenario, extract_query_facets, load_scenarios
from backend.app.bot.text_processing import apply_synonyms, correct_typos, normalize_matching_text, normalize_text, tokenize


_CONCEPTS: dict[str, tuple[str, ...]] = {
    "account": ("аккаунт", "кабинет", "учетная запись", "авторизация"),
    "auction": ("торги", "торг", "аукцион", "котировка"),
    "balance": ("баланс", "кошелек", "счет"),
    "bid": ("ставка", "ставки", "предложение цены", "ценовое предложение", "предложение", "максималка"),
    "bid_change": ("отменить", "убрать", "изменить", "понизить", "передумал", "ранее поданная"),
    "bid_place": ("сделать", "поставить", "подать", "совершить", "отправить", "инструкция"),
    "bid_visibility": ("чужие", "друг друга", "показываются", "участники видят", "видны другим"),
    "commission": ("комиссия", "комса", "сбор", "сумма больше"),
    "complaint": ("ужас", "отврат", "плохой", "тупит", "хрень", "невозможно работать", "издеваетесь"),
    "connect": ("подключить", "оформить", "активировать", "получить доступ"),
    "contract": ("договор", "дкп", "документы по сделке"),
    "credentials": ("пароль", "логин", "код входа"),
    "deposit": ("депозит", "обеспечительный платеж"),
    "definition": ("что такое", "это", "означает", "объясните термин"),
    "documents": ("документы", "доки", "договор", "дкп", "счет", "акт"),
    "document_access": ("где скачать", "скачать", "получить договор", "где договор", "открыть документ"),
    "employee": ("менеджер", "сотрудник", "михаил", "алексей", "реник"),
    "eligibility": ("можно ли", "можно вернуть", "можно возвратить", "не участвовал", "не торговался", "какие средства", "подлежат", "условия", "возвращают ли", "остаток", "ошибочно пополнил"),
    "error": ("ошибка", "сбой", "не работает", "не проходит", "не принимает", "не получается", "не могу оплатить", "не делает", "не дает", "отклоняется", "завис", "сломался", "не загружается", "не грузится", "не открывается", "белый экран", "тупит"),
    "feedback": ("предлагаю", "предложение", "добавьте", "сделайте", "улучшение", "хочу предложить"),
    "filter": ("фильтр", "фильтрация", "сортировка", "каталог", "поиск", "выбор года", "регион", "марка", "модель", "объявление"),
    "filter_problem": ("не отбирает", "не меняет", "не обновляется", "неверные лоты", "неподходящие лоты", "лишние лоты", "все подряд", "работает наоборот", "сбрасывает фильтр", "некорректная фильтрация"),
    "format": ("открытые", "закрытые", "формат", "видят цены", "друг друга", "котировка"),
    "image": ("фото", "фотография", "картинка", "изображение"),
    "page_blank": ("белый экран", "страница пустая", "пустой экран"),
    "legal_form": ("физлицо", "юридическое лицо", "юрлицо", "ооо", "самозанятый", "самозанятая"),
    "login": ("авторизоваться", "авторизация", "войти", "вход", "зайти"),
    "location": ("адрес", "где находится", "местонахождение", "локация", "куда ехать", "стоянка", "стоянки", "в каком месте", "где стоит"),
    "lot": ("лот", "автомобиль", "машина", "авто", "тачка", "транспортное средство", "объявление", "марка", "модель", "стоянка", "стоянки"),
    "no_response": ("не отвечает", "не ответил", "не ответила", "не ответили", "нет ответа", "без ответа", "молчит", "тишина", "игнорирует", "не выходит на связь", "не получается связаться", "не присылает", "ни ответа", "пропал", "потерялся"),
    "not_visible": ("не видно", "не отображается", "исчезла", "пропала", "не появилась", "не изменился", "не вижу", "нигде нет", "на счете нет", "не выслали", "не прислали", "не получил", "не пополнился", "не активирован", "не активировался"),
    "office": ("офис", "ваша организация", "ваша компания", "к вам приехать", "к вам ехать"),
    "organization": ("ваша организация", "ваша компания", "вы находитесь", "ваш адрес"),
    "payment": ("оплата", "оплатить", "платеж", "деньги", "перечислил", "списал", "пополнил"),
    "payment_done": ("прошел", "списался", "списал", "в банке", "перечислил", "оплаченный", "деньги сняли"),
    "prepare": ("подготовить", "подготовиться", "учесть заранее", "проверить заранее"),
    "pickup": ("забрать", "получить", "выдача", "отдадут", "отдают", "выдают"),
    "premium": ("премиум", "премиальный"),
    "refund": ("возврат", "вернуть", "возвратить", "возвращать", "вернули", "возвращают", "обратно деньги", "депозит обратно", "возвращение средств", "вывести", "возвратный"),
    "refund_application": ("заявление", "запрос", "подать", "направить", "отправить", "куда писать", "шаблон"),
    "registration": ("регистрация", "регистрирование", "зарегистрироваться", "создать аккаунт", "новый покупатель"),
    "recover": ("восстановить", "забыл пароль", "не принимает пароль", "не могу войти", "не получается войти"),
    "seller": ("продавец", "страховая", "страховщик", "страхование", "страхования", "ресо", "альфа", "ингосстрах", "росгосстрах", "вск", "согласие", "ренессанс", "сбер страхование", "совкомбанк", "зетта", "югория", "тинькофф", "т страхование"),
    "status": ("статус", "состояние", "когда", "сколько ждать", "сколько идет", "завершенность", "завершились", "закончились", "окончены", "итог", "результат"),
    "outcome": ("победил", "выиграл", "победитель", "результат торгов", "итог торгов"),
    "support": ("поддержка", "почтовая ветка", "письмо", "электронная переписка"),
    "bot": ("бот", "не помогло", "ваш ответ"),
    "tariff": ("тариф", "доступ", "подписка"),
    "tariff_explicit": ("тариф", "подписка"),
    "search_action": ("искать", "найти", "подобрать", "отыскать"),
    "visit": ("приехать", "визит", "прием", "записаться", "пропуск", "лично", "попасть"),
    "win_stage": ("выиграл", "победа", "после торгов", "передан", "передача", "оплачен", "после оплаты"),
}


# These profiles describe semantic differences, not audit phrases. They form
# the constrained reranker layer and are intentionally much smaller than the
# set of user formulations.
_PROFILES: tuple[tuple[str, tuple[str, ...], tuple[str, ...], tuple[str, ...]], ...] = (
    ("transfer.seller_no_response", ("seller", "no_response"), ("lot", "win_stage"), ("employee",)),
    ("tariff.connect", ("premium", "connect"), (), ()),
    ("tariff.premium", ("premium",), ("tariff",), ("connect",)),
    ("tariff.status", ("tariff", "payment_done", "not_visible"), (), ()),
    ("payment.not_visible", ("payment", "payment_done", "not_visible"), ("balance",), ("tariff",)),
    ("payment.checkout_problem", ("payment", "error"), (), ("payment_done", "not_visible")),
    ("commission.explained", ("commission",), ("lot", "payment"), ("balance",)),
    ("balance.topup.commission", ("commission", "balance"), (), ()),
    ("refund.application", ("refund", "refund_application"), (), ("status",)),
    ("refund.timing_status", ("refund", "status"), (), ()),
    ("refund.timing_status", ("refund_application", "status"), (), ()),
    ("refund.eligibility", ("refund", "eligibility"), ("deposit", "balance"), ("refund_application", "status")),
    ("bid.place", ("bid", "bid_place"), ("auction",), ("not_visible", "bid_change", "bid_visibility", "status")),
    ("bid.price_terms", ("bid", "definition"), (), ("error", "bid_change", "not_visible")),
    ("bid.not_visible", ("bid", "not_visible"), (), ()),
    ("bid.modify_cancel", ("bid", "bid_change"), (), ("not_visible",)),
    ("bid.modify_cancel", ("bid_change", "auction"), ("bid",), ("not_visible",)),
    ("auction.formats", ("auction", "format"), (), ("status", "bid_change")),
    ("auction.formats", ("format", "bid_visibility"), (), ("error", "bid_change")),
    ("auction.formats", ("bid", "bid_visibility"), (), ("error", "bid_change")),
    ("auction.status", ("auction", "status"), (), ()),
    ("technical.site_error", ("error",), ("account", "lot", "page_blank"), ("payment", "filter", "image", "credentials", "recover", "login")),
    ("technical.site_error", ("page_blank",), ("lot",), ()),
    ("technical.site_error", ("complaint",), (), ("filter", "payment", "bid")),
    ("technical.catalog_search_filter", ("filter", "error"), (), ()),
    ("technical.catalog_search_filter", ("filter", "filter_problem"), ("lot",), ()),
    ("technical.lot_image_missing", ("image", "error"), (), ("page_blank",)),
    ("documents.preparation_delay", ("documents", "not_visible"), ("win_stage",), ()),
    ("documents.preparation_delay", ("documents", "not_visible", "win_stage"), (), ()),
    ("documents.preparation_delay", ("documents", "win_stage", "status"), (), ()),
    ("support.office_visit", ("office",), ("location", "visit"), ("lot",)),
    ("support.callback", ("employee",), ("visit",), ("seller",)),
    ("support.email_no_response", ("support", "no_response"), (), ("seller",)),
    ("lot.catalog_search", ("filter", "lot"), (), ("error", "filter_problem")),
    ("lot.catalog_search", ("search_action", "lot"), (), ("location",)),
    ("lot.location", ("location", "lot"), (), ("office",)),
    ("account.login_problem", ("account", "credentials"), ("error",), ()),
    ("account.login_problem", ("account", "recover"), (), ()),
    ("account.login_problem", ("account", "login"), ("error",), ()),
    ("account.registration", ("registration",), ("account", "legal_form"), ()),
    ("pickup.receive_lot", ("pickup", "lot"), ("win_stage",), ("error", "no_response", "not_visible")),
    ("pickup.access_issuer", ("pickup", "lot", "error"), ("location",), ("seller",)),
    ("contract.receive", ("contract", "document_access"), ("win_stage",), ()),
    ("feedback.improvement_suggestion", ("feedback",), ("filter", "bid", "image"), ("bid_change",)),
    ("feedback.platform_complaint", ("complaint",), ("filter", "image", "payment", "bid"), ("feedback", "error")),
    ("feedback.bot_answer_complaint", ("bot",), ("support",), ("employee",)),
    ("support.office_visit", ("visit", "location"), (), ("lot",)),
    ("support.office_visit", ("visit",), (), ("lot", "pickup")),
    ("buyer.first_bid_checklist", ("prepare", "auction"), (), ("bid_place",)),
    ("buyer.first_bid_checklist", ("prepare", "bid"), (), ("bid_place",)),
    ("auction.result", ("auction", "outcome"), (), ("not_visible", "bid_change")),
    ("tariff.choose", ("tariff_explicit",), (), ("premium", "connect", "payment_done", "account", "refund")),
    ("account.registration", ("legal_form",), ("registration",), ()),
)


@dataclass(frozen=True)
class RoutingCandidate:
    scenario: Scenario
    score: float
    lexical_score: float
    char_score: float
    facet_score: float
    matched_features: tuple[str, ...] = ()


@dataclass(frozen=True)
class RoutingDecision:
    scenario: Scenario | None
    confidence: str
    score: float
    margin: float
    candidates: tuple[RoutingCandidate, ...]
    normalized_query: str
    matched_features: tuple[str, ...] = ()


_RUSSIAN_SUFFIXES = tuple(
    sorted(
        {
            "иями", "ями", "ами", "его", "ого", "ему", "ому", "ее", "ие", "ые", "ое",
            "ей", "ий", "ый", "ой", "ем", "им", "ым", "ом", "их", "ых", "ую", "юю",
            "ая", "яя", "ою", "ею", "овать", "евать", "ировать", "ается", "яется", "иться",
            "утся", "ются", "ать", "ять", "ить", "ыть", "ешь", "ете", "ем", "им", "ите",
            "ут", "ют", "ат", "ят", "ал", "ала", "али", "ил", "ила", "или", "ено", "ена",
            "ены", "ение", "ения", "ений", "ению", "ением", "иях", "ию", "ия", "ья", "ью",
            "ию", "ью", "ов", "ев", "ей", "ам", "ям", "ах", "ях", "ом", "ем", "а", "я",
            "ы", "и", "ь", "й", "у", "ю", "о", "е",
        },
        key=len,
        reverse=True,
    )
)


def _stem(token: str) -> str:
    if len(token) < 5 or not re.fullmatch(r"[а-я]+", token):
        return token
    for suffix in _RUSSIAN_SUFFIXES:
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            return token[: -len(suffix)]
    return token


@lru_cache(maxsize=1)
def _domain_vocabulary() -> tuple[str, ...]:
    words: set[str] = set()
    for aliases in _CONCEPTS.values():
        for alias in aliases:
            words.update(tokenize(normalize_text(alias)))
    return tuple(sorted((word for word in words if len(word) >= 3), key=lambda item: (len(item), item)))


def _damerau_distance(left: str, right: str) -> int:
    if left == right:
        return 0
    rows, columns = len(left) + 1, len(right) + 1
    distance = [[0] * columns for _ in range(rows)]
    for row in range(rows):
        distance[row][0] = row
    for column in range(columns):
        distance[0][column] = column
    for row in range(1, rows):
        for column in range(1, columns):
            cost = int(left[row - 1] != right[column - 1])
            distance[row][column] = min(
                distance[row - 1][column] + 1,
                distance[row][column - 1] + 1,
                distance[row - 1][column - 1] + cost,
            )
            if row > 1 and column > 1 and left[row - 1] == right[column - 2] and left[row - 2] == right[column - 1]:
                distance[row][column] = min(distance[row][column], distance[row - 2][column - 2] + cost)
    return distance[-1][-1]


def _repair_domain_token(token: str) -> str:
    if len(token) < 3 or not re.fullmatch(r"[а-я]+", token):
        return token
    vocabulary = _domain_vocabulary()
    if token in vocabulary:
        return token
    best = token
    best_ratio = 0.0
    best_distance = 10**9
    for candidate in vocabulary:
        if abs(len(token) - len(candidate)) > 2:
            continue
        distance = _damerau_distance(token, candidate)
        deletion_match = (
            len(token) - len(candidate) == 1
            and not (Counter(candidate) - Counter(token))
        )
        ratio = SequenceMatcher(None, token, candidate).ratio()
        effective_distance = min(distance, 1 if deletion_match else distance)
        if effective_distance < best_distance or (
            effective_distance == best_distance and ratio > best_ratio
        ):
            best, best_distance, best_ratio = candidate, effective_distance, ratio
    if best_distance == 1 or (best_distance <= 2 and len(token) >= 7):
        return best
    threshold = 0.83 if len(token) >= 6 else 0.86
    return best if best_ratio >= threshold else token


@lru_cache(maxsize=65_536)
def routing_normalize(text: str) -> str:
    canonical = normalize_matching_text(text)
    repaired = " ".join(_repair_domain_token(token) for token in tokenize(canonical))
    synonymized = apply_synonyms(repaired)
    return " ".join(_stem(token) for token in tokenize(synonymized))


def _word_analyzer(text: str) -> list[str]:
    tokens = routing_normalize(text).split()
    if not tokens:
        return []
    features = list(tokens)
    features.extend(f"{left}_{right}" for left, right in zip(tokens, tokens[1:]))
    return features


def _token_similar(left: str, right: str) -> bool:
    if left == right or (
        min(len(left), len(right)) >= 5
        and abs(len(left) - len(right)) <= 2
        and (left.startswith(right) or right.startswith(left))
    ):
        return True
    if min(len(left), len(right)) < 4 or abs(len(left) - len(right)) > 2:
        return False
    if left[0] != right[0]:
        return False
    return SequenceMatcher(None, left, right).ratio() >= 0.86


def _basic_tokens(text: str) -> tuple[str, ...]:
    return tuple(_stem(token) for token in tokenize(normalize_matching_text(text)))


def _phrase_present(query_tokens: tuple[str, ...], phrase: str) -> bool:
    phrase_variants = {
        tuple(routing_normalize(phrase).split()),
        _basic_tokens(phrase),
    }
    for phrase_tokens in phrase_variants:
        if not phrase_tokens:
            continue
        if len(phrase_tokens) == 1 and any(_token_similar(phrase_tokens[0], token) for token in query_tokens):
            return True
        # Word order is deliberately ignored here: the lexical stage already
        # captures exact phrases, while this layer must survive natural reordering.
        if len(phrase_tokens) > 1 and all(
            any(_token_similar(term, token) for token in query_tokens)
            for term in phrase_tokens
        ):
            return True
    return False


def _matched_concepts(message: str) -> frozenset[str]:
    query_tokens = tuple(dict.fromkeys([*routing_normalize(message).split(), *_basic_tokens(message)]))
    concepts = {
        name
        for name, aliases in _CONCEPTS.items()
        if any(_phrase_present(query_tokens, alias) for alias in aliases)
    }
    # Detect a negated communication action compositionally. This covers
    # inflected verbs and word reordering without adding audit sentences (or
    # every possible "X does not write/call back" phrase) to the vocabulary.
    negation = any(token in {"не", "нет", "без", "ничего"} for token in query_tokens)
    communication_stems = ("ответ", "пис", "пиш", "связ", "звон")
    if negation and any(token.startswith(communication_stems) for token in query_tokens):
        concepts.add("no_response")
    # Mail-channel roots remain identifiable even when a suffix contains a
    # transposition that the conservative token repair intentionally leaves
    # untouched.
    if any(token.startswith(("почт", "письм", "email")) for token in query_tokens):
        concepts.add("support")
    return frozenset(concepts)


def _profile_scores(message: str) -> dict[str, tuple[float, tuple[str, ...]]]:
    concepts = _matched_concepts(message)
    scores: dict[str, tuple[float, tuple[str, ...]]] = {}
    for scenario_id, required, optional, forbidden in _PROFILES:
        if not all(item in concepts for item in required):
            continue
        if any(item in concepts for item in forbidden):
            continue
        optional_hits = tuple(item for item in optional if item in concepts)
        score = 0.68 + min(0.22, len(required) * 0.06) + min(0.08, len(optional_hits) * 0.03)
        previous = scores.get(scenario_id)
        features = tuple([*(f"concept:{item}" for item in required), *(f"optional:{item}" for item in optional_hits)])
        if previous is None or score > previous[0]:
            scores[scenario_id] = (score, features)
    return scores


def _scenario_documents(scenario: Scenario) -> list[str]:
    metadata = " ".join(
        [
            scenario.title,
            scenario.scenario_id.replace(".", " ").replace("_", " "),
            *scenario.objects,
            *scenario.operations,
            *scenario.states,
            scenario.stage,
        ]
    )
    documents = [scenario.title, metadata, scenario.search_document]
    documents.extend(scenario.positive_examples)
    taxonomy_groups = [
        " ".join(str(term) for term in group.get("terms", []) if str(term).strip())
        for group in scenario.retrieval_taxonomy_terms
    ]
    if taxonomy_groups:
        documents.append(" ".join(item for item in taxonomy_groups if item))
    return [item for item in documents if item.strip()]


class HybridScenarioRouter:
    """One ranked route layer over scenario examples.

    Character n-grams provide typo tolerance. Stemmed word unigrams/bigrams
    distinguish operations and state. Scenario facets are used only as a
    bounded reranking signal, so a single generic keyword cannot dominate.
    """

    def __init__(self, scenarios: Iterable[Scenario]) -> None:
        self.scenarios = tuple(scenarios)
        self.documents: list[str] = []
        self.document_scenarios: list[int] = []
        for scenario_index, scenario in enumerate(self.scenarios):
            for document in _scenario_documents(scenario):
                self.documents.append(document)
                self.document_scenarios.append(scenario_index)

        self.char_vectorizer = TfidfVectorizer(
            preprocessor=routing_normalize,
            analyzer="char_wb",
            ngram_range=(2, 5),
            min_df=1,
            sublinear_tf=True,
            dtype=np.float32,
        )
        self.word_vectorizer = TfidfVectorizer(
            analyzer=_word_analyzer,
            min_df=1,
            sublinear_tf=True,
            dtype=np.float32,
        )
        self.char_matrix = self.char_vectorizer.fit_transform(self.documents)
        self.word_matrix = self.word_vectorizer.fit_transform(self.documents)

    @staticmethod
    def _facet_score(facets: QueryFacets, scenario: Scenario) -> tuple[float, tuple[str, ...]]:
        object_hits = facets.objects.intersection(scenario.objects)
        operation_hits = facets.operations.intersection(scenario.operations)
        state_hits = facets.states.intersection(scenario.states)
        score = min(0.10, len(object_hits) * 0.025)
        score += min(0.12, len(operation_hits) * 0.04)
        score += min(0.10, len(state_hits) * 0.04)
        features = [*(f"object:{item}" for item in sorted(object_hits))]
        features.extend(f"operation:{item}" for item in sorted(operation_hits))
        features.extend(f"state:{item}" for item in sorted(state_hits))
        if scenario.stage and facets.stage == scenario.stage:
            score += 0.05
            features.append(f"stage:{scenario.stage}")
        return min(score, 0.25), tuple(features)

    def rank(self, message: str, role: str = "guest", top_k: int = 5) -> tuple[RoutingCandidate, ...]:
        normalized = routing_normalize(message)
        if not normalized:
            return ()
        char_values = (self.char_matrix @ self.char_vectorizer.transform([message]).T).toarray().ravel()
        word_values = (self.word_matrix @ self.word_vectorizer.transform([message]).T).toarray().ravel()
        facets = extract_query_facets(message)
        profile_scores = _profile_scores(message)

        per_scenario: list[list[tuple[float, float]]] = [[] for _ in self.scenarios]
        for position, scenario_index in enumerate(self.document_scenarios):
            per_scenario[scenario_index].append((float(char_values[position]), float(word_values[position])))

        ranked: list[RoutingCandidate] = []
        for index, scenario in enumerate(self.scenarios):
            if role not in scenario.roles and "all" not in scenario.roles:
                continue
            values = per_scenario[index]
            best_char = max((item[0] for item in values), default=0.0)
            best_word = max((item[1] for item in values), default=0.0)
            lexical = best_char * 0.48 + best_word * 0.52
            facet_score, features = self._facet_score(facets, scenario)
            profile_score, profile_features = profile_scores.get(scenario.scenario_id, (0.0, ()))
            total = max(lexical * 0.86 + facet_score, profile_score + lexical * 0.12)
            ranked.append(
                RoutingCandidate(
                    scenario=scenario,
                    score=total,
                    lexical_score=lexical,
                    char_score=best_char,
                    facet_score=facet_score,
                    matched_features=(*features, *profile_features),
                )
            )
        ranked.sort(key=lambda item: item.score, reverse=True)
        return tuple(ranked[: max(1, top_k)])

    def decide(self, message: str, role: str = "guest", top_k: int = 5) -> RoutingDecision:
        candidates = self.rank(message, role, top_k)
        normalized = routing_normalize(message)
        if not candidates:
            return RoutingDecision(None, "low", 0.0, 0.0, (), normalized)
        best = candidates[0]
        margin = best.score - (candidates[1].score if len(candidates) > 1 else 0.0)
        profile_evidence = sum(
            feature.startswith("concept:") for feature in best.matched_features
        )
        # A conflict profile is decisive only when at least two independent
        # concepts agree and the candidate still leads the lexical/facet rank.
        # This keeps one broad keyword from turning a close retrieval into an
        # overconfident route while allowing compositional paraphrases through.
        decisive_profile = (
            profile_evidence >= 2
            and best.score >= 0.76
            and margin >= 0.015
        )
        if (best.score >= 0.58 and margin >= 0.045) or decisive_profile:
            confidence = "high"
            scenario = best.scenario
        elif best.score >= 0.36:
            confidence = "medium"
            scenario = None
        else:
            confidence = "low"
            scenario = None
        return RoutingDecision(
            scenario=scenario,
            confidence=confidence,
            score=best.score,
            margin=margin,
            candidates=candidates,
            normalized_query=normalized,
            matched_features=(
                (*best.matched_features, "decisive_conflict_profile")
                if decisive_profile and margin < 0.045
                else best.matched_features
            ),
        )


@lru_cache(maxsize=1)
def get_routing_v3() -> HybridScenarioRouter:
    return HybridScenarioRouter(load_scenarios())


def clear_routing_v3_cache() -> None:
    get_routing_v3.cache_clear()
    routing_normalize.cache_clear()
