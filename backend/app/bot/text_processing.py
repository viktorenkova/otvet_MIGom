from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from difflib import SequenceMatcher
import json
from pathlib import Path
import re
from typing import Any


TYPO_CORRECTIONS_PATH = Path("configs/typo_corrections.json")
SYNONYM_GROUPS_PATH = Path("configs/synonym_groups.json")
INTENT_PATTERNS_PATH = Path("configs/intent_patterns.json")
MATCHING_CONFIG_PATH = Path("configs/no_llm_matching_config.json")

FUZZY_THRESHOLD = 0.82
TOKEN_RE = re.compile(r"[a-zа-я0-9@._+\-]+", re.IGNORECASE)
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
PHONE_RE = re.compile(r"(?:\+7|8)[\s\-()]*\d{3}[\s\-()]*\d{3}[\s\-]*\d{2}[\s\-]*\d{2}")

_EN_TO_RU_LAYOUT = str.maketrans(
    "qwertyuiop[]asdfghjkl;'zxcvbnm,.",
    "йцукенгшщзхъфывапролджэячсмитьбю",
)
_RU_TRANSLIT = (
    ("shch", "щ"), ("sch", "щ"), ("yo", "е"), ("zh", "ж"),
    ("kh", "х"), ("ts", "ц"), ("ch", "ч"), ("sh", "ш"),
    ("yu", "ю"), ("ya", "я"), ("ye", "е"),
)
_RU_TRANSLIT_CHARS = str.maketrans(
    "abvgdezijklmnoprstufhcy",
    "абвгдезийклмнопрстуфхцы",
)
_SLANG_ALIASES = {
    "тачка": "машина",
    "доки": "документы",
    "комса": "комиссия",
    "акк": "аккаунт",
    "личка": "личный кабинет",
    "регаться": "регистрация",
    "бид": "ставка",
}


@dataclass(frozen=True)
class TextAnalysis:
    original: str
    normalized: str
    corrected: str
    synonym_normalized: str
    tokens: list[str]
    corrected_tokens: list[str]
    entities: dict[str, list[str]]


@dataclass(frozen=True)
class FeatureMatch:
    group: str
    term: str
    match_type: str
    canonical: str | None = None


@dataclass(frozen=True)
class IntentPatternMatch:
    pattern_id: str
    intent: str
    action: str
    score: int
    confidence_level: str
    matched_features: list[FeatureMatch] = field(default_factory=list)
    missing_groups: list[str] = field(default_factory=list)
    clarifying_options: list[str] = field(default_factory=list)


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def load_typo_corrections() -> dict[str, str]:
    return {normalize_text(str(key)): normalize_text(str(value)) for key, value in _read_json(TYPO_CORRECTIONS_PATH, {}).items()}


@lru_cache(maxsize=1)
def load_synonym_groups() -> dict[str, list[str]]:
    raw = _read_json(SYNONYM_GROUPS_PATH, {})
    return {
        normalize_text(str(canonical)): [normalize_text(str(item)) for item in variants]
        for canonical, variants in raw.items()
        if isinstance(variants, list)
    }


@lru_cache(maxsize=1)
def load_intent_patterns() -> dict[str, dict[str, Any]]:
    raw = _read_json(INTENT_PATTERNS_PATH, {})
    return raw if isinstance(raw, dict) else {}


@lru_cache(maxsize=1)
def load_matching_config() -> dict[str, Any]:
    raw = _read_json(MATCHING_CONFIG_PATH, {})
    return raw if isinstance(raw, dict) else {}


@lru_cache(maxsize=32_768)
def normalize_text(text: str) -> str:
    normalized = text.lower().replace("ё", "е")
    normalized = re.sub(r"\b(рублей|рубля|руб\.?|р\.?)\b|₽", " руб ", normalized)
    normalized = normalized.replace("№", " ")
    normalized = re.sub(r"[^a-zа-я0-9@._+\-\s]+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


@lru_cache(maxsize=32_768)
def _tokenize_cached(text: str) -> tuple[str, ...]:
    return tuple(token for token in TOKEN_RE.findall(normalize_text(text)) if token)


def tokenize(text: str) -> list[str]:
    return list(_tokenize_cached(text))


@lru_cache(maxsize=16_384)
def correct_typos(text: str) -> str:
    corrections = load_typo_corrections()
    corrected_tokens = [corrections.get(token, token) for token in tokenize(text)]
    return " ".join(corrected_tokens)


@lru_cache(maxsize=16_384)
def apply_synonyms(text: str) -> str:
    normalized = normalize_text(text)
    groups = load_synonym_groups()
    for canonical, variants in groups.items():
        phrases = sorted({canonical, *variants}, key=len, reverse=True)
        for phrase in phrases:
            if " " in phrase:
                normalized = re.sub(rf"(?<!\w){re.escape(phrase)}(?!\w)", canonical, normalized)
    tokens = []
    reverse: dict[str, str] = {}
    for canonical, variants in groups.items():
        for item in [canonical, *variants]:
            if " " not in item:
                reverse[item] = canonical
    for token in tokenize(normalized):
        tokens.append(reverse.get(token, token))
    return " ".join(tokens)


def _iter_string_values(value: Any):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _iter_string_values(item)
    elif isinstance(value, list):
        for item in value:
            yield from _iter_string_values(item)


@lru_cache(maxsize=1)
def _matching_vocabulary() -> frozenset[str]:
    """Domain vocabulary shared by routing, intent and knowledge matching."""
    words: set[str] = set()
    sources = (
        _read_json(TYPO_CORRECTIONS_PATH, {}),
        _read_json(SYNONYM_GROUPS_PATH, {}),
        _read_json(INTENT_PATTERNS_PATH, {}),
        _read_json(MATCHING_CONFIG_PATH, {}),
    )
    for source in sources:
        for value in _iter_string_values(source):
            words.update(tokenize(value))
    return frozenset(word for word in words if re.fullmatch(r"[а-я]+", word))


def _transliterate_token(token: str) -> str:
    converted = token
    for source, target in _RU_TRANSLIT:
        converted = converted.replace(source, target)
    return converted.translate(_RU_TRANSLIT_CHARS)


def _known_token_score(token: str, vocabulary: frozenset[str]) -> float:
    if token in vocabulary:
        return 1.0
    if len(token) < 4:
        return 0.0
    best = 0.0
    for candidate in vocabulary:
        if abs(len(token) - len(candidate)) > 2:
            continue
        ratio = SequenceMatcher(None, token, candidate).ratio()
        if ratio > best:
            best = ratio
    return best


def _convert_latin_words(text: str) -> str:
    """Convert keyboard-layout mistakes or Russian translit when domain evidence supports it."""
    vocabulary = _matching_vocabulary()
    tokens = text.lower().split()
    latin_positions = [index for index, token in enumerate(tokens) if re.search(r"[a-z]", token)]
    if not latin_positions:
        return normalize_text(text).replace("-", " ")

    candidates: list[tuple[float, list[str]]] = []
    for converter in (lambda value: value.translate(_EN_TO_RU_LAYOUT), _transliterate_token):
        converted = list(tokens)
        scores: list[float] = []
        for index in latin_positions:
            converted[index] = converter(tokens[index])
            converted_tokens = tokenize(converted[index])
            scores.append(max((_known_token_score(item, vocabulary) for item in converted_tokens), default=0.0))
        evidence = sum(score >= 0.82 for score in scores)
        average = sum(scores) / len(scores)
        coverage = evidence / len(scores)
        candidates.append((coverage + average, converted))

    score, converted = max(candidates, key=lambda item: item[0])
    # Do not rewrite ordinary English, identifiers, e-mails or URLs on weak evidence.
    selected = " ".join(converted) if score >= 1.25 else text
    return normalize_text(selected).replace("-", " ")


def _repair_matching_token(token: str) -> str:
    vocabulary = _matching_vocabulary()
    if token in vocabulary or len(token) < 5 or not re.fullmatch(r"[а-я]+", token):
        return token
    candidates = (
        candidate for candidate in vocabulary
        if candidate[0] == token[0] and abs(len(candidate) - len(token)) <= 1
    )
    best = max(candidates, key=lambda candidate: SequenceMatcher(None, token, candidate).ratio(), default=token)
    return best if SequenceMatcher(None, token, best).ratio() >= 0.9 else token


@lru_cache(maxsize=32_768)
def normalize_matching_text(text: str) -> str:
    """Canonical representation for every no-LLM matching layer.

    The order is intentional: base cleanup, guarded layout/translit recovery,
    curated typo correction, then shared synonym canonicalization.
    """
    converted = _convert_latin_words(text)
    corrected = correct_typos(converted)
    repaired = (_repair_matching_token(token) for token in tokenize(corrected))
    return " ".join(_SLANG_ALIASES.get(token, token) for token in repaired)


@lru_cache(maxsize=65_536)
def fuzzy_ratio(left: str, right: str) -> float:
    if len(left) < 4 or len(right) < 4:
        return 0.0
    return SequenceMatcher(None, left, right).ratio()


def token_matches(term: str, tokens: set[str], synonym_tokens: set[str] | None = None) -> FeatureMatch | None:
    normalized_term = normalize_text(term)
    synonym_tokens = synonym_tokens or set()
    if not normalized_term:
        return None
    term_tokens = tokenize(normalized_term)
    if not term_tokens:
        return None
    if len(term_tokens) > 1:
        phrase = " ".join(tokens)
        synonym_phrase = " ".join(synonym_tokens)
        if normalized_term in phrase:
            return FeatureMatch(group="", term=term, match_type="exact")
        if normalized_term in synonym_phrase:
            return FeatureMatch(group="", term=term, match_type="synonym")
        return None
    term_token = term_tokens[0]
    if term_token in tokens:
        return FeatureMatch(group="", term=term, match_type="exact")
    if term_token in synonym_tokens:
        return FeatureMatch(group="", term=term, match_type="synonym", canonical=term_token)
    for token in tokens:
        if fuzzy_ratio(term_token, token) >= FUZZY_THRESHOLD:
            return FeatureMatch(group="", term=term, match_type="fuzzy")
    return None


@lru_cache(maxsize=65_536)
def phrase_matches(message: str, phrase: str) -> tuple[bool, str]:
    normalized_message = normalize_text(message)
    corrected_message = normalize_matching_text(message)
    synonym_message = apply_synonyms(corrected_message)
    normalized_phrase = normalize_text(phrase)
    corrected_phrase = correct_typos(normalized_phrase)
    synonym_phrase = apply_synonyms(corrected_phrase)
    if not normalized_phrase:
        return False, ""
    if normalized_phrase in normalized_message or corrected_phrase in corrected_message:
        return True, "exact"
    if synonym_phrase and synonym_phrase in synonym_message:
        return True, "synonym"
    phrase_tokens = tokenize(corrected_phrase)
    if len(phrase_tokens) < 2:
        match = token_matches(corrected_phrase, set(tokenize(corrected_message)), set(tokenize(synonym_message)))
        return (match is not None, match.match_type if match else "")
    message_tokens = set(tokenize(corrected_message))
    synonym_tokens = set(tokenize(synonym_message))
    matched = 0
    fuzzy = False
    synonym = False
    for token in phrase_tokens:
        match = token_matches(token, message_tokens, synonym_tokens)
        if not match:
            continue
        matched += 1
        fuzzy = fuzzy or match.match_type == "fuzzy"
        synonym = synonym or match.match_type == "synonym"
    if matched == len(phrase_tokens):
        return True, "fuzzy" if fuzzy else "synonym" if synonym else "exact"
    return False, ""


def extract_entities(message: str, context: Any | None = None) -> dict[str, list[str]]:
    entities: dict[str, list[str]] = {
        "lot_id": [],
        "payment_id": [],
        "phone": [],
        "email": [],
        "amount": [],
        "duration": [],
    }
    text = message
    normalized = normalize_text(message)
    context_lot_id = str(getattr(context, "lot_id", "") or "") if context is not None else ""
    if context_lot_id:
        entities["lot_id"].append(context_lot_id)
    for match in re.finditer(r"(?<!\w)(?:лот|lot)\s*#?\s*([a-zа-я0-9\-]{2,})", normalized, flags=re.IGNORECASE):
        candidate = match.group(1)
        if any(char.isdigit() for char in candidate):
            entities["lot_id"].append(candidate)
    for match in re.finditer(r"(?<!\w)(?:платеж|payment|операци[яи]|счет)\s*#?\s*([a-z0-9\-]{3,})", normalized, flags=re.IGNORECASE):
        candidate = match.group(1)
        if any(char.isdigit() for char in candidate):
            entities["payment_id"].append(candidate)
    entities["phone"].extend(match.group(0) for match in PHONE_RE.finditer(text))
    entities["email"].extend(match.group(0) for match in EMAIL_RE.finditer(text))
    for match in re.finditer(r"\b\d[\d\s]*(?:руб|₽)\b", normalized):
        entities["amount"].append(re.sub(r"\s+", " ", match.group(0)).strip())
    for match in re.finditer(r"\b\d+\s*(?:минут[а-я]*|час[а-я]*|дн[еяй]*|недел[яиюь]*|месяц[а-я]*)\b", normalized):
        entities["duration"].append(match.group(0))
    return {key: list(dict.fromkeys(value)) for key, value in entities.items()}


def analyze_text(message: str, context: Any | None = None) -> TextAnalysis:
    normalized = normalize_text(message)
    corrected = normalize_matching_text(message)
    synonym_normalized = apply_synonyms(corrected)
    return TextAnalysis(
        original=message,
        normalized=normalized,
        corrected=corrected,
        synonym_normalized=synonym_normalized,
        tokens=tokenize(normalized),
        corrected_tokens=tokenize(corrected),
        entities=extract_entities(message, context),
    )


def confidence_level(score: int) -> str:
    if score >= 80:
        return "high"
    if score >= 50:
        return "medium"
    return "low"


def _group_match(group_name: str, terms: list[Any], analysis: TextAnalysis) -> FeatureMatch | None:
    tokens = set(analysis.corrected_tokens)
    synonym_tokens = set(tokenize(analysis.synonym_normalized))
    for term in terms:
        normalized_term = normalize_text(str(term))
        if " " in normalized_term:
            matched, match_type = phrase_matches(analysis.corrected, normalized_term)
            if matched:
                return FeatureMatch(group=group_name, term=str(term), match_type=match_type)
        match = token_matches(str(term), tokens, synonym_tokens)
        if match:
            return FeatureMatch(group=group_name, term=str(term), match_type=match.match_type, canonical=match.canonical)
    return None


def match_intent_patterns(
    message: str,
    context: Any | None = None,
    analysis: TextAnalysis | None = None,
) -> list[IntentPatternMatch]:
    analysis = analysis or analyze_text(message, context)
    matches: list[IntentPatternMatch] = []
    for pattern_id, pattern in load_intent_patterns().items():
        required_any = pattern.get("required_any", {})
        if not isinstance(required_any, dict):
            continue
        matched_features: list[FeatureMatch] = []
        missing_groups: list[str] = []
        score = 0
        for group_name, terms in required_any.items():
            if not isinstance(terms, list):
                continue
            match = _group_match(str(group_name), terms, analysis)
            if match:
                matched_features.append(match)
                score += 35
                if match.match_type == "synonym":
                    score += 10
                elif match.match_type == "fuzzy":
                    score += 5
            else:
                missing_groups.append(str(group_name))
        if matched_features:
            if not missing_groups:
                score += 35
            matches.append(
                IntentPatternMatch(
                    pattern_id=str(pattern_id),
                    intent=str(pattern.get("intent") or pattern_id),
                    action=str(pattern.get("action") or "answer"),
                    score=score,
                    confidence_level=confidence_level(score),
                    matched_features=matched_features,
                    missing_groups=missing_groups,
                    clarifying_options=[str(item) for item in pattern.get("clarifying_options", [])],
                )
            )
    return sorted(matches, key=lambda item: (item.score, -len(item.missing_groups)), reverse=True)


def best_intent_pattern(
    message: str,
    context: Any | None = None,
    analysis: TextAnalysis | None = None,
) -> IntentPatternMatch | None:
    matches = match_intent_patterns(message, context, analysis=analysis)
    return matches[0] if matches else None


def has_specific_problem(
    message: str,
    intent: str,
    context: Any | None = None,
    analysis: TextAnalysis | None = None,
) -> bool:
    analysis = analysis or analyze_text(message, context)
    text = analysis.corrected
    specific_markers = ("мой", "моя", "мне", "у меня", "я оплатил", "я оплатила", "мне начислили")
    topic_terms = ("лот", "платеж", "тариф", "штраф", "возврат", "отказ", "передача", "ставка")
    problem_terms = (
        "не могу",
        "не работает",
        "нет",
        "не появился",
        "не отображается",
        "не передают",
        "не отдают",
        "не выдают",
        "начислили",
        "списали",
        "списались",
        "завис",
        "долго",
        "срок",
    )
    has_marker = any(marker in text for marker in specific_markers)
    has_entity = any(analysis.entities[key] for key in ("lot_id", "payment_id", "amount", "duration"))
    has_problem = any(term in text for term in problem_terms) or bool(analysis.entities["amount"] or analysis.entities["duration"])
    has_topic = any(term in text for term in topic_terms)
    if intent in {"lot", "penalty", "refund", "refusal", "transfer", "pickup"}:
        has_topic = True
    if intent == "tariffs" and any(term in text for term in ("оплатил", "оплатила", "списали", "списались", "нет доступа", "не появился", "не подключился", "не активировался")):
        has_topic = True
    if intent == "payment" and any(term in text for term in ("платеж", "оплата", "оплатил", "оплатила", "чек", "квитанция")):
        has_topic = True
    return has_topic and has_problem and (has_marker or has_entity)
