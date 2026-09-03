from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path
import re
from typing import Any

from backend.app.bot.text_processing import normalize_matching_text
from backend.app.config import get_settings


@dataclass(frozen=True)
class AnswerContract:
    scenario_id: str
    template_kind: str
    approved_template: str
    required_fact_ids: tuple[str, ...]
    allowed_fact_ids: tuple[str, ...]
    forbidden_fact_ids: tuple[str, ...]
    facts: dict[str, str]


@dataclass(frozen=True)
class AnswerVerification:
    passed: bool
    answer: str
    used_fact_ids: tuple[str, ...]
    reason: str


STOPWORDS = {
    "более", "будет", "быть", "вашего", "вашей", "вашим", "ваших", "если",
    "когда", "который", "можно", "нужно", "после", "перед", "также", "только",
    "через", "чтобы", "этого", "этот",
}
PROTECTED_VALUE_PATTERN = re.compile(
    r"https?://\S+|[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}|"
    r"(?:\+7|8)[\s\-()]*\d{3}[\s\-()]*\d{3}[\s\-]*\d{2}[\s\-]*\d{2}|"
    r"\b\d+(?:[.,]\d+)?\s*(?:₽|руб(?:лей|ля|ль)?|дн(?:ей|я|ь)|час(?:ов|а)?|%)?\b",
    flags=re.IGNORECASE,
)
PROMPT_INJECTION_OUTPUT_PATTERN = re.compile(
    r"\b(?:игнорир\w*\s+(?:предыдущ\w*|системн\w*)|системн\w*\s+инструкц\w*|"
    r"system\s+prompt|developer\s+message|как\s+языков\w+\s+модел\w+)\b|"
    r"\[[a-z0-9_.-]+\.fact\.\d+\]",
    re.IGNORECASE,
)
PROMISE_PATTERN = re.compile(
    r"\b(?:гарантир\w*|обеща\w*|точно\s+(?:вернут|передад|получит|выигра)|"
    r"безусловно\s+(?:вернут|передад|получит|выигра))\b",
    re.IGNORECASE,
)
SEMANTIC_MARKERS: dict[str, re.Pattern[str]] = {
    "negation": re.compile(r"\b(?:не|нет|нельзя|невозможно|запрещен\w*|без)\b", re.IGNORECASE),
    "obligation": re.compile(r"\b(?:должен\w*|обязан\w*|необходим\w*|требуется|нужно)\b", re.IGNORECASE),
    "possibility": re.compile(r"\b(?:можно|может|вправе|разрешен\w*)\b", re.IGNORECASE),
}


def _tokens(text: str) -> set[str]:
    return {
        token for token in normalize_matching_text(text).split()
        if len(token) >= 4 and token not in STOPWORDS
    }


def _protected_values(text: str) -> set[str]:
    return {match.group(0).casefold().rstrip(".,;:!?") for match in PROTECTED_VALUE_PATTERN.finditer(text)}


def _sentences(text: str) -> list[str]:
    return [item.strip() for item in re.split(r"(?<=[.!?])\s+|[\r\n]+", text) if item.strip()]


def _marker_names(text: str) -> set[str]:
    return {name for name, pattern in SEMANTIC_MARKERS.items() if pattern.search(text)}


def _scoped_semantics_supported(candidate: str, reference: str) -> bool:
    reference_sentences = _sentences(reference)
    for sentence in _sentences(candidate):
        markers = _marker_names(sentence)
        if not markers:
            continue
        sentence_tokens = _tokens(sentence)
        closest = max(
            reference_sentences,
            key=lambda item: len(sentence_tokens & _tokens(item)),
            default="",
        )
        if not markers.issubset(_marker_names(closest)):
            return False
    return True


@lru_cache(maxsize=1)
def load_answer_contracts() -> dict[str, AnswerContract]:
    path = get_settings().knowledge_root / "v3_1" / "answer_contracts.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        str(row["scenario_id"]): AnswerContract(
            scenario_id=str(row["scenario_id"]),
            template_kind=str(row["template_kind"]),
            approved_template=str(row["approved_template"]),
            required_fact_ids=tuple(str(item) for item in row["required_fact_ids"]),
            allowed_fact_ids=tuple(str(item) for item in row["allowed_fact_ids"]),
            forbidden_fact_ids=tuple(str(item) for item in row["forbidden_fact_ids"]),
            facts={str(key): str(value) for key, value in row["facts"].items()},
        )
        for row in payload["records"]
    }


def get_answer_contract(scenario_id: str) -> AnswerContract | None:
    return load_answer_contracts().get(scenario_id)


def fact_context(contract: AnswerContract) -> str:
    return "\n".join(
        f"[{fact_id}] {contract.facts[fact_id]}"
        for fact_id in contract.allowed_fact_ids
        if fact_id in contract.facts
    )


def verify_answer(candidate: str, fallback: str, contract: AnswerContract | None) -> AnswerVerification:
    if contract is None:
        return AnswerVerification(False, fallback, (), "missing_scenario_contract")
    if normalize_matching_text(candidate) == normalize_matching_text(fallback):
        return AnswerVerification(True, candidate, contract.required_fact_ids, "deterministic_approved_template")

    allowed_corpus = " ".join(contract.facts[fact_id] for fact_id in contract.allowed_fact_ids)
    reference = f"{contract.approved_template} {allowed_corpus}"
    if PROMPT_INJECTION_OUTPUT_PATTERN.search(candidate):
        return AnswerVerification(False, fallback, contract.required_fact_ids, "prompt_injection_output")
    allowed_values = _protected_values(reference)
    unsupported_values = _protected_values(candidate) - allowed_values
    if unsupported_values:
        return AnswerVerification(False, fallback, contract.required_fact_ids, "unsupported_protected_value")
    if PROMISE_PATTERN.search(candidate):
        return AnswerVerification(False, fallback, contract.required_fact_ids, "unsupported_promise")
    if not _scoped_semantics_supported(candidate, reference):
        return AnswerVerification(False, fallback, contract.required_fact_ids, "semantic_marker_changed")

    candidate_tokens = _tokens(candidate)
    allowed_tokens = _tokens(reference)
    lexical_support = len(candidate_tokens & allowed_tokens) / max(1, len(candidate_tokens))
    if lexical_support < 0.72:
        return AnswerVerification(False, fallback, contract.required_fact_ids, "insufficient_fact_support")

    used = tuple(
        fact_id for fact_id in contract.allowed_fact_ids
        if len(_tokens(contract.facts[fact_id]) & candidate_tokens) / max(1, len(_tokens(contract.facts[fact_id]))) >= 0.45
    )
    if not set(contract.required_fact_ids).issubset(used):
        return AnswerVerification(False, fallback, contract.required_fact_ids, "required_fact_missing")
    return AnswerVerification(True, candidate, used, "verified_wording_only")
