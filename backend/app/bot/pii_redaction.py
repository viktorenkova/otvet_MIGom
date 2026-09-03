from __future__ import annotations

import re


PII_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"https?://\S+|www\.\S+|t\.me/\S+", re.IGNORECASE), "[url]"),
    (re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}"), "[email]"),
    (re.compile(r"(?<!\d)(?:\+?7|8)[\s\-()]*\d(?:[\s\-()]*\d){9}(?!\d)"), "[телефон]"),
    (re.compile(r"(?<!\w)@[A-Za-z0-9_]{5,}"), "[telegram]"),
    (
        re.compile(
            r"\b[a-z0-9._-]*(?:xtt|\s+at\s+|\s+собака\s+)[a-z0-9._-]+"
            r"(?:\s*\.\s*|\s+)(?:ru|com)\b",
            re.IGNORECASE,
        ),
        "[контакт]",
    ),
    (re.compile(r"(?i)(?<![A-Z0-9])[A-HJ-NPR-Z0-9]{17}(?![A-Z0-9])"), "[VIN]"),
    (
        re.compile(
            r"(?<![А-ЯЁA-Z0-9])[АВЕКМНОРСТУХABEKMHOPCTYX]\s*\d{3}\s*"
            r"[АВЕКМНОРСТУХABEKMHOPCTYX]{2}\s*\d{2,3}(?!\d)",
            re.IGNORECASE,
        ),
        "[госномер]",
    ),
    (
        re.compile(
            r"\b(?:ФИО|имя|клиент|клиента|покупатель)\s*[:\-]?\s*"
            r"[А-ЯЁ][а-яё]+(?:\s+[А-ЯЁ][а-яё]+){0,2}",
            re.IGNORECASE,
        ),
        "[имя]",
    ),
    (
        re.compile(
            r"\b(?:адрес|проживаю|нахожусь)\s*[:\-]?\s+[^.!?\n]{4,80}",
            re.IGNORECASE,
        ),
        "[адрес]",
    ),
    (
        re.compile(
            r"\b(?:паспорт|инн|снилс)\s*[:№\-]?\s*[A-Za-zА-Яа-я0-9\- ]{4,30}",
            re.IGNORECASE,
        ),
        "[документ]",
    ),
    (
        re.compile(
            r"\b(?:лот|договор|заявка|плат[её]ж|обращение)\s*№?\s*"
            r"[A-Za-zА-Яа-я0-9][A-Za-zА-Яа-я0-9\-_/]{3,}\b",
            re.IGNORECASE,
        ),
        "[идентификатор]",
    ),
    (re.compile(r"\b\d{4,}\b"), "[идентификатор]"),
)


def redact_for_external_llm(text: str) -> str:
    result = text
    for pattern, replacement in PII_PATTERNS:
        result = pattern.sub(replacement, result)
    return re.sub(r"\s+", " ", result).strip()


def detected_pii_kinds(text: str) -> tuple[str, ...]:
    return tuple(replacement for pattern, replacement in PII_PATTERNS if pattern.search(text))
