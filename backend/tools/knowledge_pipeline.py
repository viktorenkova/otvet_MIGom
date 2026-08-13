from __future__ import annotations

import argparse
from collections import Counter
import csv
from dataclasses import dataclass
from datetime import datetime
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import sqlite3
from typing import Any, Iterable

from backend.app.bot.text_processing import normalize_text, tokenize
from backend.app.bot.scenario_engine import match_scenario


@dataclass(frozen=True)
class ImportedMessage:
    text: str
    source: str
    fallback: bool = False
    negative_feedback: bool = False
    source_message_id: str | None = None
    conversation_id: str | None = None
    created_at: str | None = None
    speaker_key: str | None = None
    message_kind: str = "candidate"
    categories: tuple[str, ...] = ()


_URL_PATTERN = re.compile(r"https?://\S+|www\.\S+|t\.me/\S+", re.IGNORECASE)
_EMAIL_PATTERN = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
_PHONE_PATTERN = re.compile(r"(?<!\d)(?:\+?7|8)[\s\-()]*\d(?:[\s\-()]*\d){9}(?!\d)")
_TELEGRAM_PATTERN = re.compile(r"(?<!\w)@[A-Za-z0-9_]{5,}")
_OBFUSCATED_CONTACT_PATTERN = re.compile(
    r"\b[a-z0-9._-]*(?:xtt|\s+at\s+|\s+собака\s+)[a-z0-9._-]+(?:\s*\.\s*|\s+)(?:ru|com)\b",
    re.IGNORECASE,
)
_LABELLED_NAME_PATTERN = re.compile(
    r"\b(ФИО|фио|Имя|имя)\s*[:\-]?\s*"
    r"[А-ЯЁ][а-яё]+(?:\s+[А-ЯЁ][а-яё]+){0,2}"
)
_CLIENT_NAME_PATTERN = re.compile(
    r"\b(Клиент|клиент|Клиента|клиента|Покупатель|покупатель)\s*[:\-]?\s*"
    r"[А-ЯЁ][а-яё]+\s+[А-ЯЁ][а-яё]+(?:\s+[А-ЯЁ][а-яё]+)?"
)
_LONG_IDENTIFIER_PATTERN = re.compile(r"\b\d{4,}\b")

PII_PATTERNS = (
    (_URL_PATTERN, "[url]"),
    (_EMAIL_PATTERN, "[email]"),
    (_PHONE_PATTERN, "[phone]"),
    (_TELEGRAM_PATTERN, "[telegram]"),
    (_OBFUSCATED_CONTACT_PATTERN, "[contact]"),
    (
        _LABELLED_NAME_PATTERN,
        r"\1: [name]",
    ),
    (
        _CLIENT_NAME_PATTERN,
        r"\1: [name]",
    ),
    (
        re.compile(r"\b(лот|заявка|плат[её]ж|обращение)\s*№?\s*\d{4,}\b", re.IGNORECASE),
        r"\1 [identifier]",
    ),
    (_LONG_IDENTIFIER_PATTERN, "[identifier]"),
)

_MESSAGE_ID_RE = re.compile(r"message-?\d+", re.IGNORECASE)
_DATE_FORMAT = "%d.%m.%Y %H:%M:%S"
_TOPIC_PATTERNS: dict[str, re.Pattern[str]] = {
    "registration_access": re.compile(
        r"регистрац|аккаунт|личн(?:ый|ого) кабинет|логин|парол|код|войти|доступ",
        re.IGNORECASE,
    ),
    "tariff_payment": re.compile(
        r"тариф|оплат|плат[её]ж|кошел|баланс|комисси|сч[её]т|списал",
        re.IGNORECASE,
    ),
    "lot_vehicle_info": re.compile(
        r"\bлот|автомоб|машин|\bvin\b|\bвин\b|карточк|фото|осмотр",
        re.IGNORECASE,
    ),
    "bid_auction": re.compile(
        r"ставк|торг|аукцион|котиров|позици|стартов(?:ая|ой) цен",
        re.IGNORECASE,
    ),
    "win_transfer_pickup": re.compile(
        r"выигр|побед|передач|забрат|получ|выдач|договор|документ",
        re.IGNORECASE,
    ),
    "seller_flow": re.compile(r"продав|продаж|выставить|разместить|размещен", re.IGNORECASE),
    "contact_support": re.compile(r"перезвон|позвон|связат|обращени|поддержк", re.IGNORECASE),
    "technical_problem": re.compile(
        r"ошибк|не работает|не открыв|не отображ|не видн|пропал|слетел|завис",
        re.IGNORECASE,
    ),
    "refund_dispute": re.compile(r"возврат|отказ|штраф|спор|претензи|расторж", re.IGNORECASE),
}
_QUESTION_SIGNAL_RE = re.compile(
    r"\?|\bкак\b|\bпочему\b|\bгде\b|\bкогда\b|\bможно\b|\bнужно\b|"
    r"\bне\s+\w+|просит|вопрос|проблем|необходимо|хочет",
    re.IGNORECASE,
)
_ACKNOWLEDGEMENT_RE = re.compile(
    r"^(?:добрый\s+(?:день|вечер|утро)[!. ]*|спасибо[!. ]*|"
    r"принял[аи]?[!. ]*|взял[аи]?\s+в\s+работу[!. ]*|ок[!. ]*)$",
    re.IGNORECASE,
)
_SYSTEM_RE = re.compile(
    r"created group|converted (?:this )?group|joined group|added .+ group|"
    r"создал[аи]? групп|добавил[аи]? .+ групп|закрепил[аи]? сообщ",
    re.IGNORECASE,
)
_ASSIGNEE_RE = re.compile(
    r"\b(?:Для|для)\s+(?P<name>[А-ЯЁ][а-яё]+(?:\s+[А-ЯЁ][а-яё]+)?)\s*:?",
    re.UNICODE,
)
_ROUTING_NOISE_RE = re.compile(
    r"\(?\s*(?:соед(?:\.|инил[аи]?|инить)?|попросил[аи]?\s+(?:соединить|связаться)"
    r"(?:\s+с\s+(?:ним|ней|менеджером))?|просил[аи]?\s+связь\s+с\s+менеджером)\s*\)?",
    re.IGNORECASE,
)
_PHONE_ADJACENT_NAME_RE = re.compile(
    r"(?<=\[phone\]\s)(?:(?:[А-ЯЁ][а-яё]+\s+){1,2}[А-ЯЁ][а-яё]+|"
    r"[А-ЯЁ][а-яё]+(?=\s*(?:[:,.]|просит|говорит|спрашивает|хочет|лот\b|не\b)))"
)
_CAPITALIZED_SEQUENCE_RE = re.compile(r"(?<!\w)(?:[А-ЯЁ][а-яё]+\s+){1,2}[А-ЯЁ][а-яё]+(?!\w)")
_CAPITALIZED_TOKEN_RE = re.compile(r"(?<!\w)[А-ЯЁ][а-яё]{2,}(?!\w)")
_PUBLIC_ENTITY_ALLOWLIST = {
    "Мигторг",
    "Телеграм",
}


@dataclass
class _HtmlNode:
    tag: str
    attrs: dict[str, str]
    children: list[Any]


@dataclass(frozen=True)
class _HtmlMessage:
    message_id: str
    created_at: str | None
    author: str | None
    text: str
    is_system: bool


class _WordTelegramHtmlParser(HTMLParser):
    _VOID_TAGS = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "source"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.messages: list[_HtmlNode] = []
        self._root: _HtmlNode | None = None
        self._stack: list[_HtmlNode] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = {key: value or "" for key, value in attrs}
        if self._root is None:
            message_id = attributes.get("id", "")
            if tag == "div" and _MESSAGE_ID_RE.fullmatch(message_id):
                self._root = _HtmlNode(tag, attributes, [])
                self._stack = [self._root]
            return

        node = _HtmlNode(tag, attributes, [])
        self._stack[-1].children.append(node)
        if tag == "br":
            node.children.append("\n")
        if tag not in self._VOID_TAGS:
            self._stack.append(node)

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)

    def handle_endtag(self, tag: str) -> None:
        if self._root is None:
            return
        if len(self._stack) == 1 and tag == self._root.tag:
            self.messages.append(self._root)
            self._root = None
            self._stack = []
            return
        for index in range(len(self._stack) - 1, 0, -1):
            if self._stack[index].tag == tag:
                del self._stack[index:]
                return

    def handle_data(self, data: str) -> None:
        if self._root is not None and data:
            self._stack[-1].children.append(data)


def _node_text(node: _HtmlNode) -> str:
    parts: list[str] = []
    for child in node.children:
        parts.append(_node_text(child) if isinstance(child, _HtmlNode) else child)
    return re.sub(r"[ \t\r\f\v]+", " ", "".join(parts)).strip()


def _leaf_divs(node: _HtmlNode) -> list[_HtmlNode]:
    result: list[_HtmlNode] = []
    if node.tag == "div" and not any(isinstance(child, _HtmlNode) and child.tag == "div" for child in node.children):
        return [node]
    for child in node.children:
        if isinstance(child, _HtmlNode):
            result.extend(_leaf_divs(child))
    return result


def _looks_like_author(text: str) -> bool:
    compact = " ".join(text.split())
    return bool(
        compact
        and len(compact) <= 80
        and len(compact.split()) <= 6
        and not any(char in compact for char in "?!:\n")
        and not re.search(r"\d", compact)
    )


def _parse_created_at(value: str | None) -> str | None:
    if not value:
        return None
    cleaned = re.sub(r"\s+UTC[+\-]\d{2}:\d{2}$", "", value).strip()
    try:
        return datetime.strptime(cleaned, _DATE_FORMAT).isoformat()
    except ValueError:
        return None


def _decode_html(path: Path) -> str:
    raw = path.read_bytes()
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        return raw.decode("utf-16")
    for encoding in ("utf-8-sig", "utf-16", "cp1251"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def _extract_html_messages(path: Path) -> list[_HtmlMessage]:
    parser = _WordTelegramHtmlParser()
    parser.feed(_decode_html(path))
    result: list[_HtmlMessage] = []
    previous_author: str | None = None
    for root in parser.messages:
        leaves = [(leaf, " ".join(_node_text(leaf).split())) for leaf in _leaf_divs(root)]
        leaves = [(leaf, text) for leaf, text in leaves if text]
        date_index = next((index for index, (leaf, _) in enumerate(leaves) if leaf.attrs.get("title")), None)
        created_at = _parse_created_at(leaves[date_index][0].attrs.get("title")) if date_index is not None else None
        after_date = leaves[date_index + 1 :] if date_index is not None else []
        author: str | None = None
        if len(after_date) >= 2 and _looks_like_author(after_date[0][1]):
            author = after_date[0][1]
            content_parts = [text for _, text in after_date[1:]]
        else:
            author = previous_author if created_at else None
            content_parts = [text for _, text in after_date]
        if author:
            previous_author = author
        text = " ".join(dict.fromkeys(part for part in content_parts if part)).strip()
        if not text and date_index is None:
            text = " ".join(dict.fromkeys(text for _, text in leaves)).strip()
        result.append(
            _HtmlMessage(
                message_id=root.attrs.get("id", ""),
                created_at=created_at,
                author=author,
                text=text,
                is_system=not created_at or bool(_SYSTEM_RE.search(text)),
            )
        )
    return result


def redact(text: str) -> str:
    result = text
    for pattern, replacement in PII_PATTERNS:
        result = pattern.sub(replacement, result)
    return re.sub(r"\s+", " ", result).strip()[:500]


def redact_freeform_entities(text: str) -> str:
    """Conservatively remove proper names that are not captured by field labels."""
    result = _CAPITALIZED_SEQUENCE_RE.sub("[entity]", text)

    def replace_token(match: re.Match[str]) -> str:
        token = match.group(0)
        if token in _PUBLIC_ENTITY_ALLOWLIST:
            return token
        prefix = result[: match.start()].rstrip()
        if not prefix or prefix.endswith((".", "!", "?", ":")):
            return token
        return "[entity]"

    return _CAPITALIZED_TOKEN_RE.sub(replace_token, result)


def clean_support_text(text: str) -> str:
    """Remove internal call-routing boilerplate while retaining the reported issue."""
    result = _ASSIGNEE_RE.sub("[assignee] ", text)
    result = re.sub(
        r"(?<!\w)(?:[А-ЯЁ][а-яё]+(?:\s+|,\s*)){1,3}(?=\[phone\])",
        "[name] ",
        result,
    )
    result = _PHONE_ADJACENT_NAME_RE.sub("[name]", result)
    result = re.sub(r"(?<=\[phone\]\s)[А-ЯЁ][а-яё]+(?=\s+соед)", "[name]", result)
    result = _ROUTING_NOISE_RE.sub(" ", result)
    result = re.sub(
        r"\b(?:обращение\s+действу(?:ю|е)щего\s+клиента|суть\s+обращения|"
        r"содержание\s+заявки|согласие\s+с\s+политик(?:ой|и)\s+конфиденциальности|"
        r"yes\s+подтверждаем)\b\s*[:\-]?",
        " ",
        result,
        flags=re.IGNORECASE,
    )
    result = re.sub(r"\[(?:assignee|phone|name|participant|contact|entity)\]", " ", result)
    result = re.sub(r"\s*[,;:]\s*$", "", result)
    return re.sub(r"\s+", " ", result).strip(" ,.;:—-")


def _categories(text: str) -> tuple[str, ...]:
    return tuple(name for name, pattern in _TOPIC_PATTERNS.items() if pattern.search(text))


def _meaningful_text(text: str) -> str:
    cleaned = re.sub(r"\[(?:phone|email|url|telegram|identifier|name|participant|contact|entity)\]", " ", text)
    return " ".join(cleaned.split()).strip(" ,.;:—-")


def _message_kind(text: str, *, is_system: bool, categories: tuple[str, ...]) -> str:
    meaningful = _meaningful_text(text)
    if is_system or _SYSTEM_RE.search(meaningful):
        return "system"
    if not meaningful or len(tokenize(normalize_text(meaningful))) < 2:
        return "contact_only"
    if _ACKNOWLEDGEMENT_RE.fullmatch(meaningful):
        return "acknowledgement"
    if categories and _QUESTION_SIGNAL_RE.search(meaningful):
        return "candidate"
    if categories:
        return "domain_context"
    return "other"


def _speaker_key(author: str | None, speakers: dict[str, str]) -> str | None:
    if not author:
        return None
    normalized = normalize_text(author)
    if not normalized:
        return None
    if normalized not in speakers:
        speakers[normalized] = f"speaker-{len(speakers) + 1:03d}"
    return speakers[normalized]


def normalize_html_messages(path: Path) -> list[ImportedMessage]:
    """Parse a Word-saved Telegram export without retaining participant names."""
    messages = _extract_html_messages(path)
    speakers: dict[str, str] = {}
    assignee_names = {
        match.group("name")
        for message in messages
        for match in _ASSIGNEE_RE.finditer(message.text)
    }
    participant_names = sorted(
        {
            *{message.author for message in messages if message.author and len(message.author.strip()) >= 3},
            *assignee_names,
        },
        key=len,
        reverse=True,
    )
    result: list[ImportedMessage] = []
    current_conversation: str | None = None
    for message in messages:
        raw_has_contact = bool(_PHONE_PATTERN.search(message.text))
        if raw_has_contact:
            current_conversation = f"{path.stem}:{message.message_id}"
        text_without_participants = message.text
        for participant_name in participant_names:
            name_stem = re.escape(participant_name[: max(4, len(participant_name) - 2)])
            text_without_participants = re.sub(
                rf"(?<!\w)(?:{re.escape(participant_name)}|{name_stem}[А-Яа-яЁё]*)(?!\w)",
                "[participant]",
                text_without_participants,
                flags=re.IGNORECASE,
            )
        safe_text = clean_support_text(redact_freeform_entities(redact(text_without_participants)))
        categories = _categories(safe_text)
        kind = _message_kind(safe_text, is_system=message.is_system, categories=categories)
        result.append(
            ImportedMessage(
                text=safe_text,
                source=path.name,
                source_message_id=message.message_id,
                conversation_id=current_conversation,
                created_at=message.created_at,
                speaker_key=_speaker_key(message.author, speakers),
                message_kind=kind,
                categories=categories,
            )
        )
    return result


def _load_html(path: Path) -> list[ImportedMessage]:
    return [item for item in normalize_html_messages(path) if item.message_kind == "candidate" and item.text]


def _load_sqlite(path: Path) -> list[ImportedMessage]:
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    rows = con.execute(
        """
        SELECT m.id, m.message,
               EXISTS(
                 SELECT 1 FROM quality_events q
                 WHERE q.session_id = m.session_id
                   AND q.created_at = m.created_at
                   AND (q.action = 'clarify' OR COALESCE(q.fallback_reason, '') <> '')
               ) AS fallback,
               EXISTS(
                 SELECT 1 FROM feedback f
                 WHERE f.session_id = m.session_id AND f.rating <= 2
               ) AS negative_feedback
        FROM dialog_messages m
        WHERE m.role IN ('guest', 'authorized')
        ORDER BY m.id
        """
    ).fetchall()
    con.close()
    return [
        ImportedMessage(redact(str(row["message"])), "sqlite", bool(row["fallback"]), bool(row["negative_feedback"]))
        for row in rows
        if str(row["message"] or "").strip()
    ]


def _load_jsonl(path: Path) -> list[ImportedMessage]:
    result = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        text = str(item.get("message") or item.get("text") or "").strip()
        if text:
            result.append(ImportedMessage(redact(text), path.name))
    return result


def _load_csv(path: Path) -> list[ImportedMessage]:
    result = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for item in csv.DictReader(handle):
            text = str(item.get("message") or item.get("text") or item.get("transcript") or "").strip()
            if text:
                result.append(ImportedMessage(redact(text), path.name))
    return result


def load_messages(path: Path) -> list[ImportedMessage]:
    if path.suffix.casefold() in {".sqlite", ".sqlite3", ".db"}:
        return _load_sqlite(path)
    if path.suffix.casefold() == ".jsonl":
        return _load_jsonl(path)
    if path.suffix.casefold() == ".csv":
        return _load_csv(path)
    if path.suffix.casefold() in {".html", ".htm"}:
        return _load_html(path)
    raise ValueError(f"Unsupported input format: {path.suffix}")


def load_many(paths: Iterable[Path]) -> list[ImportedMessage]:
    result: list[ImportedMessage] = []
    for path in paths:
        result.extend(load_messages(path))
    return result


def _signature(text: str) -> frozenset[str]:
    return frozenset(term for term in tokenize(normalize_text(_meaningful_text(text))) if len(term) > 2)


def cluster_messages(messages: Iterable[ImportedMessage], threshold: float = 0.72) -> list[list[ImportedMessage]]:
    clusters: list[tuple[frozenset[str], list[ImportedMessage]]] = []
    for message in messages:
        signature = _signature(message.text)
        if not signature:
            continue
        best_index = -1
        best_score = 0.0
        for index, (centroid, _) in enumerate(clusters):
            score = len(signature & centroid) / max(1, len(signature | centroid))
            if score > best_score:
                best_index, best_score = index, score
        if best_index >= 0 and best_score >= threshold:
            centroid, items = clusters[best_index]
            items.append(message)
            clusters[best_index] = (centroid | signature, items)
        else:
            clusters.append((signature, [message]))
    return [items for _, items in clusters]


def build_candidates(messages: list[ImportedMessage]) -> dict:
    candidates = []
    for cluster in sorted(cluster_messages(messages), key=len, reverse=True):
        examples = list(dict.fromkeys(item.text for item in cluster))[:10]
        decisions = [match_scenario(item.text, "guest") for item in cluster]
        matched = Counter(
            decision.scenario.scenario_id
            for decision in decisions
            if decision.scenario and decision.confidence == "high"
        )
        dominant, dominant_count = matched.most_common(1)[0] if matched else (None, 0)
        coverage = dominant_count / len(cluster)
        digest = hashlib.sha256("\n".join(examples).encode("utf-8")).hexdigest()[:12]
        candidates.append(
            {
                "cluster_id": digest,
                "volume": len(cluster),
                "fallback_rate": round(sum(item.fallback for item in cluster) / len(cluster), 3),
                "negative_feedback_rate": round(sum(item.negative_feedback for item in cluster) / len(cluster), 3),
                "matched_scenario_id": dominant if coverage >= 0.6 else None,
                "matched_scenario_coverage": round(coverage, 3),
                "review_status": "needs_review",
                "examples": examples,
                "categories": dict(
                    Counter(category for item in cluster for category in item.categories).most_common()
                ),
                "source_files": dict(Counter(item.source for item in cluster).most_common()),
            }
        )
    covered = sum(item["volume"] for item in candidates if item["matched_scenario_id"])
    return {
        "schema_version": 1,
        "publication_allowed": False,
        "messages": len(messages),
        "covered_messages": covered,
        "estimated_coverage": round(covered / len(messages), 3) if messages else 0.0,
        "candidates": candidates,
    }


def _privacy_audit(messages: list[ImportedMessage]) -> dict[str, int]:
    joined = "\n".join(item.text for item in messages)
    return {
        "email_matches": len(_EMAIL_PATTERN.findall(joined)),
        "phone_matches": len(_PHONE_PATTERN.findall(joined)),
        "url_matches": len(_URL_PATTERN.findall(joined)),
        "telegram_handle_matches": len(_TELEGRAM_PATTERN.findall(joined)),
        "obfuscated_contact_matches": len(_OBFUSCATED_CONTACT_PATTERN.findall(joined)),
        "labelled_name_matches": len(_LABELLED_NAME_PATTERN.findall(joined))
        + len(_CLIENT_NAME_PATTERN.findall(joined)),
        "long_identifier_matches": len(_LONG_IDENTIFIER_PATTERN.findall(joined)),
    }


def _ingestion_summary(messages: list[ImportedMessage]) -> dict[str, Any]:
    dates = sorted(item.created_at for item in messages if item.created_at)
    return {
        "source_messages": len(messages),
        "candidate_messages": sum(item.message_kind == "candidate" for item in messages),
        "message_kinds": dict(Counter(item.message_kind for item in messages).most_common()),
        "categories": dict(
            Counter(category for item in messages for category in item.categories).most_common()
        ),
        "source_files": dict(Counter(item.source for item in messages).most_common()),
        "date_start": dates[0] if dates else None,
        "date_end": dates[-1] if dates else None,
        "privacy_audit": _privacy_audit(messages),
    }


def _write_normalized(messages: list[ImportedMessage], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for item in messages:
            handle.write(
                json.dumps(
                    {
                        "source": item.source,
                        "source_message_id": item.source_message_id,
                        "conversation_id": item.conversation_id,
                        "created_at": item.created_at,
                        "speaker_key": item.speaker_key,
                        "message_kind": item.message_kind,
                        "categories": list(item.categories),
                        "text_redacted": item.text,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a manually reviewed scenario backlog from support traffic.")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--include", type=Path, action="append", default=[])
    parser.add_argument("--normalized-output", type=Path)
    args = parser.parse_args()
    inputs = [args.input, *args.include]
    messages: list[ImportedMessage] = []
    normalized: list[ImportedMessage] = []
    for path in inputs:
        if path.suffix.casefold() in {".html", ".htm"}:
            imported = normalize_html_messages(path)
            normalized.extend(imported)
            messages.extend(item for item in imported if item.message_kind == "candidate" and item.text)
        else:
            imported = load_messages(path)
            normalized.extend(imported)
            messages.extend(imported)
    report = build_candidates(messages)
    report["inputs"] = [path.name for path in inputs]
    report["ingestion"] = _ingestion_summary(normalized)
    report["privacy_audit"] = report["ingestion"]["privacy_audit"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.normalized_output:
        _write_normalized(normalized, args.normalized_output)
    print(
        json.dumps(
            {
                key: report[key]
                for key in ("messages", "covered_messages", "estimated_coverage", "privacy_audit")
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
