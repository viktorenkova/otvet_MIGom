from __future__ import annotations

from collections import Counter
from difflib import SequenceMatcher
import hashlib
import json
import re
from typing import Any


DEV_SESSION_PREFIXES = (
    "golden-",
    "test-",
    "variation-",
    "semantic-",
    "live-",
    "quality-",
    "perf-",
    "holdout-",
)

CONFIDENCE_RANK = {"high": 0, "medium": 1, "low": 2}


def _normalize_for_grouping(text: str) -> str:
    normalized = text.casefold().replace("ё", "е")
    normalized = re.sub(r"\b\d{3,}\b", "#", normalized)
    normalized = re.sub(r"[^a-zа-я0-9#\s]+", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def _is_similar(left: str, right: str) -> bool:
    if left == right:
        return True
    left_tokens = set(left.split())
    right_tokens = set(right.split())
    union = left_tokens | right_tokens
    jaccard = len(left_tokens & right_tokens) / len(union) if union else 0.0
    return jaccard >= 0.72 or SequenceMatcher(None, left, right).ratio() >= 0.88


def _parse_features(raw: Any) -> list[str]:
    if isinstance(raw, list):
        return [str(item) for item in raw]
    try:
        parsed = json.loads(str(raw or "[]"))
    except json.JSONDecodeError:
        return []
    return [str(item) for item in parsed] if isinstance(parsed, list) else []


def _signals_for_event(event: dict[str, Any], negative_ratings: int) -> set[str]:
    signals: set[str] = set()
    confidence = str(event.get("confidence") or "")
    action = str(event.get("action") or "")
    fallback = str(event.get("fallback_reason") or "")
    features = _parse_features(event.get("matched_features"))
    if confidence == "low":
        signals.add("низкая уверенность")
    elif confidence == "medium":
        signals.add("средняя уверенность")
    if action == "clarify":
        signals.add("бот запросил уточнение")
    if not event.get("article_id"):
        signals.add("статья не выбрана")
    if fallback and fallback not in {"safety_conflict", "clarification_other"}:
        signals.add(f"fallback: {fallback}")
    if "semantic_override" in features:
        signals.add("смысловой поиск изменил правило")
    if "semantic_clarification" in features:
        signals.add("смысловой поиск запросил уточнение")
    if negative_ratings:
        signals.add("оценка пользователя 1–2")
    return signals


def _priority_score(cluster: dict[str, Any]) -> int:
    signals = set(cluster["signals"])
    score = min(20, int(cluster["occurrences"]) * 4)
    if "низкая уверенность" in signals:
        score += 30
    elif "средняя уверенность" in signals:
        score += 15
    if "бот запросил уточнение" in signals:
        score += 20
    if "статья не выбрана" in signals:
        score += 15
    if "смысловой поиск изменил правило" in signals:
        score += 10
    if any(str(signal).startswith("fallback:") for signal in signals):
        score += 10
    score += min(30, int(cluster["negative_ratings"]) * 15)
    if not cluster.get("resolved_intent") and "бот запросил уточнение" in signals:
        score += 10
    return min(score, 100)


def _priority_label(score: int) -> str:
    if score >= 70:
        return "Критический"
    if score >= 45:
        return "Высокий"
    if score >= 25:
        return "Средний"
    return "Низкий"


def build_review_queue(
    events: list[dict[str, Any]],
    feedback_by_session: dict[str, list[int]],
    *,
    include_dev_sessions: bool = False,
) -> list[dict[str, Any]]:
    choices_by_session: dict[str, list[dict[str, Any]]] = {}
    for event in events:
        features = _parse_features(event.get("matched_features"))
        if "clarification_choice" in features:
            choices_by_session.setdefault(str(event.get("session_id")), []).append(event)

    clusters: list[dict[str, Any]] = []
    for event in events:
        session_id = str(event.get("session_id") or "")
        if not include_dev_sessions and session_id.startswith(DEV_SESSION_PREFIXES):
            continue
        action = str(event.get("action") or "")
        features = _parse_features(event.get("matched_features"))
        if action == "safety_refusal" or "clarification_choice" in features or "clarification_choice:other" in features:
            continue

        ratings = [int(value) for value in feedback_by_session.get(session_id, []) if value is not None]
        negative_ratings = sum(1 for rating in ratings if rating <= 2)
        signals = _signals_for_event(event, negative_ratings)
        if not signals:
            continue

        redacted_message = str(event.get("redacted_message") or "").strip()
        normalized = _normalize_for_grouping(redacted_message)
        if not normalized:
            continue
        intent = str(event.get("intent") or "unknown")
        target = next(
            (
                cluster
                for cluster in clusters
                if cluster["group_intent"] == intent
                and _is_similar(cluster["normalized_representative"], normalized)
            ),
            None,
        )
        later_choices = [
            choice
            for choice in choices_by_session.get(session_id, [])
            if int(choice.get("id") or 0) > int(event.get("id") or 0)
        ]
        resolved_intent = str(later_choices[0].get("intent") or "") if later_choices else ""
        if target is None:
            target = {
                "normalized_representative": normalized,
                "group_intent": intent,
                "variations": [],
                "occurrences": 0,
                "signals": set(),
                "intents": Counter(),
                "article_ids": Counter(),
                "confidences": Counter(),
                "negative_ratings": 0,
                "ratings": [],
                "first_seen": str(event.get("created_at") or ""),
                "last_seen": str(event.get("created_at") or ""),
                "resolved_intent": resolved_intent,
            }
            clusters.append(target)
        if redacted_message not in target["variations"] and len(target["variations"]) < 30:
            target["variations"].append(redacted_message)
        target["occurrences"] += 1
        target["signals"].update(signals)
        target["intents"][intent] += 1
        if event.get("article_id"):
            target["article_ids"][str(event["article_id"])] += 1
        target["confidences"][str(event.get("confidence") or "unknown")] += 1
        target["negative_ratings"] += negative_ratings
        target["ratings"].extend(ratings)
        target["first_seen"] = min(target["first_seen"], str(event.get("created_at") or ""))
        target["last_seen"] = max(target["last_seen"], str(event.get("created_at") or ""))
        if not target["resolved_intent"] and resolved_intent:
            target["resolved_intent"] = resolved_intent

    queue: list[dict[str, Any]] = []
    for cluster in clusters:
        score = _priority_score(cluster)
        worst_confidence = max(
            cluster["confidences"],
            key=lambda value: CONFIDENCE_RANK.get(value, -1),
        )
        representative = min(cluster["variations"], key=lambda value: (len(value), value))
        queue.append(
            {
                "cluster_id": hashlib.sha1(
                    f"{cluster['group_intent']}:{cluster['normalized_representative']}".encode("utf-8")
                ).hexdigest()[:10],
                "priority": _priority_label(score),
                "priority_score": score,
                "representative_question": representative,
                "variations": list(cluster["variations"]),
                "occurrences": int(cluster["occurrences"]),
                "current_intent": cluster["intents"].most_common(1)[0][0],
                "resolved_intent": cluster["resolved_intent"],
                "article_id": cluster["article_ids"].most_common(1)[0][0] if cluster["article_ids"] else "",
                "confidence": worst_confidence,
                "signals": sorted(cluster["signals"]),
                "negative_ratings": int(cluster["negative_ratings"]),
                "average_rating": (
                    round(sum(cluster["ratings"]) / len(cluster["ratings"]), 2)
                    if cluster["ratings"]
                    else None
                ),
                "first_seen": cluster["first_seen"],
                "last_seen": cluster["last_seen"],
            }
        )
    return sorted(
        queue,
        key=lambda item: (int(item["priority_score"]), int(item["occurrences"]), item["last_seen"]),
        reverse=True,
    )
