from __future__ import annotations

import argparse
import base64
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import hmac
import json
import math
import os
from pathlib import Path
import re
import tempfile
import time
from typing import Any, Callable

from backend.tools.evaluate_live_queries import build_local_sender


DEFAULT_SOURCE = Path(".work/stage5-blind-pilot-review-draft-2026-09-01.json")
DEFAULT_REPORT = Path("reports/stage5-development-evaluation.json")
DEFAULT_DETAILS = Path(".work/stage5-development-evaluation-details.json")
APPROVED = "approved"
WORD_RE = re.compile(r"[0-9a-zа-яё]+", re.IGNORECASE)
ACKNOWLEDGEMENT_RE = re.compile(r"^\s*(?:спасибо|благодарю)\b", re.IGNORECASE)
CRM_NOTE_RE = re.compile(
    r"\b(?:обращение\s+(?:от|не\s+от)|клиент\w*|пользователь\w*|просит|ожидает|"
    r"набирает|жд[её]т\s+(?:звонка|обратн\w*\s+связ))\b",
    re.IGNORECASE,
)
LOCAL_TRUSTED_CONTEXT_SECRET = "stage5-local-development-context-secret-v1"


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _trusted_context_token(secret: str, issuer: str = "migtorg-site") -> str:
    payload = {
        "iss": issuer,
        "sub": "stage5-development-evaluator",
        "scopes": [],
        "exp": int(time.time()) + 3600,
    }
    raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    part = base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")
    signature = hmac.new(secret.encode("utf-8"), part.encode("ascii"), hashlib.sha256).digest()
    return part + "." + base64.urlsafe_b64encode(signature).decode("ascii").rstrip("=")


def _rate(passed: int, total: int) -> float:
    return round(passed / total * 100.0, 2) if total else 0.0


def _wilson_lower(passed: int, total: int, z: float = 1.959963984540054) -> float:
    if not total:
        return 0.0
    observed = passed / total
    denominator = 1 + z * z / total
    centre = observed + z * z / (2 * total)
    margin = z * math.sqrt(observed * (1 - observed) / total + z * z / (4 * total * total))
    return round((centre - margin) / denominator * 100.0, 2)


def _approved(review: dict[str, Any] | None) -> bool:
    return str((review or {}).get("status") or "") == APPROVED


def load_approved(source: Path) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    raw = source.read_bytes()
    payload = json.loads(raw.decode("utf-8"))
    singles = [item for item in payload.get("cases", []) if _approved(item.get("review"))]
    dialogues: list[dict[str, Any]] = []
    for dialogue in payload.get("dialogues", []):
        turns = [turn for turn in dialogue.get("turns", []) if _approved(turn.get("review"))]
        if turns:
            dialogues.append({**dialogue, "turns": turns})
    metadata = {
        "path": str(source),
        "sha256": _sha256(raw),
        "dataset_version": payload.get("dataset_version"),
        "single_count": len(singles),
        "dialogue_count": len(dialogues),
        "dialogue_turn_count": sum(len(item["turns"]) for item in dialogues),
    }
    return metadata, singles, dialogues


def _actual_action(response: dict[str, Any]) -> str:
    action = str(response.get("action") or "")
    resolution = str(response.get("resolution") or "")
    if action == "safety_refusal":
        return "safe_refusal"
    if resolution == "out_of_scope":
        return "out_of_scope"
    if response.get("needs_ticket") or resolution == "escalated" or action in {
        "create_ticket", "request_callback", "handoff"
    }:
        return "support"
    if response.get("scenario_id") in {"support.contact", "support.callback"}:
        return "support"
    if action == "clarify" or resolution == "clarified":
        return "clarify"
    return "answer"


def _word_set(value: str) -> set[str]:
    return {match.group(0).casefold() for match in WORD_RE.finditer(value) if len(match.group(0)) > 2}


def _semantic_overlap(answer: str, required: list[str]) -> float | None:
    """Diagnostic lexical overlap only; it is deliberately not a release gate."""
    if not required:
        return None
    answer_words = _word_set(answer)
    expected_words = _word_set(" ".join(required))
    if not answer_words or not expected_words:
        return 0.0
    return round(len(answer_words & expected_words) / len(answer_words) * 100.0, 2)


def _evaluate_item(
    item: dict[str, Any],
    response: dict[str, Any] | None,
    error: str,
    dialogue_id: str | None = None,
) -> dict[str, Any]:
    expected = dict(item.get("expected") or {})
    actual = response or {}
    expected_action = str(expected.get("bot_action") or "")
    actual_action = _actual_action(actual) if response is not None else "transport_error"
    answer = str(actual.get("answer") or "")
    forbidden = [str(value) for value in expected.get("forbidden_information", [])]
    forbidden_hits = [value for value in forbidden if value.casefold() in answer.casefold()]
    expected_handoff = expected.get("support_handoff") if dialogue_id else None
    actual_handoff = bool(
        actual.get("needs_ticket")
        or actual.get("resolution") == "escalated"
        or actual.get("action") in {"create_ticket", "request_callback", "handoff"}
    )
    text = str(item.get("text") or "")
    situation = str(expected.get("specific_situation") or "")
    source_quality_flags = []
    if "ошибочно размечен как обращение" in situation.casefold():
        source_quality_flags.append("explicit_employee_response")
    if ACKNOWLEDGEMENT_RE.search(text):
        source_quality_flags.append("acknowledgement_not_question")
    if CRM_NOTE_RE.search(text):
        source_quality_flags.append("crm_case_note_style")
    action_evaluable = not {
        "explicit_employee_response", "acknowledgement_not_question"
    }.intersection(source_quality_flags)
    return {
        "id": item["id"],
        "dialogue_id": dialogue_id,
        "text": text,
        "expected": expected,
        "response": actual,
        "error": error,
        "checks": {
            "transport_ok": response is not None and not error,
            "action_ok": (
                response is not None and actual_action == expected_action
                if action_evaluable else None
            ),
            "forbidden_content_ok": not forbidden_hits,
            "support_handoff_ok": (
                None if expected_handoff is None else bool(expected_handoff) == actual_handoff
            ),
        },
        "diagnostics": {
            "expected_action": expected_action,
            "actual_action": actual_action,
            "scenario_id": actual.get("scenario_id"),
            "intent": actual.get("intent"),
            "resolution": actual.get("resolution"),
            "confidence_level": actual.get("confidence_level"),
            "forbidden_hits": forbidden_hits,
            "required_information_overlap_pct": _semantic_overlap(
                answer, [str(value) for value in expected.get("required_information", [])]
            ),
            "source_quality_flags": source_quality_flags,
        },
    }


def _send(sender: Callable[[dict[str, Any]], dict[str, Any]], payload: dict[str, Any]) -> tuple[dict[str, Any] | None, str]:
    try:
        return sender(payload), ""
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def run(
    singles: list[dict[str, Any]],
    dialogues: list[dict[str, Any]],
    sender: Callable[[dict[str, Any]], dict[str, Any]],
    role: str = "guest",
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    trusted_token = _trusted_context_token(LOCAL_TRUSTED_CONTEXT_SECRET) if role == "authorized" else ""

    def payload(message: str, session_id: str) -> dict[str, Any]:
        result: dict[str, Any] = {
            "message": message,
            "session_id": session_id,
            "context": {"page_type": "public_site"},
        }
        if trusted_token:
            result["trusted_context_token"] = trusted_token
        return result

    for item in singles:
        response, error = _send(sender, payload(
            item["text"], f"stage5-development-{item['id']}",
        ))
        results.append(_evaluate_item(item, response, error))
    for dialogue in dialogues:
        session_id = f"stage5-development-{dialogue['id']}"
        for turn in dialogue["turns"]:
            response, error = _send(sender, payload(turn["text"], session_id))
            results.append(_evaluate_item(turn, response, error, str(dialogue["id"])))
    return results


def rescore(
    previous: dict[str, Any],
    source: dict[str, Any],
    inputs: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    previous_report = previous.get("report") or {}
    if (previous_report.get("source") or {}).get("sha256") != source["sha256"]:
        raise ValueError("Cached responses do not match the current source SHA-256")
    results = []
    for old in previous.get("results", []):
        current = inputs.get(str(old["id"])) or {}
        item = {
            "id": old["id"],
            "text": current.get("text") or old.get("text") or "",
            "expected": current.get("expected") or old.get("expected") or {},
        }
        results.append(
            _evaluate_item(
                item,
                old.get("response") or None,
                str(old.get("error") or ""),
                old.get("dialogue_id"),
            )
        )
    expected_count = source["single_count"] + source["dialogue_turn_count"]
    if len(results) != expected_count:
        raise ValueError(f"Cached response count {len(results)} != {expected_count}")
    return results


def _metric(results: list[dict[str, Any]], check: str) -> dict[str, Any]:
    eligible = [item for item in results if item["checks"].get(check) is not None]
    passed = sum(item["checks"].get(check) is True for item in eligible)
    return {
        "passed": passed,
        "total": len(eligible),
        "rate_pct": _rate(passed, len(eligible)),
        "wilson_95_lower_pct": _wilson_lower(passed, len(eligible)),
    }


def summarize(source: dict[str, Any], results: list[dict[str, Any]]) -> dict[str, Any]:
    by_expected_action: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_primary_topic: dict[str, list[dict[str, Any]]] = defaultdict(list)
    confusion: Counter[tuple[str, str]] = Counter()
    for item in results:
        expected = item["diagnostics"]["expected_action"]
        actual = item["diagnostics"]["actual_action"]
        by_expected_action[expected].append(item)
        by_primary_topic[str(item.get("expected", {}).get("primary_topic") or "<missing>")].append(item)
        confusion[(expected, actual)] += 1
    overlap_values = [
        float(item["diagnostics"]["required_information_overlap_pct"])
        for item in results
        if item["diagnostics"]["required_information_overlap_pct"] is not None
    ]
    action_failures = [item for item in results if item["checks"]["action_ok"] is False]
    confident_wrong = [
        item for item in action_failures
        if item["diagnostics"].get("confidence_level") == "high"
    ]
    quality_flags = Counter(
        flag
        for item in results
        for flag in item["diagnostics"].get("source_quality_flags", [])
    )
    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "development_diagnostic_not_blind_release_gate",
        "source": source,
        "population": {
            "approved_records": len(results),
            "single_records": source["single_count"],
            "dialogues": source["dialogue_count"],
            "dialogue_turns": source["dialogue_turn_count"],
        },
        "methodology": {
            "measured": [
                "bot action against the supplied review label",
                "dialogue support handoff against the supplied review label",
                "literal forbidden-information leakage",
            ],
            "diagnostic_only": "required-information lexical overlap is not a correctness gate",
            "not_measured": [
                "Top-1 scenario accuracy because labels contain no scenario_id",
                "independent 8/10 release quality because this set was used to improve the knowledge base",
            ],
        },
        "metrics": {
            "transport": _metric(results, "transport_ok"),
            "action_accuracy": _metric(results, "action_ok"),
            "forbidden_content": _metric(results, "forbidden_content_ok"),
            "dialogue_support_handoff": _metric(results, "support_handoff_ok"),
            "confident_wrong_action": {"count": len(confident_wrong)},
            "required_information_overlap_diagnostic": {
                "total": len(overlap_values),
                "mean_pct": round(sum(overlap_values) / len(overlap_values), 2) if overlap_values else 0.0,
            },
        },
        "action_by_expected": {
            action: _metric(items, "action_ok") for action, items in sorted(by_expected_action.items())
        },
        "action_by_primary_topic": {
            topic: _metric(items, "action_ok") for topic, items in sorted(by_primary_topic.items())
        },
        "action_confusion": [
            {"expected": expected, "actual": actual, "count": count}
            for (expected, actual), count in sorted(confusion.items())
        ],
        "failure_counts": {
            "action": len(action_failures),
            "forbidden_content": sum(not item["checks"]["forbidden_content_ok"] for item in results),
            "dialogue_support_handoff": sum(
                item["checks"]["support_handoff_ok"] is False for item in results
            ),
        },
        "source_quality": {
            "flags": dict(sorted(quality_flags.items())),
            "excluded_from_action_metric": sum(
                item["checks"]["action_ok"] is None for item in results
            ),
            "crm_case_note_style_is_reported_not_excluded": True,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate approved stage 5 development labels.")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--details", type=Path, default=DEFAULT_DETAILS)
    parser.add_argument(
        "--responses-from",
        type=Path,
        help="Recalculate metrics from an earlier complete response file without rerunning the bot.",
    )
    parser.add_argument("--role", choices=("guest", "authorized"), default="guest")
    args = parser.parse_args()
    source, singles, dialogues = load_approved(args.source)
    source["evaluation_role"] = args.role
    approved_items = [*singles, *(turn for dialogue in dialogues for turn in dialogue["turns"])]
    if not approved_items:
        raise ValueError(f"No approved records in source: {source}")
    item_ids = [str(item.get("id") or "") for item in approved_items]
    if any(not item_id for item_id in item_ids) or len(item_ids) != len(set(item_ids)):
        raise ValueError("Approved records must have non-empty unique ids")
    supported_actions = {"answer", "clarify", "support", "safe_refusal", "out_of_scope"}
    unknown_actions = sorted({
        str((item.get("expected") or {}).get("bot_action") or "")
        for item in approved_items
    } - supported_actions)
    if unknown_actions:
        raise ValueError(f"Unsupported expected bot actions: {unknown_actions}")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.details.parent.mkdir(parents=True, exist_ok=True)
    if args.responses_from:
        previous = json.loads(args.responses_from.read_text(encoding="utf-8"))
        inputs = {
            str(item["id"]): item
            for item in [*singles, *(turn for dialogue in dialogues for turn in dialogue["turns"])]
        }
        results = rescore(previous, source, inputs)
    else:
        if args.role == "authorized":
            os.environ["TRUSTED_CONTEXT_SECRET"] = LOCAL_TRUSTED_CONTEXT_SECRET
        with tempfile.TemporaryDirectory(prefix="stage5-development-", dir=args.details.parent) as temp_dir:
            sender = build_local_sender(Path(temp_dir) / "runtime.sqlite3")
            results = run(singles, dialogues, sender, role=args.role)
    report = summarize(source, results)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.details.write_text(json.dumps({"report": report, "results": results}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
