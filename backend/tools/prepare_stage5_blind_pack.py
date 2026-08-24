from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from backend.app.bot.text_processing import normalize_matching_text


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = ROOT / ".work/support-normalized.jsonl"
DEFAULT_PACK = ROOT / ".work/stage5-blind-review-pack.json"
DEFAULT_FORM = ROOT / ".work/stage5-independent-labeling-form.html"
DEFAULT_REPORT = ROOT / "reports/stage5-blind-preparation.json"
KNOWN_CORPORA = (
    ROOT / "tests/data/scenario_gold.jsonl",
    ROOT / "tests/data/live_query_audit_2026_08_13.json",
    ROOT / "tests/data/routing_v3_independent_acceptance.json",
    ROOT / "tests/data/routing_v3_closed_control_270.json",
)
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
PHONE_RE = re.compile(r"(?:\+7|8)[\s\-()]*\d{3}[\s\-()]*\d{3}[\s\-]*\d{2}[\s\-]*\d{2}")
LONG_ID_RE = re.compile(r"\b\d{7,}\b")
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
DIALOGUE_EXCLUDED_KINDS = {"system", "contact_only", "acknowledgement"}
BOT_ACTIONS = {"answer", "clarify", "support", "out_of_scope", "safe_refusal"}
CONFIDENCE_LEVELS = {"high", "medium"}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def _known_texts() -> set[str]:
    texts: set[str] = set()
    for path in KNOWN_CORPORA:
        if path.suffix == ".jsonl":
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        else:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if "cases" in payload:
                rows = payload["cases"]
            else:
                rows = [query for group in payload.get("groups", []) for query in group.get("queries", [])]
        for row in rows:
            text = next((row.get(key) for key in ("text", "message", "query", "q") if row.get(key)), "")
            if text:
                texts.add(normalize_matching_text(str(text)))
    return texts


def _privacy_hits(text: str) -> list[str]:
    hits = []
    for name, pattern in (("email", EMAIL_RE), ("phone", PHONE_RE), ("long_id", LONG_ID_RE), ("url", URL_RE)):
        if pattern.search(text):
            hits.append(name)
    return hits


def _stable_rank(value: str) -> str:
    return hashlib.sha256(f"stage5-blind-2026-08-24\0{value}".encode("utf-8")).hexdigest()


def _source_key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row.get("source") or ""), str(row.get("source_message_id") or "")


def _ordered(group: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        group,
        key=lambda item: (str(item.get("created_at") or ""), str(item.get("source_message_id") or "")),
    )


def _select_dialogue_groups(
    rows: list[dict[str, Any]],
    known: set[str],
) -> tuple[list[list[dict[str, Any]]], dict[str, int]]:
    """Select real multi-turn messages from the conversation initiator only."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        conversation_id = str(row.get("conversation_id") or "")
        if conversation_id:
            grouped[conversation_id].append(row)

    eligible: list[tuple[str, list[dict[str, Any]]]] = []
    privacy_rejections = 0
    known_rejections = 0
    for conversation_id, raw_group in grouped.items():
        ordered = _ordered(raw_group)
        initiator = str(ordered[0].get("speaker_key") or "")
        if not initiator:
            continue
        turns: list[dict[str, Any]] = []
        rejected = False
        for row in ordered:
            if str(row.get("speaker_key") or "") != initiator:
                continue
            if str(row.get("message_kind") or "") in DIALOGUE_EXCLUDED_KINDS:
                continue
            text = str(row.get("text_redacted") or "").strip()
            normalized = normalize_matching_text(text)
            if len(normalized.split()) < 2:
                continue
            if _privacy_hits(text):
                privacy_rejections += 1
                rejected = True
                break
            if normalized in known:
                known_rejections += 1
                rejected = True
                break
            turns.append(row)
        if not rejected and len(turns) >= 2:
            eligible.append((conversation_id, turns))

    eligible.sort(key=lambda item: _stable_rank(item[0]))
    unique: list[list[dict[str, Any]]] = []
    seen_signatures: set[tuple[str, ...]] = set()
    for _, turns in eligible:
        signature = tuple(normalize_matching_text(str(row["text_redacted"])) for row in turns)
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        unique.append(turns)
    return unique, {
        "eligible_user_dialogues": len(unique),
        "dialogue_privacy_rejections": privacy_rejections,
        "dialogue_exact_known_corpus_rejections": known_rejections,
    }


def _non_empty_strings(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(item, str) and item.strip() for item in value)


def _valid_expected(expected: Any, *, dialogue_turn: bool) -> bool:
    if not isinstance(expected, dict):
        return False
    if not all(str(expected.get(field) or "").strip() for field in ("primary_topic", "specific_situation")):
        return False
    if expected.get("bot_action") not in BOT_ACTIONS:
        return False
    if expected.get("confidence") not in CONFIDENCE_LEVELS:
        return False
    for field, none_field in (
        ("required_information", "no_required_information"),
        ("forbidden_information", "no_forbidden_information"),
    ):
        values = expected.get(field)
        explicit_none = expected.get(none_field) is True
        if not explicit_none and not (_non_empty_strings(values) and bool(values)):
            return False
        if explicit_none and values not in ([], None):
            return False
    if not isinstance(expected.get("multiple_valid_answers"), bool):
        return False
    alternatives = expected.get("acceptable_alternatives")
    if expected["multiple_valid_answers"] and not (_non_empty_strings(alternatives) and bool(alternatives)):
        return False
    if not expected["multiple_valid_answers"] and alternatives not in ([], None):
        return False
    if dialogue_turn:
        for field in ("continues_previous_topic", "resolved_after_turn", "support_handoff"):
            if not isinstance(expected.get(field), bool):
                return False
        if not isinstance(expected.get("known_context"), list) or not _non_empty_strings(
            expected.get("known_context")
        ):
            return False
    return True


def prepare(source: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    rows = [
        json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    candidates = [row for row in rows if row.get("message_kind") == "candidate"]
    known = _known_texts()
    eligible = []
    privacy_rejections = 0
    novelty_rejections = 0
    seen: set[str] = set()
    for row in candidates:
        text = str(row.get("text_redacted") or "").strip()
        normalized = normalize_matching_text(text)
        if len(normalized.split()) < 2 or normalized in seen:
            continue
        seen.add(normalized)
        if _privacy_hits(text):
            privacy_rejections += 1
            continue
        if normalized in known:
            novelty_rejections += 1
            continue
        eligible.append(row)

    dialogue_groups, dialogue_metrics = _select_dialogue_groups(rows, known)
    selected_dialogues = dialogue_groups[:50]
    dialogue_message_ids = {
        _source_key(row)
        for group in selected_dialogues for row in group
    }
    singles = [
        row for row in eligible
        if _source_key(row) not in dialogue_message_ids
    ]
    singles.sort(key=lambda row: _stable_rank(normalize_matching_text(str(row["text_redacted"]))))
    selected_singles = singles[:500]

    cases = [
        {
            "id": f"blind-single-{index:04d}",
            "text": str(row["text_redacted"]),
            "source_ref_sha256": hashlib.sha256(
                f"{row.get('source')}#{row.get('source_message_id')}".encode("utf-8")
            ).hexdigest(),
            "expected": {},
            "review": {"status": "pending", "reviewer_id": "", "router_contributor": None},
        }
        for index, row in enumerate(selected_singles, start=1)
    ]
    dialogues = [
        {
            "id": f"blind-dialogue-{index:03d}",
            "conversation_ref_sha256": hashlib.sha256(
                f"{group[0].get('source')}#{group[0].get('conversation_id')}".encode("utf-8")
            ).hexdigest(),
            "turns": [
                {
                    "id": f"blind-dialogue-{index:03d}-turn-{turn:02d}",
                    "turn": turn,
                    "text": str(row["text_redacted"]),
                    "source_ref_sha256": hashlib.sha256(
                        f"{row.get('source')}#{row.get('source_message_id')}".encode("utf-8")
                    ).hexdigest(),
                    "expected": {},
                    "review": {"status": "pending", "reviewer_id": "", "router_contributor": None},
                }
                for turn, row in enumerate(group, start=1)
            ],
        }
        for index, group in enumerate(selected_dialogues, start=1)
    ]
    pack = {
        "schema_version": 1,
        "dataset_version": "stage5-blind-review-2026.08.24.2",
        "publication_allowed": False,
        "selection": "deterministic SHA-256 ranking; user-only real dialogues; no phrase-based manual selection",
        "source_sha256": _sha(source),
        "reviewer_attestation": {
            "reviewer_id": "",
            "router_contributor": None,
            "confidentiality_confirmed": False,
            "personal_data_absent_confirmed": False,
            "review_completed_at": "",
        },
        "cases": cases,
        "dialogues": dialogues,
    }
    readiness = validate_reviewed_pack(pack)
    report = {
        "schema_version": 1,
        "source": {
            "sha256": _sha(source),
            "candidate_messages": len(candidates),
            "eligible_unique_messages": len(eligible),
        },
        "selection": {
            "single_turn_count": len(cases),
            "dialogue_count": len(dialogues),
            "dialogue_turn_count": sum(len(item["turns"]) for item in dialogues),
            "privacy_rejections": privacy_rejections,
            "exact_known_corpus_rejections": novelty_rejections,
            **dialogue_metrics,
        },
        "review_pack_sha256": hashlib.sha256(_json_bytes(pack)).hexdigest(),
        "readiness": readiness,
    }
    return pack, report


def validate_reviewed_pack(pack: dict[str, Any]) -> dict[str, Any]:
    cases = list(pack.get("cases", []))
    dialogues = list(pack.get("dialogues", []))
    turns = [turn for dialogue in dialogues for turn in dialogue.get("turns", [])]
    records = [(row, False) for row in cases] + [(row, True) for row in turns]
    attestation = pack.get("reviewer_attestation", {})
    independent_reviewer = (
        bool(attestation.get("reviewer_id"))
        and attestation.get("router_contributor") is False
        and attestation.get("confidentiality_confirmed") is True
        and attestation.get("personal_data_absent_confirmed") is True
        and bool(attestation.get("review_completed_at"))
    )
    labels_complete = bool(records) and all(
        row.get("review", {}).get("status") == "approved"
        and bool(row.get("review", {}).get("reviewer_id"))
        and row.get("review", {}).get("router_contributor") is False
        and _valid_expected(row.get("expected"), dialogue_turn=dialogue_turn)
        for row, dialogue_turn in records
    )
    checks = {
        "single_turn_count_gte_500": len(cases) >= 500,
        "dialogue_count_gte_50": len(dialogues) >= 50,
        "independent_reviewer_attested": independent_reviewer,
        "labels_complete": labels_complete,
    }
    return {
        "freeze_ready": all(checks.values()),
        "checks": checks,
        "missing_dialogues": max(0, 50 - len(dialogues)),
        "pending_review_records": sum(
            row.get("review", {}).get("status") != "approved" for row, _ in records
        ),
        "invalid_label_records": sum(
            not _valid_expected(row.get("expected"), dialogue_turn=dialogue_turn)
            for row, dialogue_turn in records
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--pack", type=Path, default=DEFAULT_PACK)
    parser.add_argument("--form", type=Path, default=DEFAULT_FORM)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    pack, report = prepare(args.source)
    from backend.tools.build_stage5_labeling_form import build as build_labeling_form

    form = build_labeling_form(pack)
    try:
        form_path = str(args.form.resolve().relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        form_path = str(args.form)
    report["review_form"] = {
        "path": form_path,
        "sha256": hashlib.sha256(form.encode("utf-8")).hexdigest(),
    }
    args.pack.parent.mkdir(parents=True, exist_ok=True)
    args.form.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.pack.write_bytes(_json_bytes(pack))
    args.form.write_bytes(form.encode("utf-8"))
    args.report.write_bytes(_json_bytes(report))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["readiness"]["freeze_ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
