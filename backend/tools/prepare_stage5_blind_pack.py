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


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in eligible:
        grouped[str(row.get("conversation_id") or "")].append(row)
    dialogue_groups = [
        sorted(group, key=lambda item: (str(item.get("created_at") or ""), str(item.get("source_message_id") or "")))
        for conversation_id, group in grouped.items()
        if conversation_id and len(group) >= 2
    ]
    dialogue_groups.sort(key=lambda group: _stable_rank(str(group[0].get("conversation_id") or "")))
    selected_dialogues = dialogue_groups[:50]
    dialogue_message_ids = {
        (str(row.get("source") or ""), str(row.get("source_message_id") or ""))
        for group in selected_dialogues for row in group
    }
    singles = [
        row for row in eligible
        if (str(row.get("source") or ""), str(row.get("source_message_id") or "")) not in dialogue_message_ids
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
            "turns": [
                {
                    "turn": turn,
                    "text": str(row["text_redacted"]),
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
        "dataset_version": "stage5-blind-review-2026.08.24.1",
        "publication_allowed": False,
        "selection": "deterministic SHA-256 ranking; no phrase-based manual selection",
        "source_sha256": _sha(source),
        "reviewer_attestation": {
            "reviewer_id": "",
            "router_contributor": None,
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
        },
        "review_pack_sha256": hashlib.sha256(
            json.dumps(pack, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "readiness": readiness,
    }
    return pack, report


def validate_reviewed_pack(pack: dict[str, Any]) -> dict[str, Any]:
    cases = list(pack.get("cases", []))
    dialogues = list(pack.get("dialogues", []))
    turns = [turn for dialogue in dialogues for turn in dialogue.get("turns", [])]
    records = [*cases, *turns]
    attestation = pack.get("reviewer_attestation", {})
    independent_reviewer = bool(attestation.get("reviewer_id")) and attestation.get("router_contributor") is False
    labels_complete = bool(records) and all(
        row.get("review", {}).get("status") == "approved"
        and bool(row.get("review", {}).get("reviewer_id"))
        and row.get("review", {}).get("router_contributor") is False
        and bool(row.get("expected"))
        for row in records
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
        "pending_review_records": sum(row.get("review", {}).get("status") != "approved" for row in records),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--pack", type=Path, default=DEFAULT_PACK)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    pack, report = prepare(args.source)
    args.pack.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.pack.write_text(json.dumps(pack, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["readiness"]["freeze_ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
