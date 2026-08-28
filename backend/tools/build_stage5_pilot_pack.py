from __future__ import annotations

import argparse
import collections
import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from backend.tools.build_stage5_labeling_form import build as build_labeling_form
from backend.tools.prepare_stage5_blind_pack import _valid_expected


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DRAFT = ROOT / ".work/stage5-blind-review-draft-2026-08-28.json"
DEFAULT_SOURCE = ROOT / ".work/support-normalized.jsonl"
DEFAULT_PACK = ROOT / ".work/stage5-blind-pilot-review-pack.json"
DEFAULT_FORM = ROOT / ".work/stage5-blind-pilot-labeling-form.html"
DEFAULT_REPORT = ROOT / "reports/stage5-pilot-preparation.json"
EXCLUDED_KINDS = {"system", "contact_only", "acknowledgement"}
PILOT_NEW_SINGLE_COUNT = 100


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def _source_ref(row: dict[str, Any]) -> str:
    return hashlib.sha256(
        f"{row.get('source')}#{row.get('source_message_id')}".encode("utf-8")
    ).hexdigest()


def _ordered(group: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        group,
        key=lambda item: (str(item.get("created_at") or ""), str(item.get("source_message_id") or "")),
    )


def _buyer_source_refs(rows: list[dict[str, Any]]) -> set[str]:
    grouped: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        if conversation_id := str(row.get("conversation_id") or ""):
            grouped[conversation_id].append(row)

    buyer_refs: set[str] = set()
    for group in grouped.values():
        ordered = _ordered(group)
        initiator = str(ordered[0].get("speaker_key") or "")
        if not initiator:
            continue
        for row in ordered:
            if str(row.get("speaker_key") or "") != initiator:
                continue
            if str(row.get("message_kind") or "") in EXCLUDED_KINDS:
                continue
            buyer_refs.add(_source_ref(row))
    return buyer_refs


def build_pilot(draft: dict[str, Any], rows: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    buyer_refs = _buyer_source_refs(rows)
    clean_cases = [case for case in draft.get("cases", []) if case.get("source_ref_sha256") in buyer_refs]
    approved = [case for case in clean_cases if case.get("review", {}).get("status") == "approved"]
    expert = [case for case in clean_cases if case.get("review", {}).get("status") == "needs_review"]
    pending = [case for case in clean_cases if case.get("review", {}).get("status") == "pending"]
    selected_pending = pending[:PILOT_NEW_SINGLE_COUNT]
    if len(selected_pending) != PILOT_NEW_SINGLE_COUNT:
        raise ValueError(
            f"Not enough clean pending buyer cases: {len(selected_pending)} < {PILOT_NEW_SINGLE_COUNT}"
        )

    selected_cases = deepcopy([*approved, *expert, *selected_pending])
    dialogues = deepcopy(draft.get("dialogues", []))
    reviewer_id = str(draft.get("reviewer_attestation", {}).get("reviewer_id") or "").strip()
    promoted_dialogue_turns = 0
    dialogue_expert_ids: list[str] = []
    dialogue_ids: list[str] = []
    for dialogue in dialogues:
        for turn in dialogue.get("turns", []):
            status = turn.get("review", {}).get("status")
            if status == "needs_review":
                dialogue_expert_ids.append(str(turn["id"]))
                continue
            if not _valid_expected(turn.get("expected"), dialogue_turn=True):
                raise ValueError(f"Dialogue turn has incomplete labels: {turn.get('id')}")
            turn["review"] = {
                "status": "approved",
                "reviewer_id": reviewer_id,
                "router_contributor": False,
            }
            dialogue_ids.append(str(turn["id"]))
            promoted_dialogue_turns += 1

    single_expert_ids = [str(case["id"]) for case in selected_cases if case.get("review", {}).get("status") == "needs_review"]
    pending_single_ids = [str(case["id"]) for case in selected_cases if case.get("review", {}).get("status") == "pending"]
    approved_single_ids = [str(case["id"]) for case in selected_cases if case.get("review", {}).get("status") == "approved"]
    review_order = [
        *single_expert_ids,
        *dialogue_expert_ids,
        *dialogue_ids,
        *pending_single_ids,
        *approved_single_ids,
    ]

    original_status_counts = collections.Counter(
        case.get("review", {}).get("status", "missing") for case in draft.get("cases", [])
    )
    clean_status_counts = collections.Counter(
        case.get("review", {}).get("status", "missing") for case in clean_cases
    )
    excluded_status_counts = {
        status: original_status_counts[status] - clean_status_counts[status]
        for status in sorted(original_status_counts)
    }
    attestation = deepcopy(draft.get("reviewer_attestation", {}))
    attestation["review_completed_at"] = ""
    pack = {
        "schema_version": 1,
        "dataset_version": "stage5-blind-pilot-2026.08.28.1",
        "parent_dataset_version": draft.get("dataset_version"),
        "publication_allowed": False,
        "selection": (
            "user-approved reduced blind pilot; buyer-initiator singles only; "
            "81 clean approved + 2 clean expert + 100 new clean singles; all 50 dialogues"
        ),
        "source_sha256": draft.get("source_sha256"),
        "pilot_does_not_replace_release_gate": True,
        "draft_export_filename": "stage5-blind-pilot-review-draft.json",
        "final_export_filename": "stage5-blind-pilot-reviewed-pack.json",
        "reviewer_attestation": attestation,
        "review_order": review_order,
        "cases": selected_cases,
        "dialogues": dialogues,
    }
    all_turns = [turn for dialogue in dialogues for turn in dialogue.get("turns", [])]
    all_records = [*selected_cases, *all_turns]
    status_counts = collections.Counter(
        record.get("review", {}).get("status", "missing") for record in all_records
    )
    report = {
        "schema_version": 1,
        "dataset_version": pack["dataset_version"],
        "source_sha256": pack["source_sha256"],
        "selection": {
            "single_count": len(selected_cases),
            "approved_clean_singles_carried_forward": len(approved),
            "clean_single_expert_cases": len(expert),
            "new_clean_pending_singles": len(selected_pending),
            "dialogue_count": len(dialogues),
            "dialogue_turn_count": len(all_turns),
            "promoted_dialogue_turns": promoted_dialogue_turns,
            "dialogue_expert_cases": len(dialogue_expert_ids),
            "review_record_count": len(all_records),
        },
        "excluded_non_buyer_singles_by_original_status": excluded_status_counts,
        "remaining_statuses": dict(sorted(status_counts.items())),
        "expert_first_count": len(single_expert_ids) + len(dialogue_expert_ids),
        "review_order_complete": len(review_order) == len(all_records) == len(set(review_order)),
        "pilot_pack_sha256": hashlib.sha256(_json_bytes(pack)).hexdigest(),
    }
    return pack, report


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the user-approved reduced stage 5 blind pilot.")
    parser.add_argument("--draft", type=Path, default=DEFAULT_DRAFT)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--pack", type=Path, default=DEFAULT_PACK)
    parser.add_argument("--form", type=Path, default=DEFAULT_FORM)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    draft = json.loads(args.draft.read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in args.source.read_text(encoding="utf-8").splitlines() if line.strip()]
    pack, report = build_pilot(draft, rows)
    form = build_labeling_form(pack)
    report["pilot_form_sha256"] = hashlib.sha256(form.encode("utf-8")).hexdigest()
    args.pack.parent.mkdir(parents=True, exist_ok=True)
    args.form.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.pack.write_bytes(_json_bytes(pack))
    args.form.write_bytes(form.encode("utf-8"))
    args.report.write_bytes(_json_bytes(report))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
