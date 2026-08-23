from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any


DEFAULT_QUEUE = Path("reports/routing-label-adjudication-110.json")
DEFAULT_SCENARIOS = Path("knowledge/v2/scenarios.json")
DEFAULT_DATASET = Path("tests/data/routing_v3_closed_control_270.json")
DEFAULT_OVERLAY = Path("tests/data/routing_label_adjudication_110.json")
REVIEWER = "OpenAI Codex — evidence-based review against approved MIGTORG KB"

# Only genuine label corrections are listed here. Every other label is explicitly
# confirmed after review against the scenario facts and sources.
CORRECTIONS: dict[str, list[str | None]] = {
    "widget-001": [None],
    "widget-008": ["buyer.get_started"],
    "widget-009": [None],
    "widget-022": ["bid.place"],
    "widget-025": ["bid.place", "technical.site_error"],
    "widget-028": ["lot.status_guide"],
    "widget-030": ["auction.result"],
    "widget-043": ["lot.card_information"],
    "widget-045": ["auction.completed_analytics"],
    "widget-047": ["lot.catalog_search"],
    "widget-056": ["payment.methods", "account.identification_edge_case"],
    "widget-068": ["transfer.not_confirmed"],
    "widget-072": ["contract.receive", "documents.preparation_delay"],
    "widget-095": ["pickup.access_issuer"],
    "widget-102": ["feedback.platform_complaint", "technical.site_error"],
}

AMBIGUOUS_CLARIFICATION = {"widget-001", "widget-009"}
RESOLUTION_OVERRIDES = {
    "widget-001": ["clarified"],
    "widget-009": ["clarified"],
    "widget-106": ["out_of_scope"],
    "widget-107": ["out_of_scope"],
    "widget-108": ["out_of_scope"],
    "widget-109": ["out_of_scope"],
    "widget-110": ["out_of_scope"],
}

KB_GAPS = {
    "widget-043": "БЗ подтверждает работу с фотографиями карточки, но не подтверждает наличие пакетного скачивания архивом.",
    "widget-053": "В БЗ нет единого правила о «сгорании» всех типов тарифов; требуется проверка конкретного тарифа и операции.",
    "widget-057": "Актуальная цена проверяется перед оплатой; единый срок для всех типов доступа в БЗ не определён.",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def adjudicate(
    queue: dict[str, Any],
    scenarios_payload: dict[str, Any],
    *,
    dataset_sha256: str,
    reviewed_at: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    reviewed_at = reviewed_at or datetime.now(timezone.utc).isoformat()
    scenarios = {item["scenario_id"]: item for item in scenarios_payload["records"]}
    corrected = 0
    gap_count = 0
    overlay_records: list[dict[str, Any]] = []

    for record in queue["records"]:
        case_id = record["case_id"]
        original = list(record["current_expected_scenario_ids"])
        selected = list(CORRECTIONS.get(case_id, original))
        if selected != original:
            corrected += 1
        invalid = [item for item in selected if item is not None and item not in scenarios]
        if invalid:
            raise ValueError(f"Unknown scenario IDs for {case_id}: {invalid}")

        applicable = [scenarios[item] for item in selected if item is not None]
        intents = sorted({str(item["intent"]) for item in applicable})
        kb_gap = case_id in KB_GAPS
        gap_count += int(kb_gap)
        allowed_resolutions = RESOLUTION_OVERRIDES.get(case_id)
        if case_id in AMBIGUOUS_CLARIFICATION:
            note = "Исправлено как объективно неоднозначный запрос: роль и цель пользователя не определены, поэтому требуется предметное уточнение без выбора сценария."
        elif selected == [None]:
            note = "Подтверждено как out-of-scope/safety: ни один сценарий БЗ не должен раскрывать или выдумывать запрошенные сведения."
        else:
            evidence = "; ".join(
                f"{item['scenario_id']} — {item['title']} (источник: {item['source']})"
                for item in applicable
            )
            action = "Исправлено" if selected != original else "Подтверждено"
            note = f"{action} по смыслу вопроса и approved facts: {evidence}."
        if kb_gap:
            note += f" KB gap: {KB_GAPS[case_id]} Ответ не должен утверждать отсутствующий факт."

        record.update(
            {
                "review_status": "approved",
                "reviewer": REVIEWER,
                "reviewed_at": reviewed_at,
                "adjudicated_scenario_ids": selected,
                "adjudicated_expected_intents": intents,
                "adjudicated_allowed_resolutions": allowed_resolutions,
                "label_changed": selected != original,
                "kb_gap": kb_gap,
                "adjudication_note": note,
            }
        )
        overlay_records.append(
            {
                "case_id": case_id,
                "expected_scenario_ids": selected,
                "expected_intents": intents or None,
                "allowed_resolutions": allowed_resolutions,
                "review_status": "approved",
                "kb_gap": kb_gap,
            }
        )

    if len(overlay_records) != 110:
        raise ValueError(f"Expected 110 adjudicated records, got {len(overlay_records)}")
    queue.update(
        {
            "adjudicated_at": reviewed_at,
            "pending_count": 0,
            "approved_count": len(overlay_records),
            "corrected_count": corrected,
            "confirmed_count": len(overlay_records) - corrected,
            "kb_gap_count": gap_count,
        }
    )
    overlay = {
        "schema_version": 1,
        "dataset_id": "routing-v3-closed-control-270",
        "base_dataset_sha256": dataset_sha256,
        "reviewer": REVIEWER,
        "reviewed_at": reviewed_at,
        "record_count": len(overlay_records),
        "corrected_count": corrected,
        "kb_gap_count": gap_count,
        "records": overlay_records,
    }
    return queue, overlay


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Approve and export expert-reviewed routing labels.")
    parser.add_argument("--queue", type=Path, default=DEFAULT_QUEUE)
    parser.add_argument("--scenarios", type=Path, default=DEFAULT_SCENARIOS)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--overlay", type=Path, default=DEFAULT_OVERLAY)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    queue = json.loads(args.queue.read_text(encoding="utf-8"))
    scenarios = json.loads(args.scenarios.read_text(encoding="utf-8"))
    approved, overlay = adjudicate(queue, scenarios, dataset_sha256=_sha256(args.dataset))
    args.queue.write_text(json.dumps(approved, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.overlay.parent.mkdir(parents=True, exist_ok=True)
    args.overlay.write_text(json.dumps(overlay, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "queue": str(args.queue),
        "overlay": str(args.overlay),
        "approved": approved["approved_count"],
        "corrected": approved["corrected_count"],
        "kb_gaps": approved["kb_gap_count"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
