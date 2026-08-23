from __future__ import annotations

import hashlib
import json
from pathlib import Path

from backend.tools.adjudicate_routing_labels import adjudicate
from backend.tools.analyze_quality_failures import build as build_failure_taxonomy
from backend.tools.build_label_adjudication_queue import build as build_adjudication_queue
from backend.tools.evaluate_live_queries import load_dataset


ROOT = Path(__file__).resolve().parents[1]


def test_every_closed_control_failure_has_one_primary_cause() -> None:
    payload = build_failure_taxonomy(ROOT / "reports/routing-v3-closed-control-270.json")
    assert payload["summary"]["failed_case_count"] == 55
    assert payload["summary"]["classified_case_count"] == 55
    assert all(item["primary_cause"] in payload["taxonomy"] for item in payload["cases"])


def test_all_retrospective_widget_labels_enter_adjudication_queue() -> None:
    payload = build_adjudication_queue(
        ROOT / "tests/data/routing_v3_closed_control_270.json",
        ROOT / "reports/routing-v3-closed-control-270.json",
    )
    assert payload["record_count"] == payload["pending_count"] == 110
    assert {item["review_status"] for item in payload["records"]} == {"pending_domain_expert"}


def test_domain_adjudication_approves_all_labels_and_exports_overlay(tmp_path: Path) -> None:
    dataset_path = ROOT / "tests/data/routing_v3_closed_control_270.json"
    queue = build_adjudication_queue(
        dataset_path,
        ROOT / "reports/routing-v3-closed-control-270.json",
    )
    scenarios = json.loads((ROOT / "knowledge/v2/scenarios.json").read_text(encoding="utf-8"))
    approved, overlay = adjudicate(
        queue,
        scenarios,
        dataset_sha256=hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
        reviewed_at="2026-08-23T12:00:00+03:00",
    )

    assert approved["approved_count"] == overlay["record_count"] == 110
    assert approved["pending_count"] == 0
    assert approved["corrected_count"] == 15
    assert approved["kb_gap_count"] == 3
    assert {item["review_status"] for item in approved["records"]} == {"approved"}

    overlay_path = tmp_path / "adjudication.json"
    overlay_path.write_text(json.dumps(overlay, ensure_ascii=False), encoding="utf-8")
    dataset, cases = load_dataset(dataset_path, overlay_path)
    by_id = {item["id"]: item for item in cases}
    assert dataset["adjudication"]["record_count"] == 110
    assert by_id["widget-001"]["expected"]["expected_scenario_ids"] == ["platform.about"]
    assert by_id["widget-068"]["expected"]["expected_scenario_ids"] == ["transfer.not_confirmed"]
    assert by_id["widget-110"]["expected"]["expected_scenario_ids"] == [None]
