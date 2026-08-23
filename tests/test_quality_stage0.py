from __future__ import annotations

from pathlib import Path

from backend.tools.analyze_quality_failures import build as build_failure_taxonomy
from backend.tools.build_label_adjudication_queue import build as build_adjudication_queue


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
