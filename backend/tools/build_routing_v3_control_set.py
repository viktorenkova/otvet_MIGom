from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from backend.app.bot.scenario_engine import load_scenarios


DEFAULT_WIDGET_AUDIT = Path("tests/data/live_widget_audit_110_2026_08_13.json")
DEFAULT_INDEPENDENT_AUDIT = Path("tests/data/live_query_audit_2026_08_13.json")
DEFAULT_OUTPUT = Path("tests/data/routing_v3_closed_control_270.json")


# Expert labels for the 110 live widget questions. Alternatives are intentional:
# they describe equally acceptable routes, not a preferred current implementation.
WIDGET_SCENARIOS: tuple[tuple[str | None, ...], ...] = (
    ("buyer.get_started",), ("buyer.get_started",), ("buyer.get_started",),
    ("buyer.get_started",), ("buyer.get_started",), ("lot.catalog_search", "tariff.demo"),
    ("buyer.get_started",), ("auction.formats", "buyer.get_started"), ("buyer.get_started",),
    ("buyer.first_bid_checklist",), ("account.registration",), ("account.registration",),
    ("account.activation_pending", "notification.delivery_problem"), ("account.login_problem",),
    ("account.login_problem",), ("seller.get_started",), ("seller.get_started",),
    ("account.registration",), ("account.blocked",), ("account.credential_responsibility",),
    ("bid.place",), ("bid.not_visible", "bid.place"), ("bid.autobid_extension",),
    ("bid.price_terms",), ("bid.place",), ("bid.position_service",), ("bid.modify_cancel",),
    ("auction.status",), ("bid.not_visible",), ("auction.result", "auction.status"),
    ("bid.autobid_extension",), ("auction.formats",), ("auction.result",),
    ("auction.status",), ("auction.result",), ("lot.catalog_search",),
    ("technical.catalog_search_filter",), ("lot.catalog_search",), ("lot.catalog_search",),
    ("technical.catalog_search_filter",), ("technical.lot_image_missing",),
    ("technical.lot_image_missing",), ("technical.lot_image_missing", "lot.card_information"),
    ("technical.lot_image_missing",), ("lot.catalog_search",),
    ("technical.catalog_search_filter",), ("technical.catalog_search_filter", "lot.catalog_search"),
    ("lot.catalog_search",), ("lot.catalog_search",),
    ("technical.catalog_search_filter", "lot.card_information"), ("tariff.choose",),
    ("tariff.status",), ("tariff.status",), ("tariff.status",), ("commission.explained",),
    ("payment.methods",), ("tariff.choose",), ("payment.methods", "balance.topup.commission"),
    ("payment.not_visible",), ("payment.not_visible",), ("payment.accounting_documents",),
    ("refund.eligibility",), ("commission.explained",), ("lot.payment.details", "payment.methods"),
    ("payment.accounting_documents",), ("win.next_steps",), ("pickup.receive_lot",),
    ("transfer.seller_no_response",), ("lot.location", "pickup.receive_lot"),
    ("pickup.representative",), ("contract.receive",), ("contract.receive",),
    ("contract.receive",), ("transfer.confirmed",), ("pickup.access_issuer",),
    ("transfer.notification_contact",), ("pickup.access_issuer", "lot.location"),
    ("win.next_steps", "transfer.seller_no_response"), ("pickup.receive_lot",),
    ("pickup.receive_lot",), ("refund.application",), ("refund.application",),
    ("refund.timing_status",), ("refund.application",), ("penalty.explain_or_dispute",),
    ("refusal.change_mind",), ("penalty.explain_or_dispute",), ("lot.payment.overdue",),
    ("penalty.explain_or_dispute",), ("refund.timing_status",),
    ("support.office_visit",), ("support.office_visit",), ("support.office_visit",),
    ("support.office_visit",), ("lot.location",), ("support.office_visit",),
    ("support.callback",), ("support.contact",),
    ("feedback.platform_complaint", "technical.catalog_search_filter"),
    ("feedback.improvement_suggestion",),
    ("feedback.platform_complaint", "technical.lot_image_missing"),
    ("feedback.platform_complaint", "bid.place"),
    ("feedback.platform_complaint", "technical.catalog_search_filter"),
    ("feedback.improvement_suggestion",), ("feedback.bot_answer_complaint",),
    (None,), (None,), (None,), (None,), (None,),
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _merge_expectations(group: dict[str, Any], query: dict[str, Any]) -> dict[str, Any]:
    keys = {
        "expected_scenario_ids", "expected_intents", "allowed_resolutions",
        "required_any_groups", "forbidden_answer_fragments", "expect_direct",
    }
    expected = {key: group[key] for key in keys if key in group}
    expected.update({key: query[key] for key in keys if key in query})
    return expected


def build(widget_path: Path, independent_path: Path) -> dict[str, Any]:
    widget = json.loads(widget_path.read_text(encoding="utf-8"))
    independent = json.loads(independent_path.read_text(encoding="utf-8"))
    if len(widget["results"]) != 110 or len(WIDGET_SCENARIOS) != 110:
        raise ValueError("The live widget source and expert label table must contain exactly 110 cases.")

    scenarios = {item.scenario_id: item for item in load_scenarios()}
    cases: list[dict[str, Any]] = []
    for result, expected_scenarios in zip(widget["results"], WIDGET_SCENARIOS, strict=True):
        unknown = [item for item in expected_scenarios if item is not None and item not in scenarios]
        if unknown:
            raise ValueError(f"Unknown scenario labels for widget case {result['id']}: {unknown}")
        intents = sorted({scenarios[item].intent for item in expected_scenarios if item is not None})
        expected: dict[str, Any] = {
            "expected_scenario_ids": list(expected_scenarios),
            "allowed_resolutions": ["answered", "status", "clarified", "escalated", "out_of_scope"],
        }
        if intents:
            expected["expected_intents"] = intents
        cases.append({
            "id": f"widget-{int(result['id']):03d}",
            "source": "live_widget_audit_110",
            "group": str(result.get("topic") or "widget"),
            "class": "natural",
            "text": result["q"],
            "persona": result.get("persona"),
            "expected": expected,
        })

    independent_count = 0
    for group in independent["groups"]:
        for query in group["queries"]:
            cases.append({
                "id": f"independent-{query['id']}",
                "source": "independent_live_query_audit_160",
                "group": group["name"],
                "class": query["class"],
                "text": query["text"],
                "expected": _merge_expectations(group, query),
            })
            independent_count += 1
    if independent_count != 160 or len(cases) != 270:
        raise ValueError(f"Closed control set must contain 110 + 160 = 270 cases, got {len(cases)}.")

    canonical = json.dumps(cases, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "schema_version": 1,
        "id": "routing-v3-closed-control-270",
        "locked_at": "2026-08-13",
        "policy": "Closed regression corpus. Do not add, delete, relabel or use its phrases as routing rules without a reviewed corpus version change.",
        "sources": [
            {"name": "live_widget_audit_110", "count": 110, "sha256": _sha256(widget_path)},
            {"name": "independent_live_query_audit_160", "count": 160, "sha256": _sha256(independent_path)},
        ],
        "case_count": 270,
        "cases_sha256": hashlib.sha256(canonical).hexdigest(),
        "global_forbidden_answer_fragments": independent.get("global_forbidden_answer_fragments", []),
        "release_gates": independent.get("release_gates", {}),
        "dialogues": independent.get("dialogues", []),
        "cases": cases,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the immutable Routing v3 270-case control corpus.")
    parser.add_argument("--widget-audit", type=Path, default=DEFAULT_WIDGET_AUDIT)
    parser.add_argument("--independent-audit", type=Path, default=DEFAULT_INDEPENDENT_AUDIT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = build(args.widget_audit, args.independent_audit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "count": payload["case_count"], "sha256": payload["cases_sha256"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
