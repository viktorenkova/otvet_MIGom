from copy import deepcopy
import pytest

from backend.tools.independent_acceptance import digest
from backend.tools.independent_shadow import evaluate_shadow, next_rollout_step


@pytest.fixture
def evidence():
    gate = {"eligible_for_shadow": True, "bundle_sha256": "bundle", "completed_at": "2026-09-04T09:00:00Z", "branch": "local"}
    events = [{"source_event_id": f"real-source-{i}", "observed_at": "2026-09-04T10:00:00Z",
        "candidate_latency_ms": 100.0, "baseline_latency_ms": 100.0, "candidate_ok": True,
        "baseline_ok": True, "high": True, "disagreement": False,
        "candidate_response": {"answer": "тест"}, "baseline_response": {"answer": "тест"}}
        for i in range(1000)]
    reviews = [{"source_event_id": e["source_event_id"], "event_sha256": digest(e),
        "reviewed_at": "2026-09-04T11:00:00Z", "answer_success": True,
        "critical_fact_violation": False, "critical_access_violation": False} for e in events]
    return {"blind_gate_sha256": digest(gate), "bundle_sha256": "bundle", "mode": "real_new_traffic",
        "source_verified": True, "custodian_id": "custodian", "reviewer_independent": True,
        "reviewer_id": "reviewer", "contributors": ["developer"], "blind_completed_at": "2026-09-04T09:00:00Z",
        "events": events, "reviews": reviews}, gate


def test_complete_shadow_allows_only_first_step(evidence):
    report = evaluate_shadow(*evidence, current_bundle="bundle")
    assert report["shadow_gate_passed"] and report["eligible_rollout_percentage"] == 5
    assert report["production_release_allowed"] is False


@pytest.mark.parametrize("change", ["replay", "duplicate", "old", "review_missing", "changed_answer", "nan", "old_bundle", "no_blind"])
def test_shadow_rejects_nonfresh_or_incomplete_evidence(evidence, change):
    data, gate = evidence
    if change == "replay": data["mode"] = "local_replay"
    elif change == "duplicate": data["events"][-1] = deepcopy(data["events"][0])
    elif change == "old": data["events"][0]["observed_at"] = "2026-08-01T00:00:00Z"
    elif change == "review_missing": data["reviews"].pop()
    elif change == "changed_answer": data["events"][0]["candidate_response"]["answer"] = "изменено"
    elif change == "nan": data["events"][0]["candidate_latency_ms"] = float("nan")
    elif change == "old_bundle": data["bundle_sha256"] = "old"
    elif change == "no_blind": gate["eligible_for_shadow"] = False
    assert not evaluate_shadow(data, gate, current_bundle="bundle")["shadow_gate_passed"]


def test_new_local_id_cannot_hide_known_source_event(evidence):
    data, gate = evidence
    assert not evaluate_shadow(data, gate, current_bundle="bundle",
        seen_source_events=[data["events"][0]["source_event_id"]])["shadow_gate_passed"]


def test_rollout_cannot_skip_steps_or_ignore_rollback():
    assert next_rollout_step(5, 25, current_window_passed=True, rollback_verified=True)
    assert not next_rollout_step(0, 100, current_window_passed=True, rollback_verified=True)
    assert not next_rollout_step(25, 50, current_window_passed=False, rollback_verified=True)
    assert not next_rollout_step(25, 50, current_window_passed=True, rollback_verified=False)
