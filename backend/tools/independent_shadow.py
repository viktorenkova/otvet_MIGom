"""Validate real shadow provenance and expert coverage; never change runtime rollout."""
from datetime import datetime, timezone
import random

from backend.tools.independent_acceptance import digest, load_policy, metric, p95, stamp


def evaluate_shadow(evidence, blind_gate, *, current_bundle, seen_source_events=()):
    checks = {
        "blind_gate_passed": blind_gate.get("eligible_for_shadow") is True,
        "same_blind_evidence": evidence.get("blind_gate_sha256") == digest(blind_gate),
        "current_bundle": bool(current_bundle) and evidence.get("bundle_sha256") == blind_gate.get("bundle_sha256") == current_bundle,
        "blind_timestamp_bound": evidence.get("blind_completed_at") == blind_gate.get("completed_at"),
        "real_source_attested": evidence.get("mode") == "real_new_traffic"
            and evidence.get("source_verified") is True and bool(evidence.get("custodian_id")),
        "independent_review": evidence.get("reviewer_independent") is True
            and bool(evidence.get("reviewer_id")) and evidence.get("reviewer_id") not in evidence.get("contributors", []),
    }
    result = {"shadow_gate_passed": False, "eligible_rollout_percentage": 0,
              "production_release_allowed": False, "checks": checks}
    try:
        events = evidence["events"]
        ids = [e["source_event_id"] for e in events]
        checks["new_unique_events"] = len(set(ids)) == len(ids) and all(ids) and not set(ids) & set(seen_source_events)
        checks["minimum_1000"] = len(events) >= load_policy()["shadow_min_new_events"]
        completed = stamp(evidence["blind_completed_at"])
        checks["after_blind_gate"] = all(completed < stamp(e["observed_at"]) <= datetime.now(timezone.utc) for e in events)
        checks["finite_latencies_and_booleans"] = all(
            type(e[k]) in (int, float) and 0 < e[k] < float("inf")
            for e in events for k in ("candidate_latency_ms", "baseline_latency_ms")) and all(
            type(e[k]) is bool for e in events for k in ("candidate_ok", "baseline_ok", "high", "disagreement"))
        reviews = evidence["reviews"]
        by_id = {r["source_event_id"]: r for r in reviews}
        checks["review_unique_known_ids"] = len(by_id) == len(reviews) and set(by_id) <= set(ids)
        # Reproducible control sample over the sealed event population; chosen before judging.
        control = set(random.Random(digest(sorted(ids))).sample(sorted(ids), min(50, len(ids))))
        required = {e["source_event_id"] for e in events if e["high"] or e["disagreement"]} | control
        checks["review_coverage"] = bool(required) and required <= set(by_id)
        event_map = {e["source_event_id"]: e for e in events}
        checks["reviews_bound_to_events"] = all(r["source_event_id"] in event_map and
            r["event_sha256"] == digest(event_map[r["source_event_id"]]) and
            stamp(r["reviewed_at"]) >= stamp(event_map[r["source_event_id"]]["observed_at"])
            for r in reviews)
        checks["expert_booleans"] = all(type(r[k]) is bool for r in reviews
            for k in ("answer_success", "critical_fact_violation", "critical_access_violation"))
    except (KeyError, TypeError, ValueError, OverflowError):
        checks["evidence_schema"] = False
        return result
    if not all(checks.values()):
        return result
    quality = metric(r["answer_success"] for r in reviews)
    high = [e for e in events if e["high"]]
    wrong_high = metric(not by_id[e["source_event_id"]]["answer_success"] for e in high)
    candidate_p95 = p95(e["candidate_latency_ms"] for e in events)
    baseline_p95 = p95(e["baseline_latency_ms"] for e in events)
    errors_delta = 100 * (sum(not e["candidate_ok"] for e in events) - sum(not e["baseline_ok"] for e in events)) / len(events)
    checks.update(
        expert_quality=quality["pct"] is not None and quality["pct"] >= 80,
        confident_wrong=wrong_high["pct"] is not None and wrong_high["pct"] <= 2,
        critical_zero=not any(r["critical_fact_violation"] or r["critical_access_violation"] for r in reviews),
        latency=candidate_p95 <= 5000 and (blind_gate.get("branch") == "llm" or candidate_p95 <= baseline_p95 * 1.1),
        error_degradation=errors_delta <= .5,
    )
    result.update(shadow_gate_passed=all(checks.values()), eligible_rollout_percentage=5 if all(checks.values()) else 0,
        metrics={"events": len(events), "expert_sample": quality, "confident_wrong_high": wrong_high,
            "control_sample_count": len(control), "candidate_p95_ms": candidate_p95,
            "baseline_p95_ms": baseline_p95, "api_error_delta_pp": errors_delta})
    return result


def next_rollout_step(current, requested, *, current_window_passed, rollback_verified):
    """Decision aid only. Every step requires evidence for its own window."""
    steps = [0, *load_policy()["rollout_percentages"]]
    return (current in steps[:-1] and requested == steps[steps.index(current)+1]
            and current_window_passed is True and rollback_verified is True)
