"""Offline, fail-closed evidence checks. Does not run a model or authorize deployment."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
from uuid import uuid4
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

ROOT = Path(__file__).resolve().parents[2]
POLICY_PATH = ROOT / "configs/independent_acceptance_policy.json"
OUTCOMES = Literal["answer", "clarify", "offer_ticket", "create_ticket", "status", "out_of_scope", "safe_refusal"]


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def stamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("timezone_required")
    return parsed


def normalized(value: str) -> str:
    return re.sub(r"\s+", " ", value.casefold().replace("ё", "е")).strip()


class Strict(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", allow_inf_nan=False)


class Attestation(Strict):
    reviewer_ids: list[str] = Field(min_length=1)
    contributors: list[str] = Field(min_length=1)
    independent: bool
    labels_before_candidate: bool
    unused_for_development: bool
    privacy_reviewed: bool
    source_verified: bool
    reviewed_at: str


class Case(Strict):
    id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    source_event_id: str = Field(min_length=1)
    source_conversation_id: str = Field(min_length=1)
    role: Literal["guest", "authorized"]
    topic: str = Field(min_length=1)
    slices: list[str] = Field(min_length=1)
    unambiguous: bool
    expected_scenario_ids: list[str]
    required_information: list[str]
    forbidden_information: list[str]
    allowed_outcomes: list[OUTCOMES] = Field(min_length=1)
    reviewer_id: str = Field(min_length=1)


class Dialogue(Strict):
    id: str = Field(min_length=1)
    ambiguous: bool
    turns: list[Case] = Field(min_length=2, max_length=2)


class Pack(Strict):
    protocol: str
    dataset_id: str = Field(min_length=1)
    attestation: Attestation
    cases: list[Case]
    dialogues: list[Dialogue]


class Observation(Strict):
    id: str
    response: dict
    candidates: list[str] = Field(max_length=10)
    latency_ms: float = Field(gt=0)
    baseline_latency_ms: float = Field(gt=0)
    transport_ok: bool
    outcome: OUTCOMES


class Judgment(Strict):
    id: str
    response_sha256: str
    observation_sha256: str
    reviewer_id: str
    reviewed_at: str
    answer_success: bool
    critical_fact_violation: bool
    critical_access_violation: bool
    dialogue_resolved: bool | None = None


def load_policy():
    return json.loads(POLICY_PATH.read_text(encoding="utf-8"))


def records(pack: Pack):
    return [*pack.cases, *(t for d in pack.dialogues for t in d.turns)]


def validate_pack(raw, excluded_texts=(), excluded_events=()):
    policy = load_policy()
    try:
        pack = Pack.model_validate(raw)
        stamp(pack.attestation.reviewed_at)
    except (ValidationError, ValueError, TypeError):
        return None, {"schema_valid": False}
    rows = records(pack)
    att = pack.attestation
    texts = [normalized(c.text) for c in pack.cases]
    # Short follow-ups can legitimately repeat. Entire dialogues cannot.
    pairs = [tuple(normalized(t.text) for t in d.turns) for d in pack.dialogues]
    singles_conversations = [c.source_conversation_id for c in pack.cases]
    dialogue_conversations = [d.turns[0].source_conversation_id for d in pack.dialogues]
    all_conversations = singles_conversations + dialogue_conversations
    excluded = {normalized(t) for t in excluded_texts}
    from backend.app.bot.scenario_engine import load_scenarios
    valid_scenarios = {s.scenario_id for s in load_scenarios()}
    checks = {
        "schema_valid": True,
        "current_protocol": pack.protocol == policy["protocol"],
        "single_count": len(pack.cases) >= policy["single_min"],
        "dialogue_count": len(pack.dialogues) >= policy["dialogue_min"],
        "independent_attestation": all((att.independent, att.labels_before_candidate,
            att.unused_for_development, att.privacy_reviewed, att.source_verified))
            and not set(att.reviewer_ids) & set(att.contributors)
            and all(c.reviewer_id in att.reviewer_ids for c in rows),
        "unique_record_ids": len({r.id for r in rows}) == len(rows),
        "unique_dialogue_ids": len({d.id for d in pack.dialogues}) == len(pack.dialogues),
        "unique_source_events": len({r.source_event_id for r in rows}) == len(rows),
        "separate_conversations": len(set(all_conversations)) == len(all_conversations),
        "real_two_turn_dialogues": all(d.turns[0].source_conversation_id == d.turns[1].source_conversation_id
            and d.turns[0].role == d.turns[1].role for d in pack.dialogues),
        "unique_samples": len(set(texts)) == len(texts) and len(set(pairs)) == len(pairs),
        "single_dialogue_text_separation": not set(texts) & {p[0] for p in pairs},
        "known_corpus_excluded": not any(normalized(c.text) in excluded for c in pack.cases)
            and not any(normalized(d.turns[0].text) in excluded for d in pack.dialogues)
            and not {r.source_event_id for r in rows} & set(excluded_events),
        "valid_scenario_labels": all(set(r.expected_scenario_ids) <= valid_scenarios for r in rows),
        "answer_labels_present": all(r.expected_scenario_ids and r.required_information
            for r in rows if set(r.allowed_outcomes) & {"answer", "status"}),
        "known_topics": all(r.topic in policy["required_topics"] for r in rows),
        "role_coverage": set(policy["required_roles"]) <= {r.role for r in pack.cases},
        "slice_coverage": set(policy["required_slices"]) <= {s for r in pack.cases for s in r.slices},
        "topic_coverage": set(policy["required_topics"]) <= {r.topic for r in pack.cases},
        "ambiguous_dialogues_present": any(d.ambiguous for d in pack.dialogues),
    }
    return pack, checks


def metric(values):
    values = list(values)
    passed, total = sum(values), len(values)
    return {"passed": passed, "total": total, "pct": 100 * passed / total if total else None}


def wilson(passed, total):
    if not total:
        return None
    z = 1.959963984540054
    p = passed / total
    return 100 * (p + z*z/(2*total) - z*math.sqrt(p*(1-p)/total + z*z/(4*total*total))) / (1+z*z/total)


def p95(values):
    values = sorted(values)
    return values[math.ceil(.95 * len(values))-1] if values else None


def evaluate(pack_raw, frozen, run, judgments, current_bundle, excluded_texts=(), excluded_events=()):
    pack, checks = validate_pack(pack_raw, excluded_texts, excluded_events)
    result = {"protocol": load_policy()["protocol"], "blind_gate_passed": False,
        "eligible_for_shadow": False, "production_release_allowed": False, "checks": checks,
        "bundle_sha256": current_bundle, "completed_at": run.get("completed_at"), "branch": run.get("branch"),
        "evidence_limit": "Local file integrity and human attestations; identities and source provenance require independent verification."}
    if not pack or not all(checks.values()):
        return result
    policy = load_policy()
    checks["frozen_pack"] = frozen.get("pack_sha256") == digest(pack_raw)
    checks["frozen_policy"] = frozen.get("policy_sha256") == digest(policy)
    checks["current_bundle"] = bool(current_bundle) and frozen.get("bundle_sha256") == run.get("bundle_sha256") == current_bundle
    checks["fresh_measured_run"] = run.get("kind") == "fresh_full_pipeline" and run.get("warmup_complete") is True
    checks["paired_latency"] = run.get("paired_baseline") is True and bool(run.get("baseline_bundle_sha256"))
    checks["wording_disabled"] = run.get("wording_enabled") is False
    checks["supported_branch"] = run.get("branch") in {"local", "llm"}
    checks["llm_selection_gate"] = run.get("branch") != "llm" or run.get("llm_selection_passed") is True
    try:
        observed = [Observation.model_validate(r) for r in run["records"]]
        reviewed = [Judgment.model_validate(r) for r in judgments]
        checks["chronology"] = stamp(pack.attestation.reviewed_at) <= stamp(frozen["frozen_at"]) <= stamp(run["started_at"]) <= stamp(run["completed_at"])
        checks["post_run_review"] = all(stamp(j.reviewed_at) >= stamp(run["completed_at"]) for j in reviewed)
    except (ValidationError, ValueError, KeyError, TypeError):
        checks["run_and_review_schema"] = False
        return result
    ids = {r.id for r in records(pack)}
    checks["full_observation_coverage"] = len(observed) == len(ids) and {r.id for r in observed} == ids
    checks["full_review_coverage"] = len(reviewed) == len(ids) and {r.id for r in reviewed} == ids
    observations, reviews = {r.id: r for r in observed}, {r.id: r for r in reviewed}
    checks["reviewer_independence"] = all(j.reviewer_id in pack.attestation.reviewer_ids for j in reviewed)
    checks["review_bound_to_response"] = all(j.id in observations and
        j.response_sha256 == digest(observations[j.id].response) and
        j.observation_sha256 == digest(observations[j.id].model_dump()) for j in reviewed)
    checks["response_fields_present"] = all(r.response.get("confidence_level") in {"high", "medium", "low"}
        and "scenario_id" in r.response and isinstance(r.response.get("answer"), str) for r in observed)
    checks["measured_roles_match"] = all(c.id in observations and observations[c.id].response.get("role") == c.role
        for c in records(pack))
    if not all(checks.values()):
        return result
    checks["all_transport_ok"] = all(o.transport_ok for o in observed)
    def success(c):
        o, j = observations[c.id], reviews[c.id]
        return o.transport_ok and j.answer_success and o.outcome in c.allowed_outcomes and not (
            j.critical_fact_violation or j.critical_access_violation)
    def route(c):
        return observations[c.id].response.get("scenario_id") in c.expected_scenario_ids
    singles = pack.cases
    routed = [c for c in singles if c.expected_scenario_ids and route(c)]
    labeled = [c for c in singles if c.expected_scenario_ids]
    unambiguous = [c for c in labeled if c.unambiguous]
    high = [c for c in singles if observations[c.id].response["confidence_level"] == "high"]
    metrics = {
        "e2e": metric(success(c) for c in singles),
        "top1_unambiguous": metric(route(c) for c in unambiguous),
        "recall10": metric(bool(set(c.expected_scenario_ids) & set(observations[c.id].candidates)) for c in labeled),
        "correct_route_answer": metric(success(c) for c in routed),
        "confident_wrong_high": metric(not success(c) for c in high),
        "confident_wrong_all": metric(c in high and not success(c) for c in singles),
        "ambiguous_dialogues": metric(all(success(t) for t in d.turns) and
            reviews[d.turns[1].id].dialogue_resolved is True for d in pack.dialogues if d.ambiguous),
        "slices": {s: metric(success(c) for c in singles if s in c.slices) for s in policy["required_slices"]},
        "topics": {s: metric(success(c) for c in singles if c.topic == s) for s in policy["required_topics"]},
        "outcomes": dict(Counter(observations[c.id].outcome for c in singles)),
        "ticket_created": sum(observations[c.id].outcome == "create_ticket" for c in singles),
        "mailbox_delivery_confirmed": None,
        "p95_ms": p95(o.latency_ms for o in observed),
        "baseline_p95_ms": p95(o.baseline_latency_ms for o in observed),
    }
    metrics["wilson_lower_pct"] = wilson(metrics["e2e"]["passed"], metrics["e2e"]["total"])
    metrics["p95_degradation_pct"] = (metrics["p95_ms"] / metrics["baseline_p95_ms"] - 1) * 100
    for key, threshold in (("e2e", "e2e_min_pct"), ("top1_unambiguous", "top1_min_pct"),
        ("recall10", "recall10_min_pct"), ("correct_route_answer", "correct_route_answer_min_pct"),
        ("ambiguous_dialogues", "ambiguous_dialogue_min_pct")):
        checks[key] = metrics[key]["pct"] is not None and metrics[key]["pct"] >= policy[threshold]
    checks["wilson"] = metrics["wilson_lower_pct"] >= policy["wilson_lower_min_pct"]
    checks["confident_wrong"] = bool(high) and metrics["confident_wrong_high"]["pct"] <= policy["confident_wrong_max_pct"]
    checks["critical_violations_zero"] = not any(j.critical_fact_violation or j.critical_access_violation for j in reviewed)
    checks["all_slices"] = all(m["pct"] is not None and m["pct"] >= policy["slice_e2e_min_pct"] for m in metrics["slices"].values())
    checks["all_topics"] = all(m["pct"] is not None and m["pct"] >= policy["topic_e2e_min_pct"] for m in metrics["topics"].values())
    checks["latency"] = metrics["p95_ms"] <= policy["p95_max_ms"] and (run["branch"] == "llm" or
        metrics["p95_degradation_pct"] <= policy["local_p95_degradation_max_pct"] + 1e-9)
    result.update(metrics=metrics, blind_gate_passed=all(checks.values()), eligible_for_shadow=all(checks.values()))
    return result


def begin_run(pack_raw, frozen, bundle, ledger: Path):
    """Consume the blind corpus before execution, even if the run later crashes."""
    if frozen.get("pack_sha256") != digest(pack_raw) or frozen.get("bundle_sha256") != bundle:
        raise ValueError("frozen_bundle_or_pack_mismatch")
    if frozen.get("policy_sha256") != digest(load_policy()):
        raise ValueError("frozen_policy_mismatch")
    if stamp(frozen["frozen_at"]) > datetime.now(timezone.utc):
        raise ValueError("freeze_in_future")
    receipt = {"run_id": str(uuid4()), "pack_sha256": digest(pack_raw), "bundle_sha256": bundle,
        "manifest_sha256": digest(frozen), "started_at": datetime.now(timezone.utc).isoformat(),
        "corpus_status": "disclosed_regression_only_after_this_run"}
    ledger.mkdir(parents=True, exist_ok=True)
    with (ledger / (digest(pack_raw) + ".json")).open("x", encoding="utf-8") as out:
        json.dump(receipt, out, ensure_ascii=False, indent=2)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["validate", "freeze", "begin", "evaluate"])
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--exclusions", type=Path, required=True, help="JSON texts/events registry from the independent custodian")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--bundle", required=True, help="Canonical runtime manifest SHA256 from the evaluated candidate")
    parser.add_argument("--run", type=Path)
    parser.add_argument("--reviews", type=Path)
    parser.add_argument("--shadow", type=Path, help="Optional new real-traffic evidence, evaluated only after the blind gate")
    parser.add_argument("--ledger", type=Path, default=Path(".work/independent-acceptance-ledger"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = json.loads(args.pack.read_text(encoding="utf-8"))
    exclusions = json.loads(args.exclusions.read_text(encoding="utf-8"))
    if not exclusions.get("texts") or exclusions.get("custodian_verified") is not True:
        parser.error("A nonempty, custodian-verified exclusion registry is required")
    _, checks = validate_pack(raw, exclusions["texts"], exclusions.get("events", []))
    result = {"freeze_ready": all(checks.values()), "checks": checks, "production_release_allowed": False}
    if args.command == "freeze" and all(checks.values()):
        result = {"pack_sha256": digest(raw), "policy_sha256": digest(load_policy()),
            "exclusions_sha256": digest(exclusions), "bundle_sha256": args.bundle,
            "frozen_at": datetime.now(timezone.utc).isoformat(), "production_release_allowed": False}
    elif args.command == "begin":
        if not args.manifest or not all(checks.values()):
            parser.error("begin requires a frozen, valid reviewed pack")
        result = begin_run(raw, json.loads(args.manifest.read_text(encoding="utf-8")), args.bundle, args.ledger)
    elif args.command == "evaluate":
        if not all((args.manifest, args.run, args.reviews)):
            parser.error("evaluate requires --manifest, --run and --reviews")
        frozen = json.loads(args.manifest.read_text(encoding="utf-8"))
        if frozen.get("exclusions_sha256") != digest(exclusions):
            parser.error("Exclusions differ from the frozen registry")
        receipt = json.loads((args.ledger / (digest(raw) + ".json")).read_text(encoding="utf-8"))
        run = json.loads(args.run.read_text(encoding="utf-8"))
        if (receipt["manifest_sha256"] != digest(frozen) or receipt["run_id"] != run.get("run_id")
            or receipt["bundle_sha256"] != args.bundle or stamp(run["started_at"]) < stamp(receipt["started_at"])):
            parser.error("Run does not match its first-use receipt")
        result = evaluate(raw, frozen, run,
            json.loads(args.reviews.read_text(encoding="utf-8")), args.bundle, exclusions["texts"], exclusions.get("events", []))
        if args.shadow:
            from backend.tools.independent_shadow import evaluate_shadow
            pack, _ = validate_pack(raw)
            result["shadow"] = evaluate_shadow(json.loads(args.shadow.read_text(encoding="utf-8")),
                dict(result), current_bundle=args.bundle,
                seen_source_events=[*exclusions.get("events", []), *(r.source_event_id for r in records(pack))])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Never overwrite evidence or silently re-freeze a disclosed corpus.
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(result, output, ensure_ascii=False, indent=2)
        output.write("\n")
    ready = result.get("blind_gate_passed", result.get("freeze_ready", "pack_sha256" in result))
    if "shadow" in result:
        ready = ready and result["shadow"]["shadow_gate_passed"]
    print(json.dumps({"output": str(args.output), "ready": ready}))
    return 0 if ready else 2


if __name__ == "__main__":
    raise SystemExit(main())
