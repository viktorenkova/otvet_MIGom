"""Synthetic fixtures test the gate, never supply independent product evidence."""
from copy import deepcopy
import pytest

from backend.tools.independent_acceptance import begin_run, digest, evaluate, load_policy, validate_pack, wilson


@pytest.fixture
def evidence():
    policy = load_policy()
    def case(i, conversation=None):
        return {"id": f"record-{i}", "text": f"Синтетическая проверка механизма {i}",
            "source_event_id": f"source-{i}", "source_conversation_id": conversation or f"conversation-{i}",
            "role": "guest" if i % 2 else "authorized", "topic": policy["required_topics"][i % len(policy["required_topics"])],
            "slices": [policy["required_slices"][i % len(policy["required_slices"])]], "unambiguous": True,
            "expected_scenario_ids": ["buyer.get_started"], "required_information": ["Ожидаемое содержание"],
            "forbidden_information": [], "allowed_outcomes": ["answer"], "reviewer_id": "reviewer"}
    singles = [case(i) for i in range(500)]
    dialogues = []
    for i in range(100):
        turns = [case(500+2*i+j, f"dialogue-source-{i}") for j in range(2)]
        turns[1]["role"] = turns[0]["role"]
        dialogues.append({"id": f"dialogue-{i}", "ambiguous": True, "turns": turns})
    pack = {"protocol": policy["protocol"], "dataset_id": "synthetic-test-only", "cases": singles, "dialogues": dialogues,
        "attestation": {"reviewer_ids": ["reviewer"], "contributors": ["developer"], "independent": True,
            "labels_before_candidate": True, "unused_for_development": True, "privacy_reviewed": True,
            "source_verified": True, "reviewed_at": "2026-09-04T10:00:00Z"}}
    frozen = {"pack_sha256": digest(pack), "policy_sha256": digest(policy), "bundle_sha256": "bundle",
        "frozen_at": "2026-09-04T11:00:00Z"}
    rows, reviews = [], []
    for record in singles + [t for d in dialogues for t in d["turns"]]:
        response = {"scenario_id": "buyer.get_started", "confidence_level": "high", "answer": "Проверочный ответ", "role": record["role"]}
        rows.append({"id": record["id"], "response": response, "candidates": ["buyer.get_started"],
            "latency_ms": 100.0, "baseline_latency_ms": 100.0, "transport_ok": True, "outcome": "answer"})
        reviews.append({"id": record["id"], "response_sha256": digest(response), "reviewer_id": "reviewer",
            "observation_sha256": digest(rows[-1]),
            "reviewed_at": "2026-09-04T14:00:00Z", "answer_success": True,
            "critical_fact_violation": False, "critical_access_violation": False, "dialogue_resolved": True})
    run = {"bundle_sha256": "bundle", "kind": "fresh_full_pipeline", "warmup_complete": True,
        "paired_baseline": True, "baseline_bundle_sha256": "baseline", "wording_enabled": False,
        "branch": "local", "started_at": "2026-09-04T12:00:00Z", "completed_at": "2026-09-04T13:00:00Z", "records": rows}
    return pack, frozen, run, reviews


def gate(evidence):
    return evaluate(*evidence, current_bundle="bundle")


def test_complete_gate_only_allows_shadow_not_production(evidence):
    result = gate(evidence)
    assert result["blind_gate_passed"] and result["eligible_for_shadow"]
    assert result["production_release_allowed"] is False
    assert result["metrics"]["e2e"]["total"] == 500


@pytest.mark.parametrize("mutation", [
    lambda p: p["dialogues"].pop(),
    lambda p: p["dialogues"][0]["turns"].pop(),
    lambda p: p["cases"].pop(),
    lambda p: p["attestation"].update(independent=False),
    lambda p: p["attestation"].update(unused_for_development=False),
    lambda p: p["attestation"].update(contributors=["reviewer"]),
    lambda p: p["cases"][0].update(id=p["cases"][1]["id"]),
    lambda p: p["cases"][0].update(source_event_id=p["cases"][1]["source_event_id"]),
    lambda p: p["cases"][0].update(source_conversation_id=p["dialogues"][0]["turns"][0]["source_conversation_id"]),
    lambda p: p["cases"][0].update(text=p["cases"][1]["text"]),
    lambda p: p["cases"][0].update(expected_scenario_ids=["invented"]),
])
def test_bad_pack_cannot_freeze(evidence, mutation):
    pack = evidence[0]
    mutation(pack)
    _, checks = validate_pack(pack)
    assert not all(checks.values())


def test_known_text_and_event_rejected_without_printing_text(evidence):
    pack = evidence[0]
    _, checks = validate_pack(pack, [pack["cases"][0]["text"]], [])
    assert not checks["known_corpus_excluded"]
    _, checks = validate_pack(pack, [], [pack["dialogues"][0]["turns"][1]["source_event_id"]])
    assert not checks["known_corpus_excluded"]


@pytest.mark.parametrize("field", ["pack_sha256", "policy_sha256", "bundle_sha256"])
def test_changed_frozen_artifact_rejected(evidence, field):
    evidence[1][field] = "changed"
    assert not gate(evidence)["blind_gate_passed"]


@pytest.mark.parametrize("change", ["missing", "duplicate", "response", "reviewer", "old_time", "nan", "snapshot"])
def test_incomplete_or_unbound_evidence_rejected(evidence, change):
    _, _, run, reviews = evidence
    if change == "missing": run["records"].pop()
    elif change == "duplicate": run["records"][-1] = deepcopy(run["records"][0])
    elif change == "response": run["records"][0]["response"]["answer"] = "Изменено"
    elif change == "reviewer": reviews[0]["reviewer_id"] = "developer"
    elif change == "old_time": reviews[0]["reviewed_at"] = "2026-09-04T09:00:00Z"
    elif change == "nan": run["records"][0]["latency_ms"] = float("nan")
    elif change == "snapshot": run["kind"] = "cached_report"
    assert not gate(evidence)["blind_gate_passed"]


def test_high_confidence_errors_use_high_denominator(evidence):
    _, _, run, reviews = evidence
    for i in range(100, 500):
        run["records"][i]["response"]["confidence_level"] = "medium"
        reviews[i]["response_sha256"] = digest(run["records"][i]["response"])
        reviews[i]["observation_sha256"] = digest(run["records"][i])
    for i in range(3): reviews[i]["answer_success"] = False
    result = gate(evidence)
    assert result["metrics"]["confident_wrong_high"]["pct"] == 3
    assert result["metrics"]["confident_wrong_all"]["pct"] == .6
    assert not result["checks"]["confident_wrong"]


def test_blanket_abstention_cannot_pass(evidence):
    _, _, run, reviews = evidence
    for row, review in zip(run["records"], reviews):
        row["outcome"] = "clarify"
        row["response"].update(scenario_id=None, confidence_level="low")
        review["response_sha256"] = digest(row["response"])
        review["observation_sha256"] = digest(row)
    result = gate(evidence)
    assert result["metrics"]["e2e"]["passed"] == 0
    assert not result["blind_gate_passed"]


def test_critical_violation_on_dialogue_blocks_gate(evidence):
    evidence[3][-1]["critical_access_violation"] = True
    assert not gate(evidence)["checks"]["critical_violations_zero"]


def test_local_latency_regression_blocks_even_below_five_seconds(evidence):
    for row, review in zip(evidence[2]["records"], evidence[3]):
        row["latency_ms"] = 111.0
        review["observation_sha256"] = digest(row)
    assert not gate(evidence)["checks"]["latency"]


def test_missing_dialogue_resolution_is_failure(evidence):
    for review in evidence[3][500:]: review["dialogue_resolved"] = None
    assert not gate(evidence)["checks"]["ambiguous_dialogues"]


def test_wilson_and_empty_denominators():
    assert wilson(420, 500) == pytest.approx(80.53, abs=.02)
    assert wilson(0, 0) is None


def test_corpus_consumed_before_execution_and_cannot_restart(evidence, tmp_path):
    receipt = begin_run(evidence[0], evidence[1], "bundle", tmp_path)
    assert receipt["run_id"]
    with pytest.raises(FileExistsError):
        begin_run(evidence[0], evidence[1], "bundle", tmp_path)


def test_changed_observation_cannot_reuse_review(evidence):
    evidence[2]["records"][0]["candidates"] = []
    assert not gate(evidence)["checks"]["review_bound_to_response"]


def test_preparation_creates_no_fake_cases_or_success(tmp_path):
    from backend.tools.prepare_independent_acceptance import prepare
    import json
    report = prepare(tmp_path)
    pack = json.loads((tmp_path / "pack-template.json").read_text(encoding="utf-8"))
    assert pack["cases"] == pack["dialogues"] == []
    assert report["new_independent_singles"] == 0
    assert report["production_release_allowed"] is False
    with pytest.raises(FileExistsError):
        prepare(tmp_path)


def test_cli_freeze_first_use_and_evaluate_bound_files(evidence, tmp_path, monkeypatch):
    import json
    from datetime import datetime, timezone
    from backend.tools.independent_acceptance import main
    pack, _, run, reviews = evidence
    def write(name, value):
        (tmp_path/name).write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    def command(action, output, *extra):
        monkeypatch.setattr("sys.argv", ["gate", action, "--pack", str(tmp_path/"pack.json"),
            "--exclusions", str(tmp_path/"exclusions.json"), "--bundle", "bundle", "--output", str(tmp_path/output),
            "--ledger", str(tmp_path/"ledger"), *extra])
        return main()
    write("pack.json", pack)
    write("exclusions.json", {"custodian_verified": True, "texts": ["известный старый запрос"], "events": []})
    assert command("freeze", "manifest.json") == 0
    assert command("begin", "receipt.json", "--manifest", str(tmp_path/"manifest.json")) == 0
    receipt = json.loads((tmp_path/"receipt.json").read_text(encoding="utf-8"))
    run.update(run_id=receipt["run_id"], started_at=receipt["started_at"], completed_at=datetime.now(timezone.utc).isoformat())
    for review in reviews:
        review["reviewed_at"] = datetime.now(timezone.utc).isoformat()
    write("run.json", run)
    write("reviews.json", reviews)
    assert command("evaluate", "report.json", "--manifest", str(tmp_path/"manifest.json"),
        "--run", str(tmp_path/"run.json"), "--reviews", str(tmp_path/"reviews.json")) == 0
    assert json.loads((tmp_path/"report.json").read_text(encoding="utf-8"))["eligible_for_shadow"] is True
