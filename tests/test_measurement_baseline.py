from copy import deepcopy

from backend.tools.verify_measurement_baseline import compare_runs, scenario_coverage


def run(pid):
    return {"process_id": pid, "path": str(pid), "case_ids": ["one"], "manifest": {"version": "a"},
        "dataset_sha256": "dataset", "overlay_sha256": "overlay", "manifest_unchanged": True,
        "scenario_coverage": {"complete": True}, "threshold_config_matches": True,
        "candidate_calibration_artifact_matches": True,
        "rows": [{"id": "one", "transport_ok": True, "trace_complete": True,
            "response": {"scenario_id": "a"}, "candidate_ids": ["a"], "decision": {"reason": "same"}}]}


def test_identical_fresh_results_close_measurements_only():
    report = compare_runs([run(1), run(2), run(3)])
    assert report["measurement_stage_closed"]
    assert report["production_release_allowed"] is False


def test_new_runtime_cannot_be_proved_by_old_report():
    runs = [run(1), run(2), run(3)]
    runs[1]["manifest"] = {"version": "b"}
    assert not compare_runs(runs)["measurement_stage_closed"]


def test_equal_aggregate_counts_do_not_hide_changed_routes():
    runs = [run(1), run(2), run(3)]
    runs[1]["rows"][0]["response"]["scenario_id"] = "b"
    report = compare_runs(runs)
    assert not report["measurement_stage_closed"]
    assert report["differences"] == [{"case_id": "one", "path": "2", "changed": ["response"]}]


def test_missing_cases_or_reused_process_cannot_pass():
    runs = [run(1), run(2), run(3)]
    runs[1]["rows"] = []
    assert not compare_runs(runs)["measurement_stage_closed"]
    assert not compare_runs([run(1), run(1), run(1)])["measurement_stage_closed"]


def test_declared_supplementary_cards_do_not_require_scorer_columns():
    report = scenario_coverage({"a"}, {"a"}, {"a", "legacy"}, {"a"}, {"legacy"})
    assert report["complete"]
    assert report["supplementary_card_ids"] == ["legacy"]


def test_missing_canonical_card_or_scorer_column_blocks_coverage():
    assert not scenario_coverage({"a"}, set(), {"legacy"}, {"a"}, {"legacy"})["complete"]
    assert not scenario_coverage({"a"}, {"a"}, {"a"}, set(), set())["complete"]


def test_unknown_or_overlapping_supplementary_card_blocks_coverage():
    assert not scenario_coverage({"a"}, {"a"}, {"a", "unknown"}, {"a"}, set())["complete"]
    assert not scenario_coverage({"a"}, {"a"}, {"a"}, {"a"}, {"a"})["complete"]
