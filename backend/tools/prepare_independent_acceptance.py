"""Prepare empty schemas and a handoff; never generate or label independent queries."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from backend.tools.independent_acceptance import Pack, Observation, Judgment, digest, load_policy, validate_pack


def prepare(destination: Path):
    policy = load_policy()
    template = {"protocol": policy["protocol"], "dataset_id": "TO_BE_ASSIGNED_BY_CUSTODIAN",
        "attestation": {"reviewer_ids": [], "contributors": [], "independent": False,
            "labels_before_candidate": False, "unused_for_development": False, "privacy_reviewed": False,
            "source_verified": False, "reviewed_at": ""}, "cases": [], "dialogues": []}
    files = {"pack-template.json": template, "pack.schema.json": Pack.model_json_schema(),
        "observation.schema.json": Observation.model_json_schema(), "judgment.schema.json": Judgment.model_json_schema(),
        "exclusions-template.json": {"custodian_verified": False, "texts": [], "events": []},
        "run-template.json": {"run_id": "FROM_FIRST_USE_RECEIPT", "bundle_sha256": "FROM_CURRENT_RUNTIME_MANIFEST",
            "kind": "fresh_full_pipeline", "warmup_complete": False, "paired_baseline": False,
            "baseline_bundle_sha256": "", "wording_enabled": False, "branch": "local",
            "started_at": "", "completed_at": "", "records": []}}
    destination.mkdir(parents=True, exist_ok=True)
    for name, value in files.items():
        with (destination / name).open("x", encoding="utf-8") as out:
            json.dump(value, out, ensure_ascii=False, indent=2)
            out.write("\n")
    _, checks = validate_pack(template)
    return {"generated_at": datetime.now(timezone.utc).isoformat(), "protocol": policy["protocol"],
        "status": "engineering_handoff_only", "blind_gate_passed": False, "production_release_allowed": False,
        "new_independent_singles": 0, "new_independent_dialogues": 0, "required_singles": 500, "required_dialogues": 100,
        "template_checks": checks, "artifact_sha256": {name: digest(value) for name, value in files.items()},
        "not_performed": ["independent_collection_and_labeling", "fresh_blind_run", "expert_response_review",
            "new_real_shadow_1000", "full_suite_and_browser_release_checks", "rollout"],
        "historical_evidence": {"stage5_preparation": "500 singles / 50 dialogues reported historically; independence for this candidate is not established",
            "stage6_local_shadow": "Replayed development events; ineligible as new production shadow"}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=Path(".work/stage6-independent-handoff"))
    parser.add_argument("--report", type=Path, default=Path("reports/independent-acceptance-preparation.json"))
    args = parser.parse_args()
    report = prepare(args.destination)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("x", encoding="utf-8") as out:
        json.dump(report, out, ensure_ascii=False, indent=2)
        out.write("\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
