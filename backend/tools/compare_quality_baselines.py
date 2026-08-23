from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


DEFAULT_LOCAL = Path("reports/quality-stage0-local.json")
DEFAULT_REMOTE = Path("reports/quality-stage0-dev.json")
DEFAULT_OUTPUT = Path("reports/quality-stage0-local-dev-comparison.json")

COMPARABILITY_KEYS = (
    "git_sha",
    "knowledge_sha256",
    "scenarios_sha256",
    "matching_config_sha256",
    "application_bundle_sha256",
    "routing_bundle_sha256",
    "prompt_bundle_sha256",
    "knowledge_mode",
    "llm_enabled",
    "llm_provider",
    "llm_primary_model",
)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def compare(local_path: Path, remote_path: Path) -> dict[str, Any]:
    local = _load(local_path)
    remote = _load(remote_path)
    local_manifest = local.get("runtime_manifest")
    remote_manifest = remote.get("runtime_manifest")
    manifest_available = isinstance(local_manifest, dict) and isinstance(remote_manifest, dict)
    manifest_checks = {
        key: bool(manifest_available and local_manifest.get(key) == remote_manifest.get(key))
        for key in COMPARABILITY_KEYS
    }
    same_dataset = local.get("dataset", {}).get("sha256") == remote.get("dataset", {}).get("sha256")

    local_cases = {item["id"]: item for item in local.get("single_turn_results", [])}
    remote_cases = {item["id"]: item for item in remote.get("single_turn_results", [])}
    common_ids = sorted(set(local_cases).intersection(remote_cases))
    response_matches = 0
    mismatches: list[dict[str, Any]] = []
    for case_id in common_ids:
        left = local_cases[case_id].get("response") or {}
        right = remote_cases[case_id].get("response") or {}
        comparable = {
            "scenario_id": left.get("scenario_id") == right.get("scenario_id"),
            "intent": left.get("intent") == right.get("intent"),
            "resolution": left.get("resolution") == right.get("resolution"),
        }
        if all(comparable.values()):
            response_matches += 1
        else:
            mismatches.append({
                "id": case_id,
                "local": {key: left.get(key) for key in comparable},
                "remote": {key: right.get(key) for key in comparable},
                "equal": comparable,
            })
    same_build = manifest_available and all(manifest_checks.values())
    deterministic_gate_passed = same_build and same_dataset and len(mismatches) <= 1
    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "local_report": str(local_path),
        "remote_report": str(remote_path),
        "manifest_available": manifest_available,
        "same_build": same_build,
        "same_dataset": same_dataset,
        "manifest_checks": manifest_checks,
        "case_count_local": len(local_cases),
        "case_count_remote": len(remote_cases),
        "common_case_count": len(common_ids),
        "matching_route_response_count": response_matches,
        "mismatch_count": len(mismatches),
        "deterministic_gate_passed": deterministic_gate_passed,
        "blocking_reasons": [
            reason
            for reason, blocked in (
                ("remote_manifest_missing", not manifest_available),
                ("build_or_runtime_mismatch", manifest_available and not same_build),
                ("dataset_mismatch", not same_dataset),
                ("more_than_one_route_result_mismatch", len(mismatches) > 1),
            )
            if blocked
        ],
        "mismatches": mismatches,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prove whether local and dev quality runs are comparable.")
    parser.add_argument("--local", type=Path, default=DEFAULT_LOCAL)
    parser.add_argument("--remote", type=Path, default=DEFAULT_REMOTE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = compare(args.local, args.remote)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: payload[key] for key in (
        "manifest_available", "same_build", "same_dataset", "common_case_count",
        "mismatch_count", "deterministic_gate_passed", "blocking_reasons",
    )}, ensure_ascii=False, indent=2))
    return 0 if payload["deterministic_gate_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
