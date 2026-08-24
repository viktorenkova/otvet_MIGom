from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


DEFAULT_SOURCE = Path("reports/quality-stage0-adjudicated-error-taxonomy.json")
DEFAULT_OUTPUT = Path("tests/data/stage3_hard_negatives.json")


def build(source_path: Path) -> dict:
    source_raw = source_path.read_bytes()
    source = json.loads(source_raw.decode("utf-8"))
    cases = []
    for item in source["cases"]:
        if item.get("routing_subcause") != "rerank_wrong" or not item.get("correct_candidate_in_top10"):
            continue
        digest = hashlib.sha256(str(item["id"]).encode("utf-8")).digest()
        cases.append({
            "id": item["id"],
            "split": "validation" if digest[0] % 5 == 0 else "development",
            "group": item["group"],
            "class": item["class"],
            "text": item["text"],
            "expected_scenario_ids": item["expected_scenario_ids"],
            "historical_top10_candidate_ids": item["top10_candidate_ids"],
            "historical_wrong_scenario_id": item["actual_scenario_id"],
        })
    canonical = json.dumps(cases, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "schema_version": 1,
        "version": "2026.08.24.1",
        "purpose": "Real adjudicated rerank conflicts. Query text is benchmark/training data only and must never be copied into rules or KB aliases.",
        "source_artifact": source_path.name,
        "source_sha256": hashlib.sha256(source_raw).hexdigest(),
        "case_count": len(cases),
        "development_count": sum(case["split"] == "development" for case in cases),
        "validation_count": sum(case["split"] == "validation" for case in cases),
        "cases_sha256": hashlib.sha256(canonical).hexdigest(),
        "cases": cases,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = build(args.source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("case_count", "development_count", "validation_count", "cases_sha256")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
