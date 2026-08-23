from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any


DEFAULT_SOURCE = Path("knowledge/v2/scenarios.json")
DEFAULT_KNOWLEDGE = Path("knowledge/v3_1/scenarios.json")
DEFAULT_CONFLICTS = Path("knowledge/v3_1/scenario_conflicts.json")
SENTINELS = {"__clarification__", "__no_scenario__"}
FINANCIAL_OR_CONTRACT_INTENTS = {
    "payment",
    "refund",
    "penalty",
    "tariffs",
    "transfer",
    "pickup",
    "refusal",
}


def validate(
    source: dict[str, Any],
    knowledge: dict[str, Any],
    conflicts: dict[str, Any],
) -> dict[str, Any]:
    errors: list[str] = []
    records = knowledge.get("records", [])
    source_records = source.get("records", [])
    by_id = {item.get("scenario_id"): item for item in records}
    source_by_id = {item.get("scenario_id"): item for item in source_records}
    if knowledge.get("schema_version") != "3.1":
        errors.append("schema_version must be 3.1")
    if len(records) != len(source_records):
        errors.append(f"scenario count changed: {len(source_records)} -> {len(records)}")
    if len(by_id) != len(records):
        errors.append("scenario_id values are not unique")

    signatures: Counter[str] = Counter()
    financial_fact_count = 0
    source_fact_count = 0
    migrated_fact_count = 0
    for scenario_id, record in by_id.items():
        prefix = f"{scenario_id}:"
        required = (
            "domain",
            "taxonomy",
            "discriminators",
            "source_version",
            "fact_records",
            "answer_policy",
            "search_document",
            "atomic_unit_ids",
        )
        for field in required:
            if not record.get(field):
                errors.append(f"{prefix} missing {field}")
        taxonomy = record.get("taxonomy", {})
        for field in ("domain", "objects", "actions", "states", "stage"):
            if not taxonomy.get(field):
                errors.append(f"{prefix} taxonomy.{field} is empty")
        signature = str(record.get("discriminators", {}).get("signature") or "")
        signatures[signature] += 1
        if not record.get("positive_examples") or not record.get("negative_examples"):
            errors.append(f"{prefix} positive/negative discriminators are required")

        source_record = source_by_id.get(scenario_id)
        if not source_record:
            errors.append(f"{prefix} missing source scenario")
            continue
        source_facts = list(source_record.get("facts", []))
        facts = list(record.get("facts", []))
        fact_records = list(record.get("fact_records", []))
        source_fact_count += len(source_facts)
        migrated_fact_count += len(fact_records)
        if source_facts != facts:
            errors.append(f"{prefix} approved fact text/order changed")
        if [item.get("text") for item in fact_records] != source_facts:
            errors.append(f"{prefix} fact_records do not preserve every fact")
        fact_ids = [str(item.get("fact_id") or "") for item in fact_records]
        if len(fact_ids) != len(set(fact_ids)) or any(not item for item in fact_ids):
            errors.append(f"{prefix} fact IDs are missing or duplicated")
        for fact in fact_records:
            if fact.get("status") != "approved" or not fact.get("source") or not fact.get("reviewed_at"):
                errors.append(f"{prefix} fact {fact.get('fact_id')} lacks approval provenance")
        if record.get("intent") in FINANCIAL_OR_CONTRACT_INTENTS:
            financial_fact_count += len(fact_records)
            if any(not item.get("source_version") for item in fact_records):
                errors.append(f"{prefix} financial/contract fact lacks source_version")

        policy = record.get("answer_policy", {})
        if policy.get("fact_scope") != "listed_fact_ids_only":
            errors.append(f"{prefix} answer policy does not restrict facts")
        if policy.get("llm_role") != "wording_only":
            errors.append(f"{prefix} LLM role is not wording_only")
        search_document = str(record.get("search_document") or "")
        for forbidden_text in [*facts, record.get("short_answer"), record.get("detailed_answer"), record.get("next_step")]:
            text = str(forbidden_text or "").strip()
            if len(text) >= 30 and text in search_document:
                errors.append(f"{prefix} search_document leaks fact or answer text")
                break

    duplicate_signatures = [item for item, count in signatures.items() if item and count > 1]
    if duplicate_signatures:
        errors.append(f"duplicate discriminator signatures: {len(duplicate_signatures)}")

    units = knowledge.get("atomic_units", [])
    unit_ids = [item.get("unit_id") for item in units]
    if len(unit_ids) != len(set(unit_ids)):
        errors.append("atomic unit IDs are not unique")
    units_by_scenario: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for unit in units:
        scenario_id = str(unit.get("canonical_scenario_id") or "")
        units_by_scenario[scenario_id].append(unit)
        if scenario_id not in by_id:
            errors.append(f"atomic unit {unit.get('unit_id')} has unknown canonical scenario")
        if not unit.get("search_document") or not unit.get("discriminator_terms"):
            errors.append(f"atomic unit {unit.get('unit_id')} lacks retrieval discriminators")
    for scenario_id, record in by_id.items():
        expected = {item["fact_id"] for item in record.get("fact_records", [])}
        assigned = [
            fact_id
            for unit in units_by_scenario.get(scenario_id, [])
            for fact_id in unit.get("fact_ids", [])
        ]
        if set(assigned) != expected or len(assigned) != len(set(assigned)):
            errors.append(f"{scenario_id}: atomic units do not cover facts exactly once")
        declared = set(record.get("atomic_unit_ids", []))
        actual = {item.get("unit_id") for item in units_by_scenario.get(scenario_id, [])}
        if declared != actual:
            errors.append(f"{scenario_id}: atomic_unit_ids mismatch")

    gap_ids = set()
    for gap in knowledge.get("knowledge_gaps", []):
        gap_id = str(gap.get("gap_id") or "")
        gap_ids.add(gap_id)
        if gap.get("scenario_id") not in by_id:
            errors.append(f"{gap_id}: unknown scenario")
        if gap.get("status") != "owner_confirmation_required":
            errors.append(f"{gap_id}: unexpected status")
        if not gap.get("safe_answer") or not gap.get("answer_policy"):
            errors.append(f"{gap_id}: safe gap policy is incomplete")
    if len(gap_ids) != 3:
        errors.append(f"expected 3 accepted knowledge gaps, got {len(gap_ids)}")

    conflict_pairs = set()
    for conflict in conflicts.get("records", []):
        pair = tuple(sorted((conflict.get("scenario_a"), conflict.get("scenario_b"))))
        if pair in conflict_pairs:
            errors.append(f"duplicate conflict pair: {pair}")
        conflict_pairs.add(pair)
        for scenario_id in pair:
            if scenario_id not in by_id and scenario_id not in SENTINELS:
                errors.append(f"unknown conflict scenario: {scenario_id}")
        if not conflict.get("distinctions") or not conflict.get("decision_policy"):
            errors.append(f"conflict {conflict.get('conflict_id')} lacks decision policy")

    return {
        "schema_version": 1,
        "knowledge_version": knowledge.get("version"),
        "valid": not errors,
        "errors": errors,
        "metrics": {
            "source_scenario_count": len(source_records),
            "active_scenario_count": len(records),
            "source_fact_count": source_fact_count,
            "migrated_fact_count": migrated_fact_count,
            "fact_loss_count": source_fact_count - migrated_fact_count,
            "financial_or_contract_fact_count": financial_fact_count,
            "financial_or_contract_facts_with_provenance_pct": 100.0 if financial_fact_count else 0.0,
            "atomic_unit_count": len(units),
            "conflict_count": len(conflict_pairs),
            "knowledge_gap_count": len(gap_ids),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate normalized MIGTORG knowledge schema v3.1.")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--knowledge", type=Path, default=DEFAULT_KNOWLEDGE)
    parser.add_argument("--conflicts", type=Path, default=DEFAULT_CONFLICTS)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = validate(
        json.loads(args.source.read_text(encoding="utf-8")),
        json.loads(args.knowledge.read_text(encoding="utf-8")),
        json.loads(args.conflicts.read_text(encoding="utf-8")),
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
