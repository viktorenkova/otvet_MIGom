from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from backend.app.bot.text_processing import normalize_matching_text


DEFAULT_SCENARIOS = Path("knowledge/v3_1/scenarios.json")
DEFAULT_CONFLICTS = Path("knowledge/v3_1/scenario_conflicts.json")
DEFAULT_OUTPUT = Path("knowledge/v3_1/answer_contracts.json")
STOPWORDS = {
    "более", "будет", "быть", "вашего", "вашей", "вашим", "ваших", "если",
    "когда", "который", "можно", "нужно", "после", "перед", "также", "только",
    "через", "чтобы", "этого", "этот",
}


def _tokens(text: str) -> set[str]:
    return {
        token for token in normalize_matching_text(text).split()
        if len(token) >= 4 and token not in STOPWORDS
    }


def _template_kind(record: dict[str, Any]) -> str:
    actions = [str(item.get("type") or "") for item in record.get("actions", [])]
    if actions and all(action == "clarify" for action in actions):
        return "clarification"
    if "open_ticket" in actions or record.get("escalation", {}).get("when"):
        return "contact"
    if "check" in record.get("operations", []) and record.get("states"):
        return "status"
    return "direct"


def build(scenarios_path: Path, conflicts_path: Path) -> dict[str, Any]:
    source = json.loads(scenarios_path.read_text(encoding="utf-8"))
    records = [row for row in source["records"] if str(row.get("status") or "active") == "active"]
    by_id = {str(row["scenario_id"]): row for row in records}
    neighbors: dict[str, set[str]] = {scenario_id: set() for scenario_id in by_id}
    conflicts = json.loads(conflicts_path.read_text(encoding="utf-8"))
    for conflict in conflicts.get("records", []):
        left = str(conflict.get("scenario_a") or "")
        right = str(conflict.get("scenario_b") or "")
        if left in by_id and right in by_id:
            neighbors[left].add(right)
            neighbors[right].add(left)

    contracts = []
    for scenario_id, record in sorted(by_id.items()):
        facts = [dict(item) for item in record.get("fact_records", [])]
        answer = " ".join(
            part.strip()
            for part in (
                str(record.get("short_answer") or ""),
                str(record.get("detailed_answer") or ""),
                str(record.get("next_step") or ""),
            )
            if part.strip()
        )
        answer_tokens = _tokens(answer)
        required = []
        for fact in facts:
            fact_tokens = _tokens(str(fact.get("text") or ""))
            coverage = len(fact_tokens & answer_tokens) / max(1, len(fact_tokens))
            if coverage >= 0.6:
                required.append(str(fact["fact_id"]))
        if not required and facts:
            required.append(str(facts[0]["fact_id"]))
        allowed = [str(fact["fact_id"]) for fact in facts]
        forbidden = sorted({
            str(fact["fact_id"])
            for neighbor in neighbors[scenario_id]
            for fact in by_id[neighbor].get("fact_records", [])
        })
        contracts.append({
            "scenario_id": scenario_id,
            "template_kind": _template_kind(record),
            "approved_template": answer,
            "required_fact_ids": required,
            "allowed_fact_ids": allowed,
            "forbidden_fact_ids": forbidden,
            "facts": {str(fact["fact_id"]): str(fact["text"]) for fact in facts},
            "llm_role": "wording_only",
            "verification_failure": "deterministic_approved_template",
        })
    return {
        "schema_version": 1,
        "knowledge_version": str(source.get("knowledge_version") or ""),
        "template_kinds": {
            "direct": "Короткий прямой ответ по фактам выбранного сценария.",
            "clarification": "Предметный вопрос только о недостающем признаке.",
            "status": "Известный порядок проверки без утверждения непроверенного статуса.",
            "contact": "Известный порядок и следующий шаг для официального обращения.",
        },
        "record_count": len(contracts),
        "records": contracts,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenarios", type=Path, default=DEFAULT_SCENARIOS)
    parser.add_argument("--conflicts", type=Path, default=DEFAULT_CONFLICTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = build(args.scenarios, args.conflicts)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"record_count": result["record_count"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
