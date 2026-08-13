from __future__ import annotations

import argparse
import json
from pathlib import Path

from backend.app.bot.scenario_engine import load_scenarios, match_scenario


def evaluate(path: Path) -> dict:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    scenarios = {item.scenario_id: item for item in load_scenarios()}
    correct = 0
    confident_wrong = 0
    required_marker_checks = 0
    required_marker_hits = 0
    forbidden_marker_checks = 0
    forbidden_marker_violations = 0
    escalation_checks = 0
    escalation_correct = 0
    unnecessary_escalations = 0
    clarification_checks = 0
    clarification_guesses = 0
    details = []
    for row in rows:
        expected = row.get("expected_scenario_id")
        expected_scenario = scenarios.get(str(expected or ""))
        inferred_role = (
            "guest"
            if not expected_scenario or "guest" in expected_scenario.roles
            else "authorized"
        )
        decision = match_scenario(str(row["message"]), str(row.get("role") or inferred_role))
        predicted = decision.scenario.scenario_id if decision.scenario else None
        is_correct = predicted == expected
        correct += int(is_correct)
        confident_wrong += int(decision.confidence == "high" and not is_correct)
        answer = decision.scenario.answer.casefold() if decision.scenario else ""
        required_markers = [str(item).casefold() for item in row.get("required_answer_markers", [])]
        forbidden_markers = [str(item).casefold() for item in row.get("forbidden_answer_markers", [])]
        required_marker_checks += len(required_markers)
        required_marker_hits += sum(marker in answer for marker in required_markers)
        forbidden_marker_checks += len(forbidden_markers)
        forbidden_marker_violations += sum(marker in answer for marker in forbidden_markers)
        if "expected_escalation" in row:
            escalation_checks += 1
            actual_escalation = bool(
                decision.scenario
                and any(str(action.get("type") or "") == "open_ticket" for action in decision.scenario.actions)
            )
            expected_escalation = bool(row["expected_escalation"])
            escalation_correct += int(actual_escalation == expected_escalation)
            unnecessary_escalations += int(actual_escalation and not expected_escalation)
        if bool(row.get("expected_clarification", False)):
            clarification_checks += 1
            scenario_actions = list(decision.scenario.actions) if decision.scenario else []
            clarification_scenario = bool(
                scenario_actions
                and all(str(action.get("type") or "") == "clarify" for action in scenario_actions)
            )
            actual_clarification = bool(
                decision.scenario is None
                or decision.clarifying_question
                or clarification_scenario
            )
            clarification_guesses += int(not actual_clarification)
        if not is_correct:
            details.append(
                {
                    "message": row["message"],
                    "expected": expected,
                    "predicted": predicted,
                    "confidence": decision.confidence,
                    "score": decision.score,
                }
            )
    total = len(rows)
    scenario_accuracy = correct / total if total else 0.0
    confident_wrong_rate = confident_wrong / total if total else 0.0
    required_facts_coverage = required_marker_hits / required_marker_checks if required_marker_checks else None
    forbidden_promises_rate = forbidden_marker_violations / forbidden_marker_checks if forbidden_marker_checks else None
    escalation_accuracy = escalation_correct / escalation_checks if escalation_checks else None
    unnecessary_escalation_rate = unnecessary_escalations / escalation_checks if escalation_checks else None
    clarification_guess_rate = clarification_guesses / clarification_checks if clarification_checks else None
    technical_gate_passed = bool(total and scenario_accuracy >= 0.85 and confident_wrong_rate <= 0.02)
    production_gate_passed = bool(
        total >= 300
        and scenario_accuracy >= 0.90
        and confident_wrong_rate <= 0.01
        and required_facts_coverage is not None
        and required_facts_coverage >= 0.95
        and forbidden_promises_rate is not None
        and forbidden_promises_rate == 0
        and escalation_accuracy is not None
        and escalation_accuracy >= 0.95
        and unnecessary_escalation_rate is not None
        and unnecessary_escalation_rate <= 0.05
        and clarification_guess_rate is not None
        and clarification_guess_rate <= 0.05
    )
    return {
        "total": total,
        "scenario_accuracy": round(scenario_accuracy, 4),
        "confident_wrong_rate": round(confident_wrong_rate, 4),
        "required_facts_coverage": round(required_facts_coverage, 4) if required_facts_coverage is not None else None,
        "forbidden_promises_rate": round(forbidden_promises_rate, 4) if forbidden_promises_rate is not None else None,
        "escalation_accuracy": round(escalation_accuracy, 4) if escalation_accuracy is not None else None,
        "unnecessary_escalation_rate": round(unnecessary_escalation_rate, 4) if unnecessary_escalation_rate is not None else None,
        "clarification_guess_rate": round(clarification_guess_rate, 4) if clarification_guess_rate is not None else None,
        "release_gate_passed": technical_gate_passed,
        "production_gate_passed": production_gate_passed,
        "errors": details,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate knowledge-v2 scenario routing.")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--gate", action="store_true")
    parser.add_argument("--production-gate", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = evaluate(args.dataset)
    serialized = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)
    if args.gate and not report["release_gate_passed"]:
        raise SystemExit(1)
    if args.production_gate and not report["production_gate_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
