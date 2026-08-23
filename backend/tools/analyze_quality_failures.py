from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from backend.app.bot.routing_v3 import get_routing_v3
from backend.app.bot.scenario_engine import load_scenarios


DEFAULT_REPORT = Path("reports/routing-v3-closed-control-270.json")
DEFAULT_JSON = Path("reports/quality-stage0-error-taxonomy.json")
DEFAULT_MARKDOWN = Path("docs/quality-stage0-error-taxonomy.md")

CAUSES = {
    "transport_error": "Запрос не был успешно обработан API.",
    "retrieval_miss": "Ни один допустимый сценарий не попал в Top-10 candidate retrieval.",
    "rerank_wrong": "Допустимый сценарий был среди кандидатов, но финально выбран соседний сценарий.",
    "wrong_clarify": "На однозначный запрос дано уточнение/fallback либо неоднозначность обработана неверно.",
    "missing_kb": "Ожидаемый сценарий отсутствует в активной БЗ.",
    "wrong_facts": "Маршрут допустим, но обязательные факты ответа отсутствуют.",
    "irrelevant_answer": "Маршрут допустим, но resolution, полнота, краткость или формулировка ответа не прошли критерии.",
    "unsafe_answer": "Ответ содержит запрещённый или неподтверждённый фрагмент.",
    "dialogue_state": "Ошибка возникла при продолжении диалога или обработке выбранного уточнения.",
}


def _candidate_ids(text: str, limit: int = 10) -> list[str]:
    return [
        candidate.scenario.scenario_id
        for candidate in get_routing_v3().rank(text, role="guest", top_k=limit)
    ]


def classify_failure(result: dict[str, Any], active_ids: set[str]) -> dict[str, Any]:
    checks = result.get("checks", {})
    expected = result.get("expected", {})
    expected_ids = [item for item in expected.get("expected_scenario_ids", []) if item is not None]
    response = result.get("response") or {}
    actual_id = response.get("scenario_id")
    candidates = _candidate_ids(str(result.get("text") or ""))
    correct_in_top10 = bool(set(expected_ids).intersection(candidates))
    unknown_expected = sorted(set(expected_ids) - active_ids)
    routing_subcause: str | None = None

    if not checks.get("transport_ok"):
        cause = "transport_error"
    elif result.get("class") == "dialogue_turn":
        cause = "dialogue_state"
    elif unknown_expected:
        cause = "missing_kb"
    elif not checks.get("route_hit"):
        routing_subcause = "rerank_wrong" if correct_in_top10 else "retrieval_miss"
        if actual_id is None and expected_ids:
            cause = "wrong_clarify"
        else:
            cause = routing_subcause
    elif not checks.get("forbidden_content_ok"):
        cause = "unsafe_answer"
    elif not checks.get("required_facts_ok"):
        cause = "wrong_facts"
    elif not checks.get("direct_ok") or response.get("resolution") == "clarified":
        cause = "wrong_clarify"
    else:
        cause = "irrelevant_answer"

    return {
        "id": result.get("id"),
        "source": "widget-110" if str(result.get("id", "")).startswith("widget-") else "independent-160",
        "group": result.get("group"),
        "class": result.get("class"),
        "text": result.get("text"),
        "expected_scenario_ids": expected.get("expected_scenario_ids", []),
        "expected_intents": expected.get("expected_intents", []),
        "actual_scenario_id": actual_id,
        "actual_intent": response.get("intent"),
        "actual_resolution": response.get("resolution"),
        "confidence_level": response.get("confidence_level"),
        "primary_cause": cause,
        "routing_subcause": routing_subcause,
        "correct_candidate_in_top10": correct_in_top10,
        "top10_candidate_ids": candidates,
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "diagnostics": result.get("diagnostics", {}),
    }


def build(report_path: Path) -> dict[str, Any]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    failed = [item for item in report.get("single_turn_results", []) if not item["checks"]["quality_pass"]]
    active_ids = {scenario.scenario_id for scenario in load_scenarios()}
    cases = [classify_failure(item, active_ids) for item in failed]
    primary = Counter(item["primary_cause"] for item in cases)
    routing = Counter(
        item["routing_subcause"]
        for item in cases
        if item.get("routing_subcause")
    )
    by_group = Counter(item["group"] for item in cases)
    confident_wrong = sum(
        item["confidence_level"] == "high"
        and item["primary_cause"] in {"retrieval_miss", "rerank_wrong"}
        for item in cases
    )
    confident_wrong_total = sum(
        item["confidence_level"] == "high"
        and "route_hit" in item["failed_checks"]
        for item in cases
    )
    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_report": str(report_path),
        "source_dataset_sha256": report.get("dataset", {}).get("sha256"),
        "taxonomy": CAUSES,
        "summary": {
            "failed_case_count": len(cases),
            "classified_case_count": len(cases),
            "by_primary_cause": dict(sorted(primary.items())),
            "routing_subcauses": dict(sorted(routing.items())),
            "by_group": dict(sorted(by_group.items())),
            "confident_wrong_primary_route_count": confident_wrong,
            "confident_wrong_total": confident_wrong_total,
        },
        "cases": cases,
    }


def render_markdown(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    lines = [
        "# Этап 0: таксономия ошибок closed-control",
        "",
        f"Источник: `{payload['source_report']}`.",
        f"Размечено: **{summary['classified_case_count']} из {summary['failed_case_count']}** провалов.",
        "",
        "## Первичные причины",
        "",
        "| Причина | Количество |",
        "|---|---:|",
    ]
    for name, count in summary["by_primary_cause"].items():
        lines.append(f"| `{name}` | {count} |")
    lines.extend([
        "",
        "## Потери routing-контура",
        "",
        "| Подпричина | Количество |",
        "|---|---:|",
    ])
    for name, count in summary["routing_subcauses"].items():
        lines.append(f"| `{name}` | {count} |")
    lines.extend([
        "",
        "`routing_subcause` дополнительно ставится ошибочному уточнению, чтобы отличить отсутствие правильного сценария в Top-10 от ошибки выбора/порога.",
        "",
        "## Все ошибки",
        "",
        "| ID | Группа | Причина | Ожидалось | Получено | Top-10 содержит правильный |",
        "|---|---|---|---|---|---|",
    ])
    for item in payload["cases"]:
        expected = " / ".join("<clarification>" if value is None else str(value) for value in item["expected_scenario_ids"])
        actual = item["actual_scenario_id"] or "<clarification>"
        lines.append(
            f"| `{item['id']}` | `{item['group']}` | `{item['primary_cause']}` | "
            f"`{expected}` | `{actual}` | {'да' if item['correct_candidate_in_top10'] else 'нет'} |"
        )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Classify every failed quality case by its primary root cause.")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--output", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = build(args.report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.markdown.write_text(render_markdown(payload), encoding="utf-8")
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    return 0 if payload["summary"]["classified_case_count"] == payload["summary"]["failed_case_count"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
