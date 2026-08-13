from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any


def _expected_label(result: dict[str, Any]) -> str:
    values = result.get("expected", {}).get("expected_scenario_ids")
    if values is None:
        return "<not_asserted>"
    return " | ".join("<clarification>" if value is None else str(value) for value in values)


def build(report: dict[str, Any]) -> dict[str, Any]:
    matrix: dict[str, Counter[str]] = defaultdict(Counter)
    conflicts: Counter[tuple[str, str]] = Counter()
    examples: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for result in report["single_turn_results"]:
        expected = _expected_label(result)
        actual = str((result.get("response") or {}).get("scenario_id") or "<clarification>")
        matrix[expected][actual] += 1
        if not result["checks"]["scenario_ok"]:
            key = (expected, actual)
            conflicts[key] += 1
            if len(examples[key]) < 3:
                examples[key].append({"id": result["id"], "text": result["text"]})
    top = [
        {
            "expected": expected,
            "actual": actual,
            "count": count,
            "examples": examples[(expected, actual)],
        }
        for (expected, actual), count in conflicts.most_common()
    ]
    return {
        "source_report": report.get("dataset", {}),
        "case_count": len(report["single_turn_results"]),
        "scenario_accuracy_pct": report["single_turn_summary"]["overall"]["scenario_ok"]["rate_pct"],
        "matrix": {expected: dict(sorted(actual.items())) for expected, actual in sorted(matrix.items())},
        "top_conflicts": top,
    }


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Routing v3: матрица перепутанных сценариев",
        "",
        f"Проверено запросов: **{payload['case_count']}**. Точность выбора сценария: **{payload['scenario_accuracy_pct']}%**.",
        "",
        "Матрица ниже строится автоматически из ответа полного API. `<clarification>` означает, что бот не выбрал один сценарий и запросил уточнение.",
        "",
        "| Ожидался сценарий | Фактически выбран | Количество | Примеры |",
        "|---|---|---:|---|",
    ]
    if not payload["top_conflicts"]:
        lines.append("| — | — | 0 | Перепутанных сценариев нет |")
    for item in payload["top_conflicts"]:
        examples = "; ".join(f"`{entry['id']}`: {entry['text']}" for entry in item["examples"])
        lines.append(f"| `{item['expected']}` | `{item['actual']}` | {item['count']} | {examples} |")
    lines.extend([
        "",
        "## Правило использования",
        "",
        "Исправляются повторяющиеся пары конфликтов на уровне признаков и reranker-профилей. Единичные формулировки не добавляются как отдельные правила маршрутизации.",
        "",
    ])
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a scenario confusion matrix from a live-query API report.")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    payload = build(report)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.markdown_output.write_text(render_markdown(payload), encoding="utf-8")
    print(json.dumps({"cases": payload["case_count"], "conflicts": len(payload["top_conflicts"])}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
