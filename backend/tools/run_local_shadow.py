from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

from backend.app.bot.intent_classifier import classify_intent
from backend.app.bot.knowledge_search import HybridSearchProvider, KnowledgeSearchResult, search_knowledge_match
from backend.app.bot.pii_redaction import detected_pii_kinds, redact_for_external_llm


DEFAULT_REPORT = Path("reports/stage6-local-shadow.json")
DEFAULT_DETAILS = Path(".work/stage6-local-shadow-details.json")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _records(payload: Any, source: Path) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []

    def append(item: dict[str, Any], fallback_id: str) -> None:
        text = str(item.get("text") or item.get("message") or item.get("text_redacted") or "").strip()
        if not text:
            return
        result.append({
            "id": str(item.get("id") or item.get("source_message_id") or fallback_id),
            "text": text,
            "role": str(item.get("role") or ""),
            "source": source.name,
        })

    if isinstance(payload, list):
        for index, item in enumerate(payload, start=1):
            if isinstance(item, dict):
                append(item, f"row-{index:04d}")
    elif isinstance(payload, dict):
        for index, item in enumerate(payload.get("cases", []), start=1):
            if isinstance(item, dict):
                append(item, f"case-{index:04d}")
        for group_index, group in enumerate(payload.get("groups", []), start=1):
            for query_index, item in enumerate(group.get("queries", []), start=1):
                if isinstance(item, dict):
                    append(item, f"group-{group_index:03d}-{query_index:03d}")
        for dialogue_index, dialogue in enumerate(payload.get("dialogues", []), start=1):
            for turn_index, item in enumerate(dialogue.get("turns", []), start=1):
                if isinstance(item, dict):
                    append(item, f"dialogue-{dialogue_index:03d}-{turn_index:03d}")
        for index, item in enumerate(payload.get("rows", []), start=1):
            if isinstance(item, dict):
                append(item, f"row-{index:04d}")
    return result


def load_records(paths: list[Path]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    records: list[dict[str, str]] = []
    sources = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        loaded = _records(payload, path)
        if not loaded:
            raise ValueError(f"No message records found in {path}")
        records.extend(loaded)
        sources.append({"path": str(path), "sha256": _sha256(path), "records": len(loaded)})
    if not records:
        raise ValueError("Shadow input is empty")
    return records, sources


def expand_records(records: list[dict[str, str]], target_count: int) -> list[dict[str, str]]:
    if target_count <= 0:
        return records
    return [
        {**records[index % len(records)], "event_id": f"shadow-{index + 1:06d}"}
        for index in range(target_count)
    ]


def redact_for_shadow(text: str) -> str:
    return redact_for_external_llm(text)


def pii_leaks(text: str) -> list[str]:
    return list(detected_pii_kinds(text))


def _decision(result: KnowledgeSearchResult) -> dict[str, Any]:
    article = result.article
    return {
        "scenario_id": article.slug if article else None,
        "intent": article.intent if article else None,
        "confidence": result.confidence,
        "fallback_reason": result.fallback_reason,
        "clarifies": bool(result.clarifying_question and not article),
        "needs_ticket": bool(article and article.needs_ticket),
    }


def _timed(call: Callable[[], KnowledgeSearchResult]) -> tuple[dict[str, Any] | None, float, str]:
    started = perf_counter()
    try:
        return _decision(call()), (perf_counter() - started) * 1000, ""
    except Exception as exc:  # pragma: no cover - exercised by integration failures
        return None, (perf_counter() - started) * 1000, f"{type(exc).__name__}: {exc}"


def run_shadow(records: list[dict[str, str]], default_role: str) -> list[dict[str, Any]]:
    baseline = HybridSearchProvider()
    results: list[dict[str, Any]] = []
    for index, item in enumerate(records, start=1):
        message = item["text"]
        fallback_role = (
            "authorized" if default_role == "mixed" and index % 2 == 0
            else "guest" if default_role == "mixed"
            else default_role
        )
        role = item.get("role") if item.get("role") in {"guest", "authorized"} else fallback_role
        intent = classify_intent(message)
        baseline_result, baseline_ms, baseline_error = _timed(
            lambda: baseline.search(message, intent, role)
        )
        candidate_result, candidate_ms, candidate_error = _timed(
            lambda: search_knowledge_match(message, intent, role)
        )
        redacted = redact_for_shadow(message)
        results.append({
            "event_id": item.get("event_id") or f"shadow-{index:06d}",
            "source": item["source"],
            "source_id": item["id"],
            "role": role,
            "message_redacted": redacted,
            "pii_leaks": pii_leaks(redacted),
            "baseline": baseline_result,
            "candidate": candidate_result,
            "baseline_latency_ms": round(baseline_ms, 3),
            "candidate_latency_ms": round(candidate_ms, 3),
            "baseline_error": baseline_error,
            "candidate_error": candidate_error,
        })
    return results


def _rate(part: int, total: int) -> float:
    return round(part * 100 / total, 2) if total else 0.0


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return round(ordered[max(0, math.ceil(len(ordered) * 0.95) - 1)], 3)


def summarize(results: list[dict[str, Any]], sources: list[dict[str, str]], target_count: int) -> dict[str, Any]:
    total = len(results)
    complete = [item for item in results if not item["baseline_error"] and not item["candidate_error"]]
    disagreements = [
        item for item in complete
        if item["baseline"]["scenario_id"] != item["candidate"]["scenario_id"]
    ]
    confident_disagreements = [
        item for item in disagreements if item["candidate"]["confidence"] == "high"
    ]
    candidate_fallbacks = sum(bool(item["candidate"]["fallback_reason"]) for item in complete)
    baseline_fallbacks = sum(bool(item["baseline"]["fallback_reason"]) for item in complete)
    candidate_clarifies = sum(bool(item["candidate"]["clarifies"]) for item in complete)
    baseline_clarifies = sum(bool(item["baseline"]["clarifies"]) for item in complete)
    pii_count = sum(bool(item["pii_leaks"]) for item in results)
    baseline_p95 = _p95([item["baseline_latency_ms"] for item in complete])
    candidate_p95 = _p95([item["candidate_latency_ms"] for item in complete])
    p95_delta_pct = round((candidate_p95 / baseline_p95 - 1) * 100, 2) if baseline_p95 else None
    local_checks = {
        "at_least_1000_requests": total >= 1000,
        "no_transport_errors": len(complete) == total,
        "no_pii_in_output": pii_count == 0,
    }
    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "local_shadow_plumbing_only_not_production_gate",
        "sources": sources,
        "population": {"requests": total, "requested_target": target_count},
        "roles": dict(Counter(str(item.get("role") or "unknown") for item in results)),
        "transport": {
            "complete": len(complete),
            "errors": total - len(complete),
            "success_rate_pct": _rate(len(complete), total),
        },
        "privacy": {"records_with_detected_pii_after_redaction": pii_count},
        "routing": {
            "scenario_disagreements": len(disagreements),
            "scenario_disagreement_rate_pct": _rate(len(disagreements), len(complete)),
            "candidate_high_confidence_disagreements": len(confident_disagreements),
            "candidate_high_confidence_disagreement_rate_pct": _rate(len(confident_disagreements), len(complete)),
            "baseline_fallback_rate_pct": _rate(baseline_fallbacks, len(complete)),
            "candidate_fallback_rate_pct": _rate(candidate_fallbacks, len(complete)),
            "baseline_clarification_rate_pct": _rate(baseline_clarifies, len(complete)),
            "candidate_clarification_rate_pct": _rate(candidate_clarifies, len(complete)),
            "candidate_scenarios": dict(Counter(
                str(item["candidate"]["scenario_id"] or "<none>") for item in complete
            ).most_common(20)),
        },
        "latency_ms": {
            "baseline_p95": baseline_p95,
            "candidate_p95": candidate_p95,
            "candidate_vs_baseline_p95_delta_pct": p95_delta_pct,
        },
        "local_plumbing_gate": {"passed": all(local_checks.values()), "checks": local_checks},
        "not_measured": [
            "expert answer quality on real shadow traffic",
            "confident-wrong rate without independent scenario labels",
            "critical unsupported facts without expert answer review",
            "production API error and latency SLO",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a local route-only shadow comparison.")
    parser.add_argument("sources", type=Path, nargs="+")
    parser.add_argument("--role", choices=("guest", "authorized", "mixed"), default="mixed")
    parser.add_argument("--target-count", type=int, default=0)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--details", type=Path, default=DEFAULT_DETAILS)
    args = parser.parse_args()
    records, sources = load_records(args.sources)
    expanded = expand_records(records, args.target_count)
    results = run_shadow(expanded, args.role)
    report = summarize(results, sources, args.target_count)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.details.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.details.write_text(json.dumps({"report": report, "results": results}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
