from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Callable

from backend.app.bot.answer_generator import GeneratedAnswer, generate_answer
from backend.app.bot.intent_classifier import classify_intent
from backend.app.bot.knowledge_search import KnowledgeSearchResult, search_knowledge_match
from backend.app.bot.pii_redaction import detected_pii_kinds, redact_for_external_llm
from backend.app.config import Settings, get_settings
from backend.tools.run_local_shadow import expand_records, load_records


DEFAULT_REPORT = Path("reports/stage6_1-llm-shadow.json")
DEFAULT_DETAILS = Path(".work/stage6_1-llm-shadow-review.json")


def run_llm_shadow(
    records: list[dict[str, str]],
    settings: Settings,
    *,
    router: Callable[[str, str, str], KnowledgeSearchResult] = search_knowledge_match,
    generator: Callable[..., GeneratedAnswer] = generate_answer,
) -> list[dict[str, Any]]:
    deterministic = settings.model_copy(update={"llm_enabled": False})
    results: list[dict[str, Any]] = []
    for index, item in enumerate(records, start=1):
        message = item["text"]
        role = item.get("role") if item.get("role") in {"guest", "authorized"} else (
            "authorized" if index % 2 == 0 else "guest"
        )
        intent = classify_intent(message)
        decision = router(message, intent, role)
        article = decision.article
        baseline = generator(
            message, intent, role, article, bool(article and article.needs_ticket),
            settings=deterministic, route_confidence=decision.confidence,
        )
        candidate = generator(
            message, intent, role, article, bool(article and article.needs_ticket),
            settings=settings, route_confidence=decision.confidence,
            llm_allowed=True,
        )
        llm_result = candidate.llm_result
        safe_candidate = redact_for_external_llm(candidate.llm_candidate)
        safe_answer = redact_for_external_llm(candidate.answer)
        results.append({
            "event_id": item.get("event_id") or f"llm-shadow-{index:04d}",
            "source": item["source"],
            "source_id": item["id"],
            "role": role,
            "message_redacted": redact_for_external_llm(message),
            "scenario_id": article.scenario if article else None,
            "route_confidence": decision.confidence,
            "baseline_answer": redact_for_external_llm(baseline.answer),
            "llm_candidate": safe_candidate,
            "effective_answer": safe_answer,
            "llm_invoked": llm_result is not None,
            "provider_success": bool(llm_result and llm_result.success),
            "provider": llm_result.provider if llm_result else None,
            "model": llm_result.model if llm_result else None,
            "latency_ms": llm_result.latency_ms if llm_result else 0,
            "estimated_cost_usd": llm_result.estimated_cost_usd if llm_result else 0.0,
            "verification_passed": candidate.verification_passed,
            "verification_reason": candidate.verification_reason,
            "fallback_used": bool(llm_result and llm_result.fallback_used),
            "detected_pii": list(detected_pii_kinds(f"{safe_candidate} {safe_answer}")),
            "expert_review": {
                "correctness": None,
                "clarity_vs_baseline": None,
                "relevance_vs_baseline": None,
                "unsupported_fact": None,
                "comment": ""
            }
        })
    return results


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    invoked = [item for item in results if item["llm_invoked"]]
    successful = [item for item in invoked if item["provider_success"]]
    accepted = [item for item in successful if item["verification_passed"]]
    rejected = [item for item in successful if not item["verification_passed"]]
    pii = [item for item in results if item["detected_pii"]]
    latencies = sorted(int(item["latency_ms"]) for item in invoked)
    p95_index = max(0, (len(latencies) * 95 + 99) // 100 - 1)
    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "expert_review_required",
        "population": {"requests": total},
        "llm": {
            "invoked": len(invoked),
            "provider_success": len(successful),
            "verifier_accepted": len(accepted),
            "verifier_rejected": len(rejected),
            "fallback_used": sum(bool(item["fallback_used"]) for item in results),
            "estimated_cost_usd": round(sum(float(item["estimated_cost_usd"]) for item in invoked), 6),
            "p95_latency_ms": latencies[p95_index] if latencies else 0,
            "verification_reasons": dict(Counter(str(item["verification_reason"]) for item in results).most_common()),
        },
        "privacy": {"records_with_detected_pii": len(pii)},
        "expert_review": {
            "completed": 0,
            "required": len(accepted),
            "gate_passed": False,
        },
        "not_measured_until_review": [
            "correctness versus deterministic baseline",
            "clarity and relevance improvement",
            "critical unsupported facts",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a redacted wording-only LLM shadow review pack.")
    parser.add_argument("sources", type=Path, nargs="+")
    parser.add_argument("--target-count", type=int, default=200)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--details", type=Path, default=DEFAULT_DETAILS)
    args = parser.parse_args()
    if not 200 <= args.target_count <= 500:
        raise ValueError("LLM shadow target-count must be between 200 and 500")
    settings = get_settings()
    if not settings.llm_enabled:
        raise ValueError("LLM_ENABLED=true is required for a real LLM shadow run")
    records, sources = load_records(args.sources)
    results = run_llm_shadow(expand_records(records, args.target_count), settings)
    report = summarize(results)
    report["sources"] = sources
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.details.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.details.write_text(json.dumps({"report": report, "results": results}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
