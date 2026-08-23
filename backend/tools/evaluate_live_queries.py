from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import tempfile
from time import perf_counter, sleep
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_DATASET = Path("tests/data/live_query_audit_2026_08_13.json")
DEFAULT_ENDPOINT = "https://chat.migtorg.com/api/chat/message"
DEFAULT_RELEASE_GATES = {
    "transport_ok_pct": 100.0,
    "route_hit_pct": 85.0,
    "answer_hit_pct": 80.0,
    "typo_answer_hit_pct": 80.0,
    "transposed_letters_answer_hit_pct": 75.0,
    "morphology_answer_hit_pct": 85.0,
    "no_response_answer_hit_pct": 90.0,
    "ambiguous_answer_hit_pct": 90.0,
    "dialogue_completion_pct": 80.0,
    "forbidden_content_ok_pct": 100.0,
    "confident_wrong_max": 2,
}
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
SPACE_RE = re.compile(r"\s+")


def _merge_expectations(group: dict[str, Any], query: dict[str, Any]) -> dict[str, Any]:
    inherited_keys = {
        "expected_scenario_ids",
        "expected_intents",
        "allowed_resolutions",
        "required_any_groups",
        "forbidden_answer_fragments",
        "expect_direct",
    }
    expected = {key: group[key] for key in inherited_keys if key in group}
    expected.update({key: value for key, value in query.items() if key in inherited_keys})
    return expected


def load_dataset(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    raw = path.read_bytes()
    dataset = json.loads(raw.decode("utf-8"))
    cases: list[dict[str, Any]] = []
    if "cases" in dataset:
        for case in dataset["cases"]:
            cases.append(
                {
                    "id": case["id"],
                    "group": case.get("group", case.get("source", "control")),
                    "class": case.get("class", "natural"),
                    "text": case["text"],
                    "expected": dict(case.get("expected", {})),
                }
            )
    else:
        for group in dataset["groups"]:
            for query in group["queries"]:
                cases.append(
                    {
                        "id": query["id"],
                        "group": group["name"],
                        "class": query["class"],
                        "text": query["text"],
                        "expected": _merge_expectations(group, query),
                    }
                )
    dataset["sha256"] = hashlib.sha256(raw).hexdigest()
    return dataset, cases


def _normalize_sentence(value: str) -> str:
    return SPACE_RE.sub(" ", re.sub(r"[^0-9a-zа-яё ]+", " ", value.casefold())).strip()


def duplicate_sentences(answer: str) -> list[str]:
    normalized = [_normalize_sentence(item) for item in SENTENCE_SPLIT_RE.split(answer)]
    eligible = [item for item in normalized if len(item.split()) >= 4]
    counts = Counter(eligible)
    return [sentence for sentence, count in counts.items() if count > 1]


def _matches_any(value: str | None, accepted: list[Any] | None) -> bool:
    if accepted is None:
        return True
    return value in accepted


def evaluate_response(
    case: dict[str, Any],
    response: dict[str, Any] | None,
    latency_ms: float,
    error: str = "",
    global_forbidden: list[str] | None = None,
) -> dict[str, Any]:
    expected = case.get("expected", {})
    answer = str((response or {}).get("answer") or "")
    answer_folded = answer.casefold()
    expected_scenarios = expected.get("expected_scenario_ids")
    expected_intents = expected.get("expected_intents")
    expected_resolutions = expected.get("allowed_resolutions")
    required_groups = expected.get("required_any_groups", [])
    forbidden = [*(global_forbidden or []), *expected.get("forbidden_answer_fragments", [])]
    missing_required_groups = [
        group
        for group in required_groups
        if not any(str(fragment).casefold() in answer_folded for fragment in group)
    ]
    forbidden_hits = [fragment for fragment in forbidden if fragment.casefold() in answer_folded]
    duplicate_hits = duplicate_sentences(answer)
    scenario_ok = _matches_any((response or {}).get("scenario_id"), expected_scenarios)
    intent_ok = _matches_any((response or {}).get("intent"), expected_intents)
    resolution_ok = _matches_any((response or {}).get("resolution"), expected_resolutions)
    required_ok = not missing_required_groups
    forbidden_ok = not forbidden_hits
    duplicate_ok = not duplicate_hits
    concise_ok = len(answer) <= 900
    direct_ok = not expected.get("expect_direct") or (response or {}).get("resolution") != "clarified"
    transport_ok = response is not None and not error
    route_hit = transport_ok and scenario_ok and intent_ok
    answer_hit = (
        transport_ok
        and scenario_ok
        and intent_ok
        and resolution_ok
        and required_ok
        and forbidden_ok
        and direct_ok
    )
    quality_pass = answer_hit and duplicate_ok and concise_ok
    return {
        **case,
        "response": response,
        "latency_ms": round(latency_ms, 2),
        "error": error,
        "checks": {
            "transport_ok": transport_ok,
            "scenario_ok": scenario_ok,
            "intent_ok": intent_ok,
            "resolution_ok": resolution_ok,
            "required_facts_ok": required_ok,
            "forbidden_content_ok": forbidden_ok,
            "direct_ok": direct_ok,
            "no_duplicate_sentences": duplicate_ok,
            "concise_ok": concise_ok,
            "route_hit": route_hit,
            "answer_hit": answer_hit,
            "quality_pass": quality_pass,
        },
        "diagnostics": {
            "missing_required_groups": missing_required_groups,
            "forbidden_hits": forbidden_hits,
            "duplicate_sentences": duplicate_hits,
            "answer_length": len(answer),
        },
    }


def _percent(numerator: int, denominator: int) -> float:
    return round((numerator / denominator * 100.0) if denominator else 0.0, 2)


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = (len(ordered) - 1) * percentile
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    if lower == upper:
        return round(ordered[lower], 2)
    fraction = index - lower
    return round(ordered[lower] + (ordered[upper] - ordered[lower]) * fraction, 2)


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    check_names = [
        "transport_ok",
        "scenario_ok",
        "intent_ok",
        "resolution_ok",
        "required_facts_ok",
        "forbidden_content_ok",
        "direct_ok",
        "no_duplicate_sentences",
        "concise_ok",
        "route_hit",
        "answer_hit",
        "quality_pass",
    ]

    def subset_metrics(items: list[dict[str, Any]]) -> dict[str, Any]:
        total = len(items)
        metrics: dict[str, Any] = {"total": total}
        for check in check_names:
            passed = sum(bool(item["checks"].get(check)) for item in items)
            metrics[check] = {"passed": passed, "rate_pct": _percent(passed, total)}
        return metrics

    by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        by_class[result.get("class", "unknown")].append(result)
        by_group[result.get("group", "unknown")].append(result)
        response = result.get("response") or {}
        by_model[str(response.get("model_used") or "unknown")].append(result)
    latencies = [result["latency_ms"] for result in results if result["checks"]["transport_ok"]]
    failed = [result for result in results if not result["checks"]["quality_pass"]]
    return {
        "overall": subset_metrics(results),
        "by_class": {name: subset_metrics(items) for name, items in sorted(by_class.items())},
        "by_group": {name: subset_metrics(items) for name, items in sorted(by_group.items())},
        "by_model": {name: subset_metrics(items) for name, items in sorted(by_model.items())},
        "model_distribution": dict(sorted(Counter(
            str((result.get("response") or {}).get("model_used") or "unknown") for result in results
        ).items())),
        "resolution_distribution": dict(sorted(Counter(
            str((result.get("response") or {}).get("resolution") or "error") for result in results
        ).items())),
        "latency_ms": {
            "mean": round(statistics.fmean(latencies), 2) if latencies else 0.0,
            "p50": _percentile(latencies, 0.50),
            "p95": _percentile(latencies, 0.95),
            "max": round(max(latencies), 2) if latencies else 0.0,
        },
        "failed_case_ids": [item["id"] for item in failed],
    }


def summarize_dialogue_completion(results: list[dict[str, Any]]) -> dict[str, Any]:
    by_dialogue: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        by_dialogue[str(result.get("dialogue_id") or "unknown")].append(result)
    completed = {
        dialogue_id: all(turn["checks"]["quality_pass"] for turn in turns)
        for dialogue_id, turns in by_dialogue.items()
    }
    passed = sum(completed.values())
    return {
        "total": len(completed),
        "passed": passed,
        "rate_pct": _percent(passed, len(completed)),
        "by_dialogue": completed,
    }


def build_local_sender(database_path: Path) -> Callable[[dict[str, Any]], dict[str, Any]]:
    os.environ["DATABASE_PATH"] = str(database_path)
    os.environ["LLM_ENABLED"] = "false"
    os.environ["LLM_PROVIDER"] = "mock"
    os.environ["LLM_PRIMARY_MODEL"] = "mock/safe-rules"
    os.environ["LLM_FALLBACK_MODEL"] = "mock/safe-rules"
    from backend.app.config import get_settings

    get_settings.cache_clear()
    from backend.app import main as runtime
    from fastapi.testclient import TestClient

    runtime.settings.llm_enabled = False
    runtime.settings.llm_provider = "mock"
    runtime.settings.llm_primary_model = "mock/safe-rules"
    runtime.settings.llm_fallback_model = "mock/safe-rules"
    runtime.warm_knowledge_indexes()
    client = TestClient(runtime.app, raise_server_exceptions=True)

    def send(payload: dict[str, Any]) -> dict[str, Any]:
        response = client.post("/api/chat/message", json=payload)
        response.raise_for_status()
        return response.json()

    return send


def evaluate_release_gates(
    results: list[dict[str, Any]],
    dialogue_completion: dict[str, Any],
    gates: dict[str, float | int] | None = None,
) -> dict[str, Any]:
    thresholds = dict(DEFAULT_RELEASE_GATES)
    thresholds.update(gates or {})
    summary = summarize(results)
    overall = summary["overall"]
    by_class = summary["by_class"]

    def class_rate(name: str) -> float:
        return float(by_class.get(name, {}).get("answer_hit", {}).get("rate_pct", 0.0))

    def combined_group_rate(names: set[str]) -> float:
        selected = [item for item in results if item.get("group") in names]
        passed = sum(item["checks"]["answer_hit"] for item in selected)
        return _percent(passed, len(selected))

    confident_wrong = sum(
        not item["checks"]["route_hit"]
        and str((item.get("response") or {}).get("confidence_level")) == "high"
        for item in results
    )
    observed = {
        "transport_ok_pct": overall["transport_ok"]["rate_pct"],
        "route_hit_pct": overall["route_hit"]["rate_pct"],
        "answer_hit_pct": overall["answer_hit"]["rate_pct"],
        "typo_answer_hit_pct": class_rate("typo"),
        "transposed_letters_answer_hit_pct": class_rate("transposed_letters"),
        "morphology_answer_hit_pct": class_rate("morphology"),
        "no_response_answer_hit_pct": combined_group_rate({"insurer_no_response", "email_no_response"}),
        "ambiguous_answer_hit_pct": combined_group_rate({"refund_ambiguous", "office_ambiguous"}),
        "dialogue_completion_pct": dialogue_completion["rate_pct"],
        "forbidden_content_ok_pct": overall["forbidden_content_ok"]["rate_pct"],
        "confident_wrong_max": confident_wrong,
    }
    checks = {
        name: (observed[name] <= limit if name.endswith("_max") else observed[name] >= limit)
        for name, limit in thresholds.items()
    }
    return {
        "passed": all(checks.values()),
        "thresholds": thresholds,
        "observed": observed,
        "checks": checks,
        "failed": [name for name, passed in checks.items() if not passed],
    }


def build_remote_sender(endpoint: str, timeout_seconds: float) -> Callable[[dict[str, Any]], dict[str, Any]]:
    def send(payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        last_error: Exception | None = None
        for attempt in range(3):
            request = Request(
                endpoint,
                data=body,
                method="POST",
                headers={
                    "Content-Type": "application/json; charset=utf-8",
                    "Accept": "application/json",
                    "User-Agent": "MIGTORG-KB-Audit/2026-08-13",
                },
            )
            try:
                with urlopen(request, timeout=timeout_seconds) as response:
                    return json.loads(response.read().decode("utf-8"))
            except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
                last_error = exc
                if attempt < 2:
                    sleep(0.4 * (attempt + 1))
        raise RuntimeError(f"remote request failed after retries: {last_error}")

    return send


def _health_endpoint_candidates(endpoint: str, explicit_endpoint: str | None = None) -> list[str]:
    if explicit_endpoint:
        return [explicit_endpoint]
    origin = endpoint.split("/api/", 1)[0].rstrip("/")
    return [f"{origin}/api/health", f"{origin}/health"]


def fetch_remote_manifest(
    endpoint: str,
    timeout_seconds: float,
    health_endpoint: str | None = None,
) -> dict[str, Any] | None:
    for candidate in _health_endpoint_candidates(endpoint, health_endpoint):
        request = Request(
            candidate,
            method="GET",
            headers={"Accept": "application/json", "User-Agent": "MIGTORG-KB-Audit/manifest-v1"},
        )
        try:
            with urlopen(request, timeout=timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError):
            continue
        manifest = payload.get("build_manifest")
        if isinstance(manifest, dict):
            return dict(manifest)
    return None


def run_case(
    case: dict[str, Any],
    sender: Callable[[dict[str, Any]], dict[str, Any]],
    run_id: str,
    global_forbidden: list[str],
) -> dict[str, Any]:
    payload = {
        "message": case["text"],
        "session_id": f"kb-audit-{run_id}-{case['id']}",
        "context": {"page_type": "public_site"},
    }
    started = perf_counter()
    try:
        response = sender(payload)
        error = ""
    except Exception as exc:  # The error is part of the audit result.
        response = None
        error = f"{type(exc).__name__}: {exc}"
    latency_ms = (perf_counter() - started) * 1000.0
    return evaluate_response(case, response, latency_ms, error, global_forbidden)


def _select_contains(items: list[Any], needle: str, key: str | None = None) -> Any:
    needle_folded = needle.casefold()
    for item in items:
        value = item.get(key, "") if key and isinstance(item, dict) else item
        if needle_folded in str(value).casefold():
            return item
    raise LookupError(f"option containing {needle!r} not found in {items!r}")


def run_dialogues(
    dialogues: list[dict[str, Any]],
    sender: Callable[[dict[str, Any]], dict[str, Any]],
    run_id: str,
    global_forbidden: list[str],
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for dialogue in dialogues:
        session_id = f"kb-audit-{run_id}-{dialogue['id']}"
        previous_response: dict[str, Any] = {}
        for turn_index, turn in enumerate(dialogue["turns"], start=1):
            case = {
                "id": f"{dialogue['id']}-t{turn_index}",
                "dialogue_id": dialogue["id"],
                "turn_index": turn_index,
                "group": "dialogue",
                "class": "dialogue_turn",
                "text": str(turn.get("text") or ""),
                "expected": {key: value for key, value in turn.items() if key not in {"text", "select_option_contains", "select_action_contains"}},
            }
            payload: dict[str, Any] = {
                "message": case["text"],
                "session_id": session_id,
                "context": {"page_type": "public_site"},
            }
            setup_error = ""
            try:
                if turn.get("select_option_contains"):
                    selected = _select_contains(
                        list(previous_response.get("clarifying_options") or []),
                        str(turn["select_option_contains"]),
                    )
                    payload["message"] = str(selected)
                    case["text"] = str(selected)
                if turn.get("select_action_contains"):
                    selected_action = _select_contains(
                        list(previous_response.get("actions") or []),
                        str(turn["select_action_contains"]),
                        "label",
                    )
                    payload["message"] = str(selected_action.get("label") or "Выбрать")
                    payload["selected_action_id"] = selected_action["id"]
                    case["text"] = payload["message"]
            except LookupError as exc:
                setup_error = str(exc)
            started = perf_counter()
            if setup_error:
                response = None
                error = setup_error
            else:
                try:
                    response = sender(payload)
                    error = ""
                except Exception as exc:  # The error is part of the audit result.
                    response = None
                    error = f"{type(exc).__name__}: {exc}"
            latency_ms = (perf_counter() - started) * 1000.0
            evaluated = evaluate_response(case, response, latency_ms, error, global_forbidden)
            results.append(evaluated)
            previous_response = response or {}
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the independent MIGTORG live-query audit set.")
    parser.add_argument("--mode", choices=("local", "remote"), required=True)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument(
        "--health-endpoint",
        help="Optional manifest endpoint; by default /api/health then /health are tried.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout-seconds", type=float, default=40.0)
    parser.add_argument(
        "--production-gate",
        action="store_true",
        help="Exit with code 2 unless every pre-approved release metric passes.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset, cases = load_dataset(args.dataset)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    global_forbidden = list(dataset.get("global_forbidden_answer_fragments", []))

    with tempfile.TemporaryDirectory(prefix="live-query-runtime-", dir=args.output.parent) as temp_dir:
        if args.mode == "local":
            sender = build_local_sender(Path(temp_dir) / "audit.sqlite3")
            from backend.app.build_manifest import build_runtime_manifest
            from backend.app.config import get_settings

            runtime_manifest = build_runtime_manifest(get_settings())
        else:
            sender = build_remote_sender(args.endpoint, args.timeout_seconds)
            runtime_manifest = fetch_remote_manifest(
                args.endpoint,
                args.timeout_seconds,
                args.health_endpoint,
            )

        results: list[dict[str, Any]] = []
        workers = 1 if args.mode == "local" else max(1, args.workers)
        if workers == 1:
            results = [run_case(case, sender, run_id, global_forbidden) for case in cases]
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {
                    pool.submit(run_case, case, sender, run_id, global_forbidden): index
                    for index, case in enumerate(cases)
                }
                indexed_results: list[tuple[int, dict[str, Any]]] = []
                for future in as_completed(futures):
                    indexed_results.append((futures[future], future.result()))
                results = [result for _, result in sorted(indexed_results)]
        dialogue_results = run_dialogues(
            list(dataset.get("dialogues", [])), sender, run_id, global_forbidden
        )

    dialogue_completion = summarize_dialogue_completion(dialogue_results)
    release_gate = evaluate_release_gates(
        results,
        dialogue_completion,
        dataset.get("release_gates"),
    )
    report = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "endpoint": args.endpoint if args.mode == "remote" else "local FastAPI /api/chat/message (LLM disabled)",
        "run_id": run_id,
        "runtime_manifest": runtime_manifest,
        "dataset": {
            "path": str(args.dataset),
            "version": dataset.get("version"),
            "sha256": dataset["sha256"],
            "single_turn_count": len(cases),
            "dialogue_count": len(dataset.get("dialogues", [])),
            "dialogue_turn_count": len(dialogue_results),
        },
        "methodology": {
            "route_hit": "accepted scenario_id and accepted intent",
            "answer_hit": "route hit + accepted resolution + required answer markers + no forbidden content + direct response when required",
            "quality_pass": "answer hit + no repeated sentence + answer length <= 900 characters",
            "remote_note": "Remote mode is a black-box end-to-end measurement; model_used comes from the API response.",
        },
        "single_turn_summary": summarize(results),
        "dialogue_summary": summarize(dialogue_results),
        "dialogue_completion": dialogue_completion,
        "release_gate": release_gate,
        "single_turn_results": results,
        "dialogue_results": dialogue_results,
    }
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "mode": args.mode,
        "single_turn": report["single_turn_summary"]["overall"],
        "dialogues": report["dialogue_summary"]["overall"],
        "model_distribution": report["single_turn_summary"]["model_distribution"],
        "latency_ms": report["single_turn_summary"]["latency_ms"],
        "release_gate": release_gate,
    }, ensure_ascii=False, indent=2))
    return 2 if args.production_gate and not release_gate["passed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
