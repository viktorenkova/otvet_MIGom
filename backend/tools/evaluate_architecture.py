"""Fresh offline A/B/C/D replay. Saved reports are labels, never runtime evidence.

Run from repository root: python -m backend.tools.evaluate_architecture
No live delivery, no label edits, no promotion and no synthetic expert scores.
"""
import argparse
from collections import Counter
from contextlib import nullcontext
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
import time
from unittest.mock import patch


def metrics(rows):
    counts = Counter(k for row in rows for k, ok in row["checks"].items() if ok)
    high = [r for r in rows if r["response"].get("confidence_level") == "high"]
    wrong = sum(not r["checks"]["answer_hit"] for r in high)
    n = len(rows)
    return {"total": n, "checks": dict(counts),
        "e2e_automatic_proxy_pct": round(100 * counts["answer_hit"] / max(1, n), 2),
        "high_count": len(high), "confident_wrong_answer_count": wrong,
        "confident_wrong_among_high_pct": round(100 * wrong / len(high), 2) if high else None,
        "confident_wrong_among_all_pct": round(100 * wrong / max(1, n), 2),
        "resolutions": dict(Counter(r["response"].get("resolution") for r in rows)),
        "ticket_offered": sum(bool(r["response"].get("needs_ticket")) for r in rows),
        "ticket_created": sum(bool(r["response"].get("ticket_id")) for r in rows),
        "ticket_delivered": None,
        "p95_ms": round(sorted(r["latency_ms"] for r in rows)[max(0, __import__('math').ceil(n * .95) - 1)], 2)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("tests/data/routing_v3_closed_control_270.json"))
    parser.add_argument("--overlay", type=Path, default=Path("tests/data/routing_label_adjudication_110.json"))
    parser.add_argument("--include-llm", action="store_true", help="Use the configured paid provider for C")
    parser.add_argument("--output", type=Path, default=Path("reports/architecture-experiment.json"))
    parser.add_argument("--details", type=Path, default=Path(".work/architecture-experiment-details.json"))
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="migtorg-architecture-") as temporary:
        os.environ.update(DATABASE_PATH=str(Path(temporary) / "evaluation.sqlite3"),
            TICKET_EMAIL_ENABLED="false", INTERNAL_STATUS_API_ENABLED="false", LLM_ENABLED="false",
            ARCHITECTURE_EXPERIMENT="true", ROUTING_ARCHITECTURE="control",
            SEMANTIC_MODEL_ALLOW_DOWNLOAD="false", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
            OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
        from backend.app import main as app
        from backend.app.bot import knowledge_search as ks
        from backend.app.models.chat import ChatRequest
        from backend.app.build_manifest import build_runtime_manifest
        from backend.tools.evaluate_live_queries import load_dataset, evaluate_response
        from fastapi.testclient import TestClient

        dataset, cases = load_dataset(args.dataset, args.overlay)
        if args.limit:
            cases = cases[:args.limit]
        app.settings.llm_understanding_enabled = False
        start = time.perf_counter()
        ks.warm_knowledge_indexes()
        warm_seconds = time.perf_counter() - start
        report = {"schema_version": 1, "generated_at": datetime.now(timezone.utc).isoformat(),
            "dataset_sha256": dataset["sha256"], "adjudication": dataset["adjudication"],
            "warmup_seconds": round(warm_seconds, 2), "fresh_runtime": True,
            "assessment": "automatic marker proxy; expert semantic assessment unavailable",
            "release_allowed": False, "variants": {}, "label_access_conflicts": [],
            "C": "not run: requires configured non-mock provider and separate explicit evaluation enablement"}
        details = {}
        frozen = build_runtime_manifest(app.settings)
        bundle_keys = ("application_bundle_sha256", "knowledge_sha256", "evaluator_sha256",
                       "scorer_artifact_sha256", "scorer_config_sha256", "candidate_scorer_config_sha256",
                       "feature_schema_sha256", "policy_bundle_sha256", "widget_bundle_sha256")
        for case in cases:
            ids = [i for i in case["expected"].get("expected_scenario_ids", []) if i]
            if ids and not any(ks.get_article_by_id(i, "guest") for i in ids):
                report["label_access_conflicts"].append(case["id"])
        client = TestClient(app.app)
        candidates_by_variant = {}
        for variant in (("A", "B", "C", "D") if args.include_llm else ("A", "B", "D")):
            app.settings.routing_architecture = "local" if variant in {"B", "C"} else "control"
            app.settings.llm_understanding_enabled = variant == "C"
            if variant == "C":
                from backend.app.config import validate_llm_runtime
                validate_llm_runtime(app.settings)
                report["C"] = "executed; schema failures and local fallback are recorded per request"
            manifest = build_runtime_manifest(app.settings)
            rows, parity_failures, candidates_by_variant[variant] = [], [], {}
            for i, case in enumerate(cases):
                oracle_ids = [x for x in case["expected"].get("expected_scenario_ids", [])
                              if x and ks.get_article_by_id(x, "guest")]
                if variant == "D" and not oracle_ids:
                    continue
                oracle = (patch.object(app, "search_knowledge_match", return_value=ks.KnowledgeSearchResult(
                    ks.get_article_by_id(oracle_ids[0], "guest"), 300, "high",
                    matched_features=["offline_oracle_label"])) if variant == "D" else nullcontext())
                with oracle:
                    started = time.perf_counter()
                    response = app.process_chat_message(ChatRequest(message=case["text"], session_id=f"{variant}-direct-{i}"))
                    elapsed = (time.perf_counter() - started) * 1000
                result = evaluate_response(case, response.model_dump(), elapsed,
                    global_forbidden=dataset.get("global_forbidden_answer_fragments", []))
                with app.logger._connect() as conn:
                    trace_row = conn.execute("SELECT trace_json FROM decision_traces WHERE message_id = ?", (response.message_id,)).fetchone()
                result["trace"] = json.loads(trace_row[0])
                candidates_by_variant[variant][case["id"]] = result["trace"].get("candidates", [])
                rows.append(result)
                if variant != "D":
                    received = client.post("/api/chat/message", json={"message": case["text"], "session_id": f"{variant}-http-{i}"})
                    actual = received.json()
                    fields = ("scenario_id", "intent", "answer", "confidence_level", "resolution", "needs_ticket")
                    if received.status_code != 200 or any(actual.get(f) != response.model_dump().get(f) for f in fields):
                        parity_failures.append(case["id"])
                if (i + 1) % 50 == 0:
                    print(f"{variant}: {i + 1}/{len(cases)}", flush=True)
            details[variant] = rows
            summary = metrics(rows) if rows else {"total": 0}
            eligible = [r for r in rows if r["id"] not in report["label_access_conflicts"]]
            expected_candidate_rows = [r for r in rows if r["expected"].get("expected_scenario_ids")
                                      and any(r["expected"]["expected_scenario_ids"])
                                      and r["id"] not in report["label_access_conflicts"]]
            recalled = sum(any(c["scenario_id"] in r["expected"]["expected_scenario_ids"]
                               for c in r["trace"].get("candidates", [])) for r in expected_candidate_rows)
            summary["access_compatible_subset"] = metrics(eligible) if eligible else None
            summary["recall_at_10"] = ({"recalled": recalled, "total": len(expected_candidate_rows)} if variant != "D" else None)
            summary.update(manifest=manifest, http_parity_failures=parity_failures,
                           expert_semantic_success=None)
            report["variants"][variant] = summary
        ids = candidates_by_variant["A"].keys() & candidates_by_variant["B"].keys()
        report["same_top10"] = all(candidates_by_variant["A"][k] == candidates_by_variant["B"][k] for k in ids)
        ending = build_runtime_manifest(app.settings)
        report["bundle_unchanged_during_run"] = all(frozen[k] == ending[k] for k in bundle_keys)
        if "C" in report["variants"]:
            b, c = report["variants"]["B"], report["variants"]["C"]
            report["llm_engineering_checks"] = {
                "automatic_proxy_gain_gte_5pp": c["e2e_automatic_proxy_pct"] - b["e2e_automatic_proxy_pct"] >= 5,
                "confident_wrong_count_not_increased": c["confident_wrong_answer_count"] <= b["confident_wrong_answer_count"],
                "p95_lte_5000ms": c["p95_ms"] <= 5000,
                "semantic_e2e_and_critical_review": None,
            }
        report["local_selection"] = "not promoted: independent semantic gate unavailable; inspect automatic comparison"
        report["next_gate"] = "resolve observed errors before dialogue/answer expansion; independent 500+100 and live shadow still required"
        for path, payload in ((args.output, report), (args.details, details)):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({k: {x: v for x, v in row.items() if x != "manifest"} for k, row in report["variants"].items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
