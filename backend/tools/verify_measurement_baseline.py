"""Reproduce one pinned offline runtime in fresh processes through direct and HTTP paths."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "tests/data/routing_v3_closed_control_270.json"
OVERLAY = ROOT / "tests/data/routing_label_adjudication_110.json"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def comparable_response(response):
    return {k: v for k, v in response.items() if k not in {"message_id", "session_id"}}


def scenario_coverage(scenarios, canonical_cards, cards, scorer_ids, supplementary_cards):
    """Require exact canonical coverage while accounting for declared legacy cards."""
    return {
        "complete": (scenarios == canonical_cards == scorer_ids
            and cards == canonical_cards | supplementary_cards
            and not canonical_cards & supplementary_cards),
        "active": len(scenarios), "canonical_cards": len(canonical_cards),
        "cards": len(cards), "scorer_columns": len(scorer_ids),
        "supplementary_card_ids": sorted(supplementary_cards),
        "supplementary_source": "knowledge_search._load_normalized_articles (active, non-suppressed records and overrides)",
        "missing_canonical_cards": sorted(scenarios - canonical_cards),
        "missing_scorer_columns": sorted(scenarios - scorer_ids),
        "unaccounted_cards": sorted(cards - canonical_cards - supplementary_cards),
    }


def compare_runs(runs):
    first = runs[0]
    expected = first["case_ids"]
    checks = {
        "fresh_processes": len({r["process_id"] for r in runs}) == len(runs),
        "all_cases_in_original_order": all(r["case_ids"] == expected and len(r["rows"]) == len(expected) for r in runs),
        "identical_manifest": all(r["manifest"] == first["manifest"] for r in runs),
        "same_dataset_and_overlay": all(r["dataset_sha256"] == first["dataset_sha256"] and
            r["overlay_sha256"] == first["overlay_sha256"] for r in runs),
        "bundle_unchanged": all(r["manifest_unchanged"] for r in runs),
        "no_transport_failures": all(row["transport_ok"] for r in runs for row in r["rows"]),
        "complete_final_trace": all(row["trace_complete"] for r in runs for row in r["rows"]),
        "full_scenario_coverage": all(r["scenario_coverage"]["complete"] for r in runs),
        "runtime_thresholds_match_manifest": all(r["threshold_config_matches"] for r in runs),
        "candidate_calibration_artifact_matches": all(r["candidate_calibration_artifact_matches"] for r in runs),
    }
    differences = []
    for run in runs[1:]:
        for a, b in zip(first["rows"], run["rows"]):
            changed = [k for k in ("response", "candidate_ids", "decision") if a[k] != b[k]]
            if changed:
                differences.append({"case_id": a["id"], "path": run["path"], "changed": changed})
    checks["identical_responses_routes_and_candidates"] = not differences
    return {"measurement_stage_closed": all(checks.values()), "checks": checks, "differences": differences,
        "production_release_allowed": False,
        "assessment": "Fresh deterministic behavior and automatic marker checks; expert semantic quality is not measured."}


def worker(mode, output):
    with tempfile.TemporaryDirectory(prefix="migtorg-measurement-") as temp:
        os.environ.update(DATABASE_PATH=str(Path(temp)/"runtime.sqlite3"), LLM_ENABLED="false",
            LLM_UNDERSTANDING_ENABLED="false", TICKET_EMAIL_ENABLED="false", INTERNAL_STATUS_API_ENABLED="false",
            ROUTING_ARCHITECTURE="control", DIALOGUE_STATE_ENABLED="false", ANSWER_ASSEMBLY_ENABLED="false",
            ARCHITECTURE_EXPERIMENT="true", SEMANTIC_DENSE_ENABLED="true", SEMANTIC_MODEL_ALLOW_DOWNLOAD="false",
            HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
            LLM_PROVIDER="mock", LLM_PRIMARY_MODEL="mock/safe-rules", LLM_FALLBACK_MODEL="mock/safe-rules",
            CHAT_MAX_CONCURRENCY="8", PYTHONHASHSEED="0")
        from backend.app import main as app
        from backend.app.bot import knowledge_search as ks
        from backend.app.bot.scenario_engine import load_scenarios
        from backend.app.bot.scenario_reranker import load_reranker_config
        from backend.app.bot.pairwise_reranker import get_pairwise_reranker
        from backend.app.build_manifest import build_runtime_manifest
        from backend.app.models.chat import ChatRequest
        from backend.tools.evaluate_live_queries import load_dataset, evaluate_response
        from backend.tools.evaluate_architecture import metrics
        from fastapi.testclient import TestClient
        dataset, cases = load_dataset(DATASET, OVERLAY)
        app.warm_knowledge_indexes()
        manifest = build_runtime_manifest(app.settings)
        scenarios = {s.scenario_id for s in load_scenarios()}
        cards = {a.slug for a in ks.load_articles()}
        canonical_cards = {a.slug for a in ks._load_v2_articles()}
        normalized = app.settings.knowledge_root / "normalized" / "migtorg_knowledge_base.json"
        supplementary_cards = ({a.slug for a in ks._load_normalized_articles(app.settings.knowledge_root, normalized)}
            if normalized.exists() else set())
        scorer = get_pairwise_reranker()
        scorer_ids = set(scorer.bundle["feature_scenario_ids"]) if scorer.available else set()
        calibrated = json.loads((ROOT / "configs/architecture_reranker_config.json").read_text(encoding="utf-8"))
        rows, evaluations = [], []
        client = TestClient(app.app)
        for i, case in enumerate(cases):
            started = time.perf_counter()
            request = {"message": case["text"], "session_id": f"measurement-{mode}-{i}"}
            if mode == "http":
                result = client.post("/api/chat/message", json=request)
                response, ok = result.json(), result.status_code == 200
            else:
                response, ok = app.process_chat_message(ChatRequest(**request)).model_dump(), True
            elapsed = (time.perf_counter()-started)*1000
            with app.logger._connect() as conn:
                trace_row = conn.execute("SELECT trace_json FROM decision_traces WHERE message_id=?", (response.get("message_id"),)).fetchone()
            trace = json.loads(trace_row[0]) if trace_row else {}
            decision = dict(trace.get("final_decision", {}))
            if "features" in decision:
                decision["features"] = sorted(decision["features"])
            rows.append({"id": case["id"], "response": comparable_response(response), "transport_ok": ok,
                "candidate_ids": [c["scenario_id"] for c in trace.get("candidates", [])], "decision": decision,
                "trace_complete": bool(decision) and "used_context" in trace and "result" in trace,
                "trace": trace})
            evaluations.append(evaluate_response(case, response, elapsed,
                global_forbidden=dataset.get("global_forbidden_answer_fragments", [])))
            if (i+1) % 50 == 0:
                print(f"{mode}: {i+1}/{len(cases)}", flush=True)
        report = {"path": mode, "process_id": os.getpid(), "case_ids": [c["id"] for c in cases],
            "dataset_sha256": sha(DATASET), "overlay_sha256": sha(OVERLAY), "manifest": manifest,
            "manifest_unchanged": manifest == build_runtime_manifest(app.settings),
            "threshold_config_matches": load_reranker_config() == manifest["effective_scorer_config"]["settings"],
            "candidate_calibration_artifact_matches": calibrated["artifact_sha256"] == manifest["scorer_artifact_sha256"],
            "scenario_coverage": scenario_coverage(scenarios, canonical_cards, cards, scorer_ids, supplementary_cards),
            "automatic_quality": metrics(evaluations), "rows": rows}
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", choices=["direct", "http", "direct_repeat"])
    parser.add_argument("--output", type=Path, default=Path("reports/measurement-baseline-verification.json"))
    args = parser.parse_args()
    if args.worker:
        worker(args.worker, args.output)
        return 0
    with tempfile.TemporaryDirectory(prefix="migtorg-measurement-driver-") as temp:
        runs = []
        for mode in ("direct", "http", "direct_repeat"):
            path = Path(temp)/(mode+".json")
            env = dict(os.environ, PYTHONHASHSEED="0", PYTHONIOENCODING="utf-8")
            subprocess.run([sys.executable, "-B", "-m", "backend.tools.verify_measurement_baseline",
                "--worker", mode, "--output", str(path)], cwd=ROOT, env=env, check=True, timeout=900)
            runs.append(json.loads(path.read_text(encoding="utf-8")))
        result = compare_runs(runs)
        result.update(generated_at=datetime.now(timezone.utc).isoformat(), corpus_count=len(runs[0]["case_ids"]),
            executions=sum(len(r["rows"]) for r in runs), manifest=runs[0]["manifest"],
            dataset_sha256=runs[0]["dataset_sha256"], overlay_sha256=runs[0]["overlay_sha256"],
            runs=[{k: v for k,v in r.items() if k not in {"rows", "manifest", "case_ids"}} for r in runs])
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
        details = ROOT / ".work/measurement-baseline-details.json"
        details.parent.mkdir(exist_ok=True)
        details.write_text(json.dumps(runs, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
        print(json.dumps({k:result[k] for k in ("measurement_stage_closed", "checks", "differences", "executions")}, ensure_ascii=False), flush=True)
        return 0 if result["measurement_stage_closed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
