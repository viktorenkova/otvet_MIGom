"""Fresh flag-off/on regression. Existing labels stay frozen; derived probes are explicit.

python -B -m backend.tools.evaluate_dialogue_state
No external provider, business API, delivery, label edits, or release promotion.
"""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("reports/dialogue-state-evaluation.json"))
    parser.add_argument("--details", type=Path, default=Path(".work/dialogue-state-details.json"))
    parser.add_argument("--skip-singles", action="store_true")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="migtorg-dialogue-") as temporary:
        os.environ.update(DATABASE_PATH=str(Path(temporary) / "evaluation.sqlite3"),
            TICKET_EMAIL_ENABLED="false", INTERNAL_STATUS_API_ENABLED="false", LLM_ENABLED="false",
            LLM_UNDERSTANDING_ENABLED="false", ARCHITECTURE_EXPERIMENT="true", ROUTING_ARCHITECTURE="control",
            DIALOGUE_STATE_ENABLED="false", SEMANTIC_MODEL_ALLOW_DOWNLOAD="false", HF_HUB_OFFLINE="1",
            TRANSFORMERS_OFFLINE="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
        from backend.app import main as app
        from backend.app.bot.dialog_logger import DialogLogger
        from backend.app.bot.knowledge_search import warm_knowledge_indexes
        from backend.app.build_manifest import build_runtime_manifest
        from backend.app.models.chat import ChatRequest
        from backend.tools.evaluate_architecture import metrics
        from backend.tools.evaluate_live_queries import load_dataset, run_case, run_dialogues, summarize_dialogue_completion
        from fastapi.testclient import TestClient

        dataset, cases = load_dataset(Path("tests/data/routing_v3_closed_control_270.json"),
                                     Path("tests/data/routing_label_adjudication_110.json"))
        # These are development probes derived from existing dialogues, not independent labels.
        derived = deepcopy(dataset["dialogues"][:3])
        for dialogue, followup in zip(derived, ("офис MIGTORG", "подать заявление", "а как его подключить?")):
            dialogue["id"] += "-free-derived"
            turn = dialogue["turns"][1]
            turn.pop("select_option_contains", None)
            turn.pop("select_action_contains", None)
            turn["text"] = followup
        report = {"schema_version": 1, "generated_at": datetime.now(timezone.utc).isoformat(),
            "fresh_runtime": True, "dataset_sha256": dataset["sha256"], "adjudication": dataset["adjudication"],
            "assessment": "automatic route/resolution/marker checks; no expert semantic assessment",
            "release_allowed": False, "variants": {},
            "derived_probe_definition": derived,
            "derived_probe_status": "development probes from existing dialogues; not independent acceptance"}
        started = time.perf_counter()
        warm_knowledge_indexes()
        report["warmup_seconds"] = round(time.perf_counter() - started, 2)
        details = {}
        client = TestClient(app.app)
        for enabled in (False, True):
            name = "dialogue_on" if enabled else "dialogue_off"
            app.settings.dialogue_state_enabled = enabled
            manifest = build_runtime_manifest(app.settings)
            latest = {}
            def sender(payload):
                # Recreate the DAO for each turn to exercise process-independent state.
                app.logger = DialogLogger(app.logger.database_path)
                payload = deepcopy(payload)
                previous = latest.get(payload["session_id"])
                if previous:
                    payload["conversation_turn_id"] = previous["message_id"]
                    payload["state_version"] = previous["state_version"]
                response = client.post("/api/chat/message", json=payload)
                response.raise_for_status()
                latest[payload["session_id"]] = response.json()
                return response.json()
            original = run_dialogues(dataset["dialogues"], sender, name, dataset.get("global_forbidden_answer_fragments", []))
            probes = run_dialogues(derived, sender, name, dataset.get("global_forbidden_answer_fragments", []))
            singles, parity = [], []
            if not args.skip_singles:
                for i, case in enumerate(cases):
                    row = run_case(case, sender, name, dataset.get("global_forbidden_answer_fragments", []))
                    singles.append(row)
                    direct = app.process_chat_message(ChatRequest(message=case["text"], session_id=f"{name}-direct-{i}")).model_dump()
                    fields = ("scenario_id", "answer", "intent", "resolution", "confidence_level", "pending_requests")
                    if any(direct.get(field) != (row.get("response") or {}).get(field) for field in fields):
                        parity.append(case["id"])
                    if (i + 1) % 50 == 0:
                        print(f"{name}: {i + 1}/{len(cases)}", flush=True)
            with app.logger._connect() as conn:
                traces = [json.loads(r[0]) for r in conn.execute("SELECT trace_json FROM decision_traces")]
            details[name] = {"original": original, "derived": probes, "singles": singles, "traces": traces}
            report["variants"][name] = {"manifest": manifest,
                "original_dialogues": summarize_dialogue_completion(original),
                "derived_dialogues": summarize_dialogue_completion(probes),
                "single_turn": metrics(singles) if singles else None,
                "http_direct_parity_failures": parity,
                "dialogue_failures": [{"id": r["id"], "checks": r["checks"], "error": r.get("error"),
                    "expected": r["expected"], "scenario_id": (r.get("response") or {}).get("scenario_id"),
                    "resolution": (r.get("response") or {}).get("resolution")} for r in original + probes if not r["checks"]["answer_hit"]]}
            print(name, json.dumps({k:v for k,v in report["variants"][name].items() if k != "manifest"}, ensure_ascii=False), flush=True)
        before = details["dialogue_off"]["singles"]
        after = details["dialogue_on"]["singles"]
        report["single_turn_changes"] = [{"id": a["id"], "before_pass": a["checks"]["answer_hit"],
            "after_pass": b["checks"]["answer_hit"], "before_scenario": a["response"]["scenario_id"],
            "after_scenario": b["response"]["scenario_id"]} for a,b in zip(before, after)
            if a["checks"]["answer_hit"] != b["checks"]["answer_hit"] or a["response"]["scenario_id"] != b["response"]["scenario_id"]]
        ending = build_runtime_manifest(app.settings)
        report["bundle_unchanged_during_run"] = all(ending[key] == report["variants"]["dialogue_off"]["manifest"][key]
            for key in ("application_bundle_sha256", "knowledge_sha256", "evaluator_sha256"))
        report["evaluator_file_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        for path, payload in ((args.output, report), (args.details, details)):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
