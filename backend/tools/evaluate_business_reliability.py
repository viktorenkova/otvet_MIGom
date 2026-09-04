"""Fresh regression after reliability changes; no business services or LLM calls."""
import json
import os
from pathlib import Path
import tempfile
import time
from datetime import datetime, timezone


def main():
    with tempfile.TemporaryDirectory(prefix="migtorg-reliability-") as temp:
        os.environ.update(DATABASE_PATH=str(Path(temp) / "eval.sqlite3"), LLM_ENABLED="false",
            LLM_UNDERSTANDING_ENABLED="false", TICKET_EMAIL_ENABLED="false", INTERNAL_STATUS_API_ENABLED="false",
            ROUTING_ARCHITECTURE="control", DIALOGUE_STATE_ENABLED="true", ANSWER_ASSEMBLY_ENABLED="true",
            ARCHITECTURE_EXPERIMENT="true", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
            SEMANTIC_MODEL_ALLOW_DOWNLOAD="false", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
        from backend.app import main as app
        from backend.app.models.chat import ChatRequest
        from backend.app.build_manifest import build_runtime_manifest
        from backend.tools.evaluate_live_queries import load_dataset, evaluate_response
        from backend.tools.evaluate_architecture import metrics
        from fastapi.testclient import TestClient
        dataset, cases = load_dataset(Path("tests/data/routing_v3_closed_control_270.json"),
                                     Path("tests/data/routing_label_adjudication_110.json"))
        app.warm_knowledge_indexes()
        manifest = build_runtime_manifest(app.settings)
        rows, parity = [], []
        http_count = 0
        client = TestClient(app.app)
        for i, case in enumerate(cases):
            started = time.perf_counter()
            response = app.process_chat_message(ChatRequest(message=case["text"], session_id=f"reliability-{i}"))
            rows.append(evaluate_response(case, response.model_dump(), (time.perf_counter()-started)*1000,
                global_forbidden=dataset.get("global_forbidden_answer_fragments", [])))
            if i % 20 == 0:
                http_count += 1
                result = client.post("/api/chat/message", json={"message": case["text"], "session_id": f"http-reliability-{i}"})
                if result.status_code != 200 or any(result.json().get(k) != response.model_dump().get(k)
                    for k in ("answer", "scenario_id", "resolution", "actions", "template_links", "action_result")):
                    parity.append(case["id"])
            if (i+1) % 50 == 0:
                print(f"reliability: {i+1}/{len(cases)}", flush=True)
        baseline = json.loads(Path("reports/answer-assembly-evaluation.json").read_text(encoding="utf-8"))
        ending = build_runtime_manifest(app.settings)
        report = {"schema_version": 1, "generated_at": datetime.now(timezone.utc).isoformat(),
            "fresh_runtime": True, "release_allowed": False,
            "assessment": "Automatic labels and marker checks; no independent expert acceptance or load test.",
            "manifest": manifest, "dataset_sha256": dataset["sha256"], "adjudication": dataset["adjudication"],
            "runtime": metrics(rows), "http_samples": http_count, "http_parity_failures": parity,
            "historical_reference": {"file": "reports/answer-assembly-evaluation.json",
                "runtime": baseline["variants"]["assembly_on"]["runtime"],
                "same_dataset": dataset["sha256"] == baseline["dataset_sha256"],
                "same_knowledge": manifest["knowledge_sha256"] == baseline["variants"]["assembly_on"]["manifest"]["knowledge_sha256"]},
            "bundle_unchanged": all(manifest[k] == ending[k] for k in
                ("application_bundle_sha256", "knowledge_sha256", "policy_bundle_sha256", "evaluator_sha256"))}
        Path("reports/business-reliability-evaluation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
        Path(".work").mkdir(exist_ok=True)
        Path(".work/business-reliability-details.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
        print(json.dumps(report["runtime"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
