"""Calibrate the actual saved scorer on development data only, then bind config.

The existing candidate snapshot is explicitly fingerprinted; this is not a fresh
retrieval or blind evaluation. Control/evaluation labels never enter calibration.
"""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    from backend.app.bot.pairwise_reranker import PairwiseScenarioReranker, MODEL_PATH
    from backend.app.bot.scenario_reranker import scenario_family
    from backend.tools.evaluate_stage3_pairwise import _cases
    from backend.tools.evaluate_stage3_reranker import _calibrate, _confidence_metrics
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path(".work/calibrated-reranker-config.json"))
    args = parser.parse_args()
    paths = [Path("tests/data/retrieval_v31_development_validation.json"),
             Path("reports/semantic-retrieval-v31-development-validation.json"),
             Path("tests/data/stage3_hard_negatives.json")]
    cases = _cases(*(json.loads(p.read_text(encoding="utf-8")) for p in paths))
    scorer = PairwiseScenarioReranker()
    if not scorer.available:
        raise RuntimeError(scorer.error)
    rows = []
    for case in cases:
        if case["split"] != "development":
            continue
        ranked = scorer.rerank(case["text"], case["candidates"])
        if not ranked:
            continue
        rows.append({"family": scenario_family(ranked[0].scenario_id),
            "correct": ranked[0].scenario_id in case["expected"],
            "probability": ranked[0].probability,
            "margin": ranked[0].probability - (ranked[1].probability if len(ranked) > 1 else 0)})
    thresholds = _calibrate(rows)
    config = json.loads(Path("configs/reranker_config.json").read_text(encoding="utf-8"))
    config.update(high_confidence_thresholds=thresholds, artifact_sha256=sha(MODEL_PATH),
        calibration={"method": "saved runtime artifact; predicted scenario family; development only",
            "candidate_source": "existing snapshot, not fresh retrieval",
            "inputs": {str(p): sha(p) for p in paths}, "development_count": len(rows),
            "development_metrics": _confidence_metrics(rows, thresholds),
            "independent_validation": False})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(config["calibration"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
