from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from backend.app.bot.pairwise_reranker import FEATURE_SCHEMA_VERSION, MODEL_PATH, pairwise_features
from backend.tools.evaluate_stage3_pairwise import _cases


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("tests/data/retrieval_v31_development_validation.json"))
    parser.add_argument("--candidates", type=Path, default=Path("reports/semantic-retrieval-v31-development-validation.json"))
    parser.add_argument("--hard-negatives", type=Path, default=Path("tests/data/stage3_hard_negatives.json"))
    parser.add_argument("--output", type=Path, default=MODEL_PATH)
    args = parser.parse_args()
    dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
    candidates = json.loads(args.candidates.read_text(encoding="utf-8"))
    hard = json.loads(args.hard_negatives.read_text(encoding="utf-8"))
    cases = [case for case in _cases(dataset, candidates, hard) if case["split"] == "development"]
    x_train, y_train, weights = [], [], []
    for case in cases:
        for rank, candidate in enumerate(case["candidates"]):
            positive = candidate["scenario_id"] in case["expected"]
            x_train.append(pairwise_features(case["text"], candidate, rank))
            y_train.append(int(positive))
            weights.append(9.0 if positive else 1.0)
    model = HistGradientBoostingClassifier(
        learning_rate=0.06, max_iter=180, max_leaf_nodes=15,
        min_samples_leaf=12, l2_regularization=1.0, random_state=20260824,
    )
    model.fit(np.asarray(x_train, dtype=np.float32), np.asarray(y_train), sample_weight=np.asarray(weights))
    bundle = {
        "schema_version": 1, "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "model_version": "stage3-pairwise-2026.08.24.1", "model": model,
        "training_pairs": len(x_train), "feature_count": len(x_train[0]),
        "inputs": {path.name: _sha(path) for path in (args.dataset, args.candidates, args.hard_negatives)},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, args.output, compress=3)
    print(json.dumps({key: bundle[key] for key in ("model_version", "training_pairs", "feature_count", "inputs")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
