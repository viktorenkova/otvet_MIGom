from __future__ import annotations

import json
from pathlib import Path

from backend.tools.evaluate_candidate_retrieval_blind import evaluate as evaluate_blind
from backend.tools.evaluate_semantic_retrieval_v31 import evaluate as evaluate_development
from backend.tools.evaluate_stage2_language_validation import evaluate as evaluate_language


GATES = (
    (
        Path("tests/data/retrieval_v31_development_validation.json"),
        Path("reports/semantic-retrieval-v31-development-validation.json"),
        evaluate_development,
    ),
    (
        Path("tests/data/routing_v3_independent_acceptance.json"),
        Path("reports/candidate-retrieval-v31-independent-116.json"),
        evaluate_blind,
    ),
    (
        Path("tests/data/stage2_language_validation.json"),
        Path("reports/stage2-language-validation.json"),
        evaluate_language,
    ),
)


def main() -> int:
    summary = {}
    passed = True
    for dataset_path, report_path, evaluator in GATES:
        dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
        result = evaluator(dataset)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        summary[report_path.name] = {
            "passed": result["passed"],
            "overall": result.get("overall"),
            "development": result.get("development"),
            "validation": result.get("validation"),
            "by_class": result.get("by_class"),
        }
        passed = passed and result["passed"]
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
