from __future__ import annotations

from difflib import SequenceMatcher
from functools import lru_cache
import json
from pathlib import Path
from typing import Any

import numpy as np

from backend.app.bot.intent_classifier import classify_intent
from backend.app.bot.routing_v3 import routing_normalize
from backend.app.bot.scenario_engine import extract_query_facets, load_scenarios
from backend.app.bot.scenario_reranker import RerankedScenario


MODEL_PATH = Path("artifacts/stage3-pairwise-reranker.joblib")
FEATURE_SCHEMA_VERSION = 1


def _jaccard(left: set[str], right: set[str]) -> float:
    return len(left & right) / len(left | right) if left or right else 0.0


@lru_cache(maxsize=1)
def _atomic_titles() -> dict[str, tuple[set[str], ...]]:
    payload = json.loads(Path("knowledge/v3_1/scenarios.json").read_text(encoding="utf-8"))
    result: dict[str, list[set[str]]] = {}
    for unit in payload["atomic_units"]:
        result.setdefault(unit["canonical_scenario_id"], []).append(set(routing_normalize(unit["title"]).split()))
    return {key: tuple(value) for key, value in result.items()}


def pairwise_features(message: str, candidate: dict[str, Any], rank: int) -> list[float]:
    scenarios = {item.scenario_id: item for item in load_scenarios()}
    scenario = scenarios[str(candidate["scenario_id"])]
    query_tokens = set(routing_normalize(message).split())
    title_tokens = set(routing_normalize(scenario.title).split())
    taxonomy_tokens = {
        token for group in scenario.retrieval_taxonomy_terms
        for term in group.get("terms", []) for token in routing_normalize(str(term)).split()
    }
    document_tokens = set(routing_normalize(scenario.search_document).split())
    positive_tokens = {
        token for example in scenario.positive_examples
        for token in routing_normalize(example).split()
    }
    atomic_titles = _atomic_titles().get(scenario.scenario_id, ())
    facets = extract_query_facets(message)
    channels = candidate.get("channels") or {}
    inferred_intent = classify_intent(message)
    state_overlap = len(facets.states.intersection(scenario.states))
    numeric = [
        float(candidate.get("score", 0.0)), 1.0 / (rank + 1),
        float(channels.get("lexical", 0.0)), float(channels.get("char", 0.0)),
        float(channels.get("word", 0.0)), float(channels.get("dense", 0.0)),
        _jaccard(query_tokens, title_tokens),
        len(query_tokens & taxonomy_tokens) / max(1, len(query_tokens)),
        SequenceMatcher(None, " ".join(sorted(query_tokens)), " ".join(sorted(title_tokens))).ratio(),
        _jaccard(query_tokens, document_tokens),
        len(query_tokens & document_tokens) / max(1, len(query_tokens)),
        len(query_tokens & positive_tokens) / max(1, len(query_tokens)),
        max((_jaccard(query_tokens, tokens) for tokens in atomic_titles), default=0.0),
        max((len(query_tokens & tokens) / max(1, len(query_tokens)) for tokens in atomic_titles), default=0.0),
        max((SequenceMatcher(None, " ".join(sorted(query_tokens)), " ".join(sorted(tokens))).ratio() for tokens in atomic_titles), default=0.0),
        float(len(facets.objects.intersection(scenario.objects))),
        float(len(facets.operations.intersection(scenario.operations))),
        float(state_overlap),
        float(bool(facets.states and scenario.states and not state_overlap)),
        float(inferred_intent == scenario.intent),
        float(inferred_intent in {"unknown", "lot"}),
        float(len(scenario.objects)), float(len(scenario.operations)), float(len(scenario.states)),
    ]
    numeric.extend(float(item == scenario.scenario_id) for item in sorted(scenarios))
    return numeric


class PairwiseScenarioReranker:
    def __init__(self, model_path: Path = MODEL_PATH) -> None:
        self.model_path = model_path
        self.bundle: dict[str, Any] | None = None
        self.error = ""
        try:
            import joblib

            bundle = joblib.load(model_path)
            if bundle.get("feature_schema_version") != FEATURE_SCHEMA_VERSION:
                raise ValueError("pairwise feature schema mismatch")
            self.bundle = bundle
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"

    @property
    def available(self) -> bool:
        return self.bundle is not None

    def rerank(self, message: str, candidates: list[dict[str, Any]]) -> tuple[RerankedScenario, ...]:
        if not self.bundle or not candidates:
            return ()
        scenario_ids = {item.scenario_id for item in load_scenarios()}
        candidates = [row for row in candidates if str(row.get("scenario_id")) in scenario_ids]
        if not candidates:
            return ()
        features = np.asarray([pairwise_features(message, row, rank) for rank, row in enumerate(candidates)], dtype=np.float32)
        probabilities = self.bundle["model"].predict_proba(features)[:, 1]
        ranked = [
            RerankedScenario(
                scenario_id=str(row["scenario_id"]), score=float(probability),
                probability=float(probability), retrieval_score=float(row.get("score", 0.0)),
            )
            for row, probability in zip(candidates, probabilities)
        ]
        return tuple(sorted(ranked, key=lambda item: (item.score, item.retrieval_score), reverse=True))


@lru_cache(maxsize=1)
def get_pairwise_reranker() -> PairwiseScenarioReranker:
    return PairwiseScenarioReranker()
