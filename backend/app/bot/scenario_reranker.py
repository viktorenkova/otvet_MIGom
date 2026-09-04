from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
import math
from pathlib import Path
from typing import Any, Iterable

from backend.app.bot.scenario_engine import Scenario, extract_query_facets, load_scenarios


CONFIG_PATH = Path("configs/reranker_config.json")


@dataclass(frozen=True)
class RerankedScenario:
    scenario_id: str
    score: float
    probability: float
    retrieval_score: float


@dataclass(frozen=True)
class RerankDecision:
    scenario_id: str | None
    confidence: str
    probability: float
    margin: float
    family: str
    ranked: tuple[RerankedScenario, ...]
    clarifying_question: str = ""
    clarifying_options: tuple[str, ...] = ()
    missing_slot: str = ""
    provider: str = "cross_encoder"
    model: str = ""


def load_reranker_config() -> dict[str, Any]:
    from backend.app.config import get_settings
    root = Path(__file__).resolve().parents[3]
    path = (root / "configs/architecture_reranker_config.json"
            if get_settings().routing_architecture == "local" else root / CONFIG_PATH)
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("artifact_sha256"):
        import hashlib
        if hashlib.sha256((root / config["model"]).read_bytes()).hexdigest() != config["artifact_sha256"]:
            raise ValueError("calibration_artifact_mismatch")
    return config


def scenario_family(scenario_id: str) -> str:
    if scenario_id in {"buyer.get_started", "seller.get_started"}:
        return "onboarding"
    if scenario_id.startswith(("payment.", "commission.", "balance.", "tariff.", "lot.payment.")):
        return "payments"
    if scenario_id.startswith("account."):
        return "registration"
    if scenario_id.startswith(("bid.", "auction.")):
        return "bidding"
    if scenario_id.startswith(("transfer.", "documents.", "contract.", "pickup.", "win.")):
        return "transfer"
    if scenario_id.startswith(("lot.catalog", "lot.card", "lot.image", "technical.catalog", "technical.lot_image")):
        return "search"
    return "default"


def _scenario_document(scenario: Scenario) -> str:
    taxonomy = " ".join(
        str(term)
        for group in scenario.retrieval_taxonomy_terms
        for term in group.get("terms", [])[:4]
    )
    return " | ".join(filter(None, (
        scenario.title,
        f"объект: {' '.join(scenario.objects)}",
        f"операция: {' '.join(scenario.operations)}",
        f"состояние: {' '.join(scenario.states)}",
        taxonomy,
    )))[:1200]


def _softmax(values: list[float]) -> list[float]:
    if not values:
        return []
    peak = max(values)
    exponents = [math.exp(value - peak) for value in values]
    total = sum(exponents)
    return [value / total for value in exponents]


@lru_cache(maxsize=1)
def _scenario_map() -> dict[str, Scenario]:
    return {scenario.scenario_id: scenario for scenario in load_scenarios()}


class CrossEncoderScenarioReranker:
    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or load_reranker_config()
        self.model_name = str(self.config["model"])
        self.model = None
        self.error = ""
        try:
            from sentence_transformers import CrossEncoder

            self.model = CrossEncoder(
                self.model_name,
                device=str(self.config.get("device", "cpu")),
                local_files_only=True,
            )
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"

    @property
    def available(self) -> bool:
        return self.model is not None

    def rerank(self, message: str, candidates: Iterable[dict[str, Any]]) -> tuple[RerankedScenario, ...]:
        candidate_rows = list(candidates)
        scenarios = _scenario_map()
        usable = [row for row in candidate_rows if row.get("scenario_id") in scenarios]
        if not usable or not self.model:
            return ()
        pairs = [[message, _scenario_document(scenarios[row["scenario_id"]])] for row in usable]
        scores = [float(value) for value in self.model.predict(
            pairs,
            batch_size=max(1, int(self.config.get("batch_size", 32))),
            show_progress_bar=False,
        )]
        probabilities = _softmax(scores)
        ranked = [
            RerankedScenario(
                scenario_id=row["scenario_id"], score=score, probability=probability,
                retrieval_score=float(row.get("score", 0.0)),
            )
            for row, score, probability in zip(usable, scores, probabilities)
        ]
        return tuple(sorted(ranked, key=lambda item: (item.score, item.retrieval_score), reverse=True))


class ConstrainedFeatureClassifier:
    """Deterministic constrained baseline over the exact same candidate IDs."""

    def rerank(self, message: str, candidates: Iterable[dict[str, Any]]) -> tuple[RerankedScenario, ...]:
        scenarios = _scenario_map()
        facets = extract_query_facets(message)
        rows = []
        raw_scores = []
        for row in candidates:
            scenario = scenarios.get(str(row.get("scenario_id")))
            if not scenario:
                continue
            score = float(row.get("score", 0.0)) * 2.0
            score += len(facets.objects.intersection(scenario.objects)) * 0.45
            score += len(facets.operations.intersection(scenario.operations)) * 0.65
            score += len(facets.states.intersection(scenario.states)) * 0.80
            if facets.states and scenario.states and not facets.states.intersection(scenario.states):
                score -= 0.45
            rows.append((row, score))
            raw_scores.append(score)
        probabilities = _softmax(raw_scores)
        ranked = [
            RerankedScenario(str(row["scenario_id"]), score, probability, float(row.get("score", 0.0)))
            for (row, score), probability in zip(rows, probabilities)
        ]
        return tuple(sorted(ranked, key=lambda item: (item.score, item.retrieval_score), reverse=True))


_SLOT_LABELS = {
    "objects": "объект вопроса",
    "operations": "нужное действие",
    "states": "текущее состояние",
    "stage": "этап сделки",
}


def _clarification(ranked: tuple[RerankedScenario, ...]) -> tuple[str, str, tuple[str, ...]]:
    scenarios = _scenario_map()
    top = [scenarios[item.scenario_id] for item in ranked[:3] if item.scenario_id in scenarios]
    if len(top) < 2:
        return "", "", ()
    for field in ("states", "operations", "objects", "stage"):
        values = [tuple(getattr(scenario, field)) if field != "stage" else (scenario.stage,) for scenario in top]
        if len(set(values)) > 1:
            options = tuple(scenario.title for scenario in top)
            return field, f"Уточните {_SLOT_LABELS[field]}, чтобы я выбрал точный сценарий:", options
    return "scenario", "Уточните, какой из вариантов ближе к вашему вопросу:", tuple(item.title for item in top)


def decide_reranked(
    message: str,
    ranked: tuple[RerankedScenario, ...],
    config: dict[str, Any] | None = None,
) -> RerankDecision:
    settings = config or load_reranker_config()
    if not ranked:
        return RerankDecision(None, "low", 0.0, 0.0, "default", (), provider=str(settings.get("provider")), model=str(settings.get("model")))
    best = ranked[0]
    second_probability = ranked[1].probability if len(ranked) > 1 else 0.0
    margin = best.probability - second_probability
    family = scenario_family(best.scenario_id)
    thresholds = settings.get("high_confidence_thresholds", {})
    threshold = float(thresholds.get(family, thresholds.get("default", 0.12)))
    min_margin = float(settings.get("minimum_margin", 0.01))
    if best.probability >= threshold and margin >= min_margin:
        return RerankDecision(best.scenario_id, "high", best.probability, margin, family, ranked, provider=str(settings.get("provider")), model=str(settings.get("model")))
    slot, question, options = _clarification(ranked)
    return RerankDecision(None, "medium", best.probability, margin, family, ranked, question, options, slot, str(settings.get("provider")), str(settings.get("model")))
