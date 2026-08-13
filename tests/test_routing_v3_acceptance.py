from __future__ import annotations

import hashlib
import json
from pathlib import Path

from backend.app.bot.routing_v3 import get_routing_v3
from backend.tools.build_routing_v3_control_set import WIDGET_SCENARIOS
from backend.tools.generate_routing_variants import TRANSFORMS, build as build_variants


ROOT = Path(__file__).resolve().parents[1]
CONTROL_PATH = ROOT / "tests/data/routing_v3_closed_control_270.json"
SEEDS_PATH = ROOT / "tests/data/routing_v3_acceptance_seeds.json"
VARIANTS_PATH = ROOT / "tests/data/routing_v3_independent_acceptance.json"


def _canonical_cases_sha256(cases: list[dict]) -> str:
    canonical = json.dumps(cases, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def test_closed_control_contains_exactly_270_locked_cases() -> None:
    payload = json.loads(CONTROL_PATH.read_text(encoding="utf-8"))
    assert payload["id"] == "routing-v3-closed-control-270"
    assert payload["case_count"] == len(payload["cases"]) == 270
    assert len(WIDGET_SCENARIOS) == 110
    assert len({item["id"] for item in payload["cases"]}) == 270
    assert payload["cases_sha256"] == _canonical_cases_sha256(payload["cases"])


def test_independent_variants_are_reproducible_and_not_seed_copies() -> None:
    committed = json.loads(VARIANTS_PATH.read_text(encoding="utf-8"))
    generated = build_variants(SEEDS_PATH)
    assert committed == generated
    assert committed["case_count"] == len(committed["cases"])
    seeds = json.loads(SEEDS_PATH.read_text(encoding="utf-8"))["seeds"]
    assert committed["case_count"] == len(seeds) * len(TRANSFORMS)
    seed_texts = {item["text"] for item in seeds}
    assert not seed_texts.intersection(item["text"] for item in committed["cases"])


def test_v3_high_confidence_routes_have_no_false_positive_on_smoke_set() -> None:
    router = get_routing_v3()
    samples = {
        "карта не принимает оплату доступа": "payment.checkout_problem",
        "изображения лота совсем не грузятся": "technical.lot_image_missing",
        "хочу отменить своё предложение цены": "bid.modify_cancel",
        "страховая не отвечает по выдаче машины": "transfer.seller_no_response",
        "хочу приехать в офис подписать договор": "support.office_visit",
    }
    for text, expected in samples.items():
        decision = router.decide(text, role="guest")
        assert decision.confidence == "high", (text, decision)
        assert decision.scenario is not None
        assert decision.scenario.scenario_id == expected
