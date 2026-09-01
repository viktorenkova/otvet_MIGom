from __future__ import annotations

import json
from pathlib import Path

from backend.app.bot.knowledge_search import _load_v2_articles, clear_knowledge_cache
from backend.app.bot.scenario_engine import load_scenarios
from backend.tools.master_knowledge import load_master
from backend.tools.migrate_knowledge_v31 import migrate
from backend.tools.compare_knowledge_v31_regressions import compare
from backend.tools.validate_knowledge_v31 import validate


ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = ROOT / "knowledge/MASTER_KNOWLEDGE.md"
KNOWLEDGE_PATH = ROOT / "knowledge/v3_1/scenarios.json"
CONFLICTS_PATH = ROOT / "knowledge/v3_1/scenario_conflicts.json"


def _payload(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_v31_migration_is_deterministic_and_preserves_every_scenario_and_fact() -> None:
    source, knowledge_gaps = load_master(SOURCE_PATH)
    committed = _payload(KNOWLEDGE_PATH)
    rebuilt = migrate(source, knowledge_gaps=knowledge_gaps)

    assert rebuilt == committed
    assert {item["scenario_id"] for item in committed["records"]} == {
        item["scenario_id"] for item in source["records"]
    }
    assert sum(len(item["facts"]) for item in source["records"]) == 585
    assert sum(len(item["fact_records"]) for item in committed["records"]) == 585


def test_v31_strict_validator_passes_every_gate() -> None:
    source, _ = load_master(SOURCE_PATH)
    result = validate(source, _payload(KNOWLEDGE_PATH), _payload(CONFLICTS_PATH))

    assert result["valid"] is True
    assert result["errors"] == []
    assert result["metrics"]["active_scenario_count"] == 142
    assert result["metrics"]["fact_loss_count"] == 0
    assert result["metrics"]["financial_or_contract_facts_with_provenance_pct"] == 100.0
    assert result["metrics"]["atomic_unit_count"] == 158
    assert result["metrics"]["knowledge_gap_count"] == 3


def test_runtime_loads_v31_with_traceable_answer_policy() -> None:
    clear_knowledge_cache()
    scenarios = load_scenarios()

    assert len(scenarios) == 142
    assert all(item.domain and item.source_version for item in scenarios)
    assert all(item.fact_records and item.answer_policy for item in scenarios)
    assert all(item.retrieval_taxonomy_terms for item in scenarios)
    assert all(item.answer_policy["fact_scope"] == "listed_fact_ids_only" for item in scenarios)


def test_search_documents_exclude_full_answers_and_fact_paragraphs() -> None:
    clear_knowledge_cache()
    scenarios = {item.scenario_id: item for item in load_scenarios()}
    articles = {item.slug: item for item in _load_v2_articles()}

    for scenario_id, scenario in scenarios.items():
        article = articles[scenario_id]
        assert article.search_document == scenario.search_document
        assert scenario.detailed_answer not in article.content
        assert scenario.short_answer not in article.search_document
        assert all(fact not in article.search_document for fact in scenario.facts)


def test_mixed_priority_topics_are_split_into_atomic_retrieval_units() -> None:
    knowledge = _payload(KNOWLEDGE_PATH)
    units_by_scenario: dict[str, list[dict]] = {}
    for unit in knowledge["atomic_units"]:
        units_by_scenario.setdefault(unit["canonical_scenario_id"], []).append(unit)

    assert len(units_by_scenario["payment.methods"]) == 5
    assert len(units_by_scenario["tariff.choose"]) == 3
    assert len(units_by_scenario["bid.autobid_extension"]) == 3
    assert len(units_by_scenario["pickup.receive_lot"]) == 2
    assert len(units_by_scenario["lot.payment.details"]) == 3
    assert len(units_by_scenario["refund.application"]) == 3


def test_every_observed_confusion_pair_has_distinctions_and_policy() -> None:
    conflicts = _payload(CONFLICTS_PATH)

    assert conflicts["record_count"] == len(conflicts["records"]) >= 40
    assert all(item["distinctions"] and item["decision_policy"] for item in conflicts["records"])
    assert all(item["decision_policy"]["never_use_full_answer_as_retrieval_evidence"] for item in conflicts["records"])


def test_accepted_knowledge_gaps_have_safe_non_invented_answers() -> None:
    gaps = _payload(KNOWLEDGE_PATH)["knowledge_gaps"]

    assert {item["gap_id"] for item in gaps} == {
        "gap.lot_photo_archive_download",
        "gap.tariff_expired_unused",
        "gap.tariff_access_term_unspecified",
    }
    assert all(item["status"] == "owner_confirmation_required" for item in gaps)
    assert all(item["safe_answer"] and item["answer_policy"] for item in gaps)


def test_stage1_regression_comparison_gate_passes() -> None:
    result = compare()

    assert result["passed"] is True
    assert result["errors"] == []
    assert {item["corpus"] for item in result["corpora"]} == {
        "gold_312",
        "independent_116",
        "live_160",
        "closed_270_adjudicated",
    }
