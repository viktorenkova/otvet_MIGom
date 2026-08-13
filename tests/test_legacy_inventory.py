import json
from pathlib import Path

from backend.app.bot.knowledge_search import clear_knowledge_cache, load_articles
from backend.app.bot.scenario_engine import clear_scenario_cache


def _json(path: str):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def test_every_active_legacy_record_has_exactly_one_valid_disposition():
    legacy = _json("knowledge/normalized/migtorg_knowledge_base.json")
    inventory = _json("knowledge/v2/legacy_inventory.json")
    active_ids = {
        str(item["id"])
        for item in legacy["records"]
        if str(item.get("status") or "active") == "active"
    }
    inventory_ids = [str(item["legacy_id"]) for item in inventory["records"]]

    assert len(inventory_ids) == len(set(inventory_ids))
    assert set(inventory_ids) == active_ids
    assert {item["status"] for item in inventory["records"]} <= set(inventory["allowed_statuses"])


def test_migrated_merged_and_deactivated_legacy_records_are_not_search_articles():
    inventory = _json("knowledge/v2/legacy_inventory.json")
    suppressed = {
        str(item["legacy_id"])
        for item in inventory["records"]
        if item["status"] in {"migrated_to_v2", "merged_into_v2", "deactivated"}
    }
    clear_scenario_cache()
    clear_knowledge_cache()
    loaded_ids = {article.slug for article in load_articles()}

    assert suppressed
    assert not (suppressed & loaded_ids)
