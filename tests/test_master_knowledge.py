from __future__ import annotations

import json
from pathlib import Path

from backend.tools.master_knowledge import load_master_bundle, render_master, validate_master
from backend.tools.migrate_knowledge_v31 import migrate


ROOT = Path(__file__).resolve().parents[1]
MASTER_PATH = ROOT / "knowledge/MASTER_KNOWLEDGE.md"
V2_PATH = ROOT / "knowledge/v2/scenarios.json"
V3_PATH = ROOT / "knowledge/v3_1/scenarios.json"


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_master_is_complete_and_human_readable_sections_are_in_sync() -> None:
    source, gaps, review_queue = load_master_bundle(MASTER_PATH)
    result = validate_master(MASTER_PATH, V2_PATH)

    assert result["valid"] is True
    assert result["errors"] == []
    expected_ids = {r["scenario_id"] for r in _json(V2_PATH)["records"]}
    assert {r["scenario_id"] for r in source["records"]} == expected_ids
    assert result["metrics"]["scenario_count"] == len(expected_ids)
    assert result["metrics"]["fact_count"] == 585
    assert result["metrics"]["knowledge_gap_count"] == 3
    assert result["metrics"]["expert_review_candidate_count"] == 9
    assert MASTER_PATH.read_text(encoding="utf-8") == render_master(source, gaps, review_queue)


def test_master_is_the_source_for_v2_and_v31() -> None:
    source, gaps, _ = load_master_bundle(MASTER_PATH)

    assert source == _json(V2_PATH)
    assert migrate(source, knowledge_gaps=gaps) == _json(V3_PATH)


def test_master_exposes_known_content_and_provenance_gaps() -> None:
    _, gaps, review_queue = load_master_bundle(MASTER_PATH)
    text = MASTER_PATH.read_text(encoding="utf-8")

    assert {gap["gap_id"] for gap in gaps} == {
        "gap.lot_photo_archive_download",
        "gap.tariff_expired_unused",
        "gap.tariff_access_term_unspecified",
    }
    assert "обязательный provenance gap" in text
    assert "evidence_path" in text
    assert "evidence_sha256" in text
    assert len(review_queue["records"]) == 9
    assert "Просроченных повторных проверок на 2026-09-01: 60" in text
