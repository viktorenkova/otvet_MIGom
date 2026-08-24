from __future__ import annotations

from backend.tools.prepare_stage5_blind_pack import validate_reviewed_pack


def _reviewed_record(record_id: str) -> dict:
    return {
        "id": record_id,
        "text": "обезличенный запрос",
        "expected": {"expected_scenario_ids": ["support.contact"]},
        "review": {"status": "approved", "reviewer_id": "independent-1", "router_contributor": False},
    }


def test_blind_pack_is_fail_closed_without_independent_review_and_fifty_dialogues() -> None:
    pack = {
        "reviewer_attestation": {"reviewer_id": "", "router_contributor": None},
        "cases": [_reviewed_record(f"case-{index}") for index in range(500)],
        "dialogues": [
            {"id": f"dialogue-{index}", "turns": [_reviewed_record(f"turn-{index}")]}
            for index in range(49)
        ],
    }
    result = validate_reviewed_pack(pack)
    assert result["freeze_ready"] is False
    assert result["missing_dialogues"] == 1
    assert result["checks"]["independent_reviewer_attested"] is False


def test_blind_pack_can_freeze_only_after_every_gate_is_satisfied() -> None:
    pack = {
        "reviewer_attestation": {"reviewer_id": "independent-1", "router_contributor": False},
        "cases": [_reviewed_record(f"case-{index}") for index in range(500)],
        "dialogues": [
            {"id": f"dialogue-{index}", "turns": [_reviewed_record(f"turn-{index}")]}
            for index in range(50)
        ],
    }
    result = validate_reviewed_pack(pack)
    assert result["freeze_ready"] is True
    assert all(result["checks"].values())
