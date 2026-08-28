from __future__ import annotations

from backend.tools.build_stage5_pilot_pack import build_pilot


def _case(record_id: str, source_ref: str, status: str) -> dict:
    return {
        "id": record_id,
        "source_ref_sha256": source_ref,
        "expected": {},
        "review": {"status": status, "reviewer_id": "01", "router_contributor": False},
    }


def _expected_dialogue() -> dict:
    return {
        "primary_topic": "тема",
        "specific_situation": "ситуация",
        "bot_action": "answer",
        "required_information": [],
        "no_required_information": True,
        "forbidden_information": [],
        "no_forbidden_information": True,
        "multiple_valid_answers": False,
        "acceptable_alternatives": [],
        "confidence": "high",
        "continues_previous_topic": False,
        "known_context": [],
        "resolved_after_turn": True,
        "support_handoff": False,
    }


def test_pilot_filters_agent_singles_and_places_experts_first(monkeypatch) -> None:
    monkeypatch.setattr("backend.tools.build_stage5_pilot_pack.PILOT_NEW_SINGLE_COUNT", 1)
    rows = [
        {"source": "s", "source_message_id": "1", "conversation_id": "c1", "speaker_key": "buyer", "message_kind": "candidate", "created_at": "1"},
        {"source": "s", "source_message_id": "2", "conversation_id": "c1", "speaker_key": "agent", "message_kind": "candidate", "created_at": "2"},
        {"source": "s", "source_message_id": "3", "conversation_id": "c2", "speaker_key": "buyer", "message_kind": "candidate", "created_at": "1"},
    ]
    from backend.tools.build_stage5_pilot_pack import _source_ref

    draft = {
        "dataset_version": "parent",
        "source_sha256": "source",
        "reviewer_attestation": {"reviewer_id": "01", "review_completed_at": "old"},
        "cases": [
            _case("buyer-expert", _source_ref(rows[0]), "needs_review"),
            _case("agent", _source_ref(rows[1]), "approved"),
            _case("buyer-pending", _source_ref(rows[2]), "pending"),
        ],
        "dialogues": [{
            "id": "d1",
            "turns": [
                {"id": "dialogue-expert", "expected": _expected_dialogue(), "review": {"status": "needs_review"}},
                {"id": "dialogue-ready", "expected": _expected_dialogue(), "review": {"status": "pending"}},
            ],
        }],
    }

    pack, report = build_pilot(draft, rows)

    assert [case["id"] for case in pack["cases"]] == ["buyer-expert", "buyer-pending"]
    assert pack["review_order"] == ["buyer-expert", "dialogue-expert", "dialogue-ready", "buyer-pending"]
    assert pack["dialogues"][0]["turns"][1]["review"]["status"] == "approved"
    assert pack["final_export_filename"] == "stage5-blind-pilot-reviewed-pack.json"
    assert report["excluded_non_buyer_singles_by_original_status"]["approved"] == 1
