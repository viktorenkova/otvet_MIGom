from __future__ import annotations

from backend.tools.build_stage5_labeling_form import build
from backend.tools.prepare_stage5_blind_pack import _select_dialogue_groups, validate_reviewed_pack


def _reviewed_record(record_id: str) -> dict:
    return {
        "id": record_id,
        "text": "обезличенный запрос",
        "expected": {
            "primary_topic": "поддержка",
            "specific_situation": "письменное обращение",
            "bot_action": "support",
            "required_information": ["Предложить официальный канал поддержки."],
            "no_required_information": False,
            "forbidden_information": [],
            "no_forbidden_information": True,
            "multiple_valid_answers": False,
            "acceptable_alternatives": [],
            "confidence": "high",
        },
        "review": {"status": "approved", "reviewer_id": "independent-1", "router_contributor": False},
    }


def test_blind_pack_is_fail_closed_without_independent_review_and_fifty_dialogues() -> None:
    pack = {
        "reviewer_attestation": {
            "reviewer_id": "",
            "router_contributor": None,
            "confidentiality_confirmed": False,
            "personal_data_absent_confirmed": False,
            "review_completed_at": "",
        },
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
    dialogue_record = _reviewed_record("dialogue-turn")
    dialogue_record["expected"].update({
        "continues_previous_topic": True,
        "known_context": [],
        "resolved_after_turn": True,
        "support_handoff": False,
    })
    pack = {
        "reviewer_attestation": {
            "reviewer_id": "independent-1",
            "router_contributor": False,
            "confidentiality_confirmed": True,
            "personal_data_absent_confirmed": True,
            "review_completed_at": "2026-08-24T12:00:00Z",
        },
        "cases": [_reviewed_record(f"case-{index}") for index in range(500)],
        "dialogues": [
            {
                "id": f"dialogue-{index}",
                "turns": [{**dialogue_record, "id": f"turn-{index}"}],
            }
            for index in range(50)
        ],
    }
    result = validate_reviewed_pack(pack)
    assert result["freeze_ready"] is True
    assert all(result["checks"].values())


def test_dialogue_selector_keeps_only_real_initiator_turns() -> None:
    rows = [
        {"conversation_id": "c-1", "speaker_key": "user", "message_kind": "contact_only", "text_redacted": "[phone]", "created_at": "1", "source_message_id": "1"},
        {"conversation_id": "c-1", "speaker_key": "user", "message_kind": "candidate", "text_redacted": "Проблема с оплатой", "created_at": "2", "source_message_id": "2"},
        {"conversation_id": "c-1", "speaker_key": "agent", "message_kind": "candidate", "text_redacted": "Уточните вид платежа", "created_at": "3", "source_message_id": "3"},
        {"conversation_id": "c-1", "speaker_key": "user", "message_kind": "domain_context", "text_redacted": "Списали, но баланс пуст", "created_at": "4", "source_message_id": "4"},
    ]

    groups, metrics = _select_dialogue_groups(rows, known=set())

    assert [[row["source_message_id"] for row in group] for group in groups] == [["2", "4"]]
    assert metrics["eligible_user_dialogues"] == 1


def test_offline_labeling_form_embeds_pack_without_bot_answers() -> None:
    pack = {
        "dataset_version": "test-v1",
        "source_sha256": "abc",
        "reviewer_attestation": {},
        "cases": [_reviewed_record("case-1")],
        "dialogues": [],
    }

    html = build(pack)

    assert "case-1" in html
    assert "Независимая разметка MIGTORG" in html
    assert "Проверить и скачать финальный JSON" in html
    assert "ответы бота в форме отсутствуют" in html
