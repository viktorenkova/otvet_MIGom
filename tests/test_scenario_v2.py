import base64
import hashlib
import hmac
import json
import time
from pathlib import Path
from uuid import uuid4

import pytest

from backend.app.bot.scenario_engine import match_scenario
from backend.app.bot.answer_generator import _redact_for_llm
from backend.app.bot.trusted_context import TrustedContextError, verify_trusted_context_token
from backend.app.integrations.status_provider import StatusResult
from backend.app import main
from backend.app.main import process_chat_message, settings
from backend.app.models.chat import ChatRequest, ChatResponse
from backend.app.models.user_context import UserContext
from backend.tools.evaluate_scenarios import evaluate


def _token(secret: str, payload: dict) -> str:
    raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    part = base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")
    signature = hmac.new(secret.encode("utf-8"), part.encode("ascii"), hashlib.sha256).digest()
    return part + "." + base64.urlsafe_b64encode(signature).decode("ascii").rstrip("=")


@pytest.mark.parametrize(
    ("message", "scenario_id"),
    [
        ("как участвовать в аукционе?", "buyer.get_started"),
        ("не видна моя ставка", "bid.not_visible"),
        ("где можно забрать мой договор?", "contract.receive"),
        ("как проверить статус торгов?", "auction.status"),
        ("как подключиться продавцу?", "seller.get_started"),
        ("как выставить лот на продажу?", "seller.publish_lot"),
        ("как вам позвонить?", "support.contact"),
        ("перезвоните мне", "support.callback"),
    ],
)
def test_screenshot_queries_use_specific_scenarios(message: str, scenario_id: str):
    decision = match_scenario(message, "guest")
    assert decision.confidence == "high"
    assert decision.scenario is not None
    assert decision.scenario.scenario_id == scenario_id


def test_ambiguous_connection_request_asks_what_to_connect():
    response = process_chat_message(
        ChatRequest(message="как подключиться", session_id=f"v2-ambiguous-connect-{uuid4()}")
    )
    assert response.resolution == "clarified"
    assert "что именно" in response.answer.lower()
    assert any(option.startswith("Как начать покупать") for option in response.clarifying_options)
    assert "Как подключиться и стать продавцом" in response.clarifying_options


def test_generic_lot_question_asks_for_the_stage_instead_of_guessing():
    response = process_chat_message(
        ChatRequest(message="вопрос по моему лоту", session_id=f"v2-generic-lot-{uuid4()}")
    )
    assert response.resolution == "clarified"
    assert "что именно нужно узнать по лоту" in response.answer.lower()
    assert "Что делать после победы в торгах" in response.clarifying_options
    assert "Как получить или забрать лот" in response.clarifying_options


@pytest.mark.parametrize(
    "message",
    [
        "хочу спросить об автомобиле, но пока не знаю как сформулировать",
        "по конкретному лоту вопрос пока общий",
        "какие данные нужно сообщить, чтобы уточнить информацию по лоту",
    ],
)
def test_semantically_generic_lot_question_also_clarifies(message: str):
    decision = match_scenario(message, "guest")
    assert decision.scenario is None
    assert decision.clarifying_question == "Что именно нужно узнать по лоту?"
    assert "scenario_ambiguity:generic_lot" in decision.matched_features


def test_call_request_does_not_add_callback_action_to_specific_support_topic():
    response = process_chat_message(
        ChatRequest(
            message="выиграл лот, что дальше, перезвоните мне",
            session_id=f"v2-topic-plus-callback-{uuid4()}",
        )
    )
    assert response.scenario_id == "win.next_steps"
    assert not any(action.type == "request_callback" for action in response.actions)
    assert any(action.id == "win.next.ticket" for action in response.actions)


def test_call_request_is_redirected_to_written_support():
    response = process_chat_message(
        ChatRequest(message="перезвоните мне", session_id=f"v2-written-support-{uuid4()}")
    )

    assert response.scenario_id == "support.callback"


def test_callback_case_note_wording_is_not_out_of_scope():
    for message in ("ожидает обратной связи", "просит набрать повторно"):
        response = process_chat_message(
            ChatRequest(message=message, session_id=f"v2-callback-note-{uuid4()}")
        )
        assert response.scenario_id == "support.callback"
    assert "по переписке" in response.answer.casefold()
    assert {action.type for action in response.actions} == {"open_ticket"}


def test_scenario_response_has_message_id_and_structured_actions():
    response = process_chat_message(ChatRequest(message="не видна моя ставка", session_id="v2-structured-actions"))
    assert response.message_id
    assert response.scenario_id == "bid.not_visible"
    # An unsigned guest may ask for help, but cannot execute a personal status action.
    assert {action.type for action in response.actions} == {"open_ticket"}


def test_unsigned_browser_context_cannot_fetch_personal_status():
    response = process_chat_message(
        ChatRequest(
            message="Проверить ставку",
            selected_action_id="bid.status.fetch",
            session_id="v2-untrusted-status",
            context=UserContext(is_authorized=True, user_id="browser-controlled", lot_id="123"),
        )
    )
    assert response.resolution == "clarified"
    assert response.data_freshness is None
    assert response.role == "guest"
    assert response.confidence_level == "low"


def test_trusted_status_action_uses_read_only_provider(monkeypatch):
    class FakeProvider:
        def fetch(self, kind, user_id, reference_id, access_token):
            assert (kind, user_id, reference_id) == ("bid", "user-42", "123")
            return StatusResult(True, kind, "accepted", "Ставка принята.", "2026-08-07T12:00:00+00:00")

    secret = "test-secret"
    monkeypatch.setattr(settings, "trusted_context_secret", secret)
    monkeypatch.setattr(settings, "internal_status_api_enabled", True)
    monkeypatch.setattr("backend.app.main.status_provider", FakeProvider())
    token = _token(
        secret,
        {
            "iss": "migtorg-site",
            "sub": "user-42",
            "exp": int(time.time()) + 60,
            "scopes": ["status:bid:read"],
        },
    )
    action = next(a for a in main._scenario_actions("bid.not_visible") if a.id == "bid.status.fetch")
    previous = ChatResponse(
        session_id="v2-trusted-status", message_id="v2-status-issued",
        answer="Статус можно проверить.", intent="bidding", scenario_id="bid.not_visible",
        role="authorized", needs_ticket=False, actions=[action],
    )
    main.logger.save_response_state(previous, {})
    response = process_chat_message(
        ChatRequest(
            message="Проверить ставку",
            selected_action_id="bid.status.fetch",
            trusted_context_token=token,
            conversation_turn_id="v2-status-issued",
            session_id="v2-trusted-status",
            context=UserContext(lot_id="123"),
        )
    )
    assert response.resolution == "status"
    assert response.data_freshness == "2026-08-07T12:00:00+00:00"
    assert "accepted" in response.answer


def test_trusted_context_rejects_expired_token():
    secret = "test-secret"
    token = _token(secret, {"iss": "migtorg-site", "sub": "u1", "exp": 1, "scopes": []})
    with pytest.raises(TrustedContextError, match="expired_token"):
        verify_trusted_context_token(token, secret, now=2)


def test_llm_prompt_redaction_removes_contact_and_long_identifier():
    redacted = _redact_for_llm(
        "Пишите user@example.com или +7 999 111-22-33, платеж 123456789, VIN XTA210990Y2765432"
    )
    assert "user@example.com" not in redacted
    assert "999 111" not in redacted
    assert "123456789" not in redacted
    assert "XTA210990Y2765432" not in redacted
    assert "[VIN]" in redacted


def test_gold_dataset_meets_release_gate():
    report = evaluate(__import__("pathlib").Path("tests/data/scenario_gold.jsonl"))
    assert report["scenario_accuracy"] >= 0.85
    assert report["confident_wrong_rate"] <= 0.02
    assert report["release_gate_passed"] is True
    assert report["production_gate_passed"] is True
    assert report["total"] >= 300


def test_expert_review_queue_is_not_loaded_as_active_knowledge():
    from backend.app.bot.scenario_engine import load_scenarios

    review_queue = json.loads(Path("knowledge/v2/review_queue.json").read_text(encoding="utf-8"))
    active_ids = {scenario.scenario_id for scenario in load_scenarios()}
    assert review_queue["publication_policy"] == "expert_approval_required"
    assert review_queue["records"]
    assert len({r["candidate_id"] for r in review_queue["records"]}) == len(review_queue["records"])
    for item in review_queue["records"]:
        assert item["status"] == "expert_review_required"
        assert item["publication_blockers"]
        assert item["proposed_scenario_id"] not in active_ids
