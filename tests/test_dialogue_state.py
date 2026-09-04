"""Mechanism regressions, not an independent semantic acceptance corpus."""
from concurrent.futures import ThreadPoolExecutor
import json
from uuid import uuid4

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from backend.app import main
from backend.app.bot.architecture_decision import decision_context
from backend.app.bot.dialog_logger import DialogLogger
from backend.app.bot.dialogue_understanding import prepare_turn, understand
from backend.app.models.chat import ChatRequest, ChatResponse
from backend.app.models.dialogue import DialogueState


def active(message="как проверить статус лота 123", **updates):
    return DialogueState(active=understand(message), status="answered", **updates)


def test_subject_question_and_callback_are_separate():
    parsed = understand("выиграл лот, что дальше, перезвоните мне")
    assert "перезвон" not in parsed.goal
    assert parsed.secondary_requests == ["перезвоните мне"]
    assert not understand("не перезвоните мне").secondary_requests


def test_independent_questions_are_retained_in_order():
    first = prepare_turn("что дает премиальный тариф? Как восстановить пароль?", DialogueState(), "guest")
    assert first.understanding.goal == "что дает премиальный тариф?"
    first.state.status = "answered"
    next_turn = prepare_turn("следующий вопрос", first.state, "guest")
    assert next_turn.transition == "next"
    assert next_turn.search_message == "Как восстановить пароль?"
    assert next_turn.state.pending_requests == []


@pytest.mark.parametrize("text", ["не 123, а 456", "лот 456", "лот №456"])
def test_entity_correction_keeps_task_and_replaces_value(text):
    old = active()
    turn = prepare_turn(text, old, "authorized")
    assert turn.transition == "correct"
    assert turn.understanding.entities["lot_id"] == "456"
    assert turn.understanding.goal == old.active.goal.replace("123", "456")
    assert "123" not in turn.search_message
    assert old.active.entities["lot_id"] == "123"


def test_negated_lot_is_removed_without_guess():
    turn = prepare_turn("лот не 123", active(), "authorized")
    assert "lot_id" not in turn.understanding.entities
    assert turn.service_reply == "clarify_entity"


def test_bare_identifier_resumes_only_pending_action():
    old = active()
    old.status, old.expected_field, old.pending_action_id = "clarifying", "lot_id", "issued-status"
    turn = prepare_turn("456", old, "authorized")
    assert turn.resume_action_id == "issued-status"
    assert turn.search_message == old.active.goal.replace("123", "456")
    assert prepare_turn("456", active(), "authorized").resume_action_id is None


def test_short_followup_uses_object_without_concatenating_turns():
    old = active("что дает премиальный тариф", active_scenario_id="tariff.premium")
    turn = prepare_turn("а как его подключить?", old, "guest")
    assert turn.transition == "continue"
    assert "тариф" in turn.search_message
    assert "что дает" not in turn.search_message


def test_topic_switch_clears_entities_and_expected_action():
    old = active()
    old.expected_field, old.pending_action_id = "lot_id", "issued-status"
    turn = prepare_turn("теперь как восстановить пароль", old, "guest")
    assert turn.transition == "switch"
    assert "lot_id" not in turn.understanding.entities
    assert turn.state.pending_action_id is None


def test_unsuccessful_clarification_offers_manual_help():
    old = active("нужен возврат")
    old.status = "clarifying"
    turn = prepare_turn("не знаю", old, "guest")
    assert turn.service_reply == "manual_help"
    assert turn.understanding.goal == old.active.goal


@pytest.mark.parametrize("text", [
    "Продавец неделю не подтверждает передачу лота.",
    "Как это работает? Я оплачиваю доступ и потом делаю ставки?",
    "Вы издеваетесь? Ставка опять не проходит за минуту до конца!",
])
def test_related_or_single_question_preserves_search_input(text):
    turn = prepare_turn(text, DialogueState(), "guest")
    assert turn.search_message == text
    assert not turn.state.pending_requests


def test_tariff_variant_remains_bound_in_followup():
    turn = prepare_turn("а сколько он стоит?", active("что дает премиальный тариф"), "guest")
    assert turn.understanding.entities["tariff_type"] == "Премиум"
    assert "Премиум" in turn.search_message


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    log = DialogLogger(str(tmp_path / "dialogue.sqlite3"))
    monkeypatch.setattr(main, "logger", log)
    monkeypatch.setattr(main.settings, "dialogue_state_enabled", True)
    captured = []
    def pipeline(request):
        captured.append(decision_context.get()["dialogue"])
        return ChatResponse(session_id=request.session_id, message_id=str(uuid4()),
            answer="Подтверждённый ответ", intent="tariffs", role="guest", needs_ticket=False,
            scenario_id="tariff.premium", resolution="answered")
    monkeypatch.setattr(main, "_process_chat_message", pipeline)
    return log, captured


def test_restart_reads_state_and_links_turn(runtime, monkeypatch):
    log, calls = runtime
    first = main.process_chat_message(ChatRequest(message="что дает премиальный тариф", session_id="s"))
    worker = DialogLogger(log.database_path)
    monkeypatch.setattr(main, "logger", worker)
    second = main.process_chat_message(ChatRequest(message="а как его подключить?", session_id="s",
        conversation_turn_id=first.message_id, state_version=first.state_version))
    assert calls[-1]["transition"] == "continue"
    assert second.state_version == 2
    assert worker.load_dialogue_state("s").previous_message_id == second.message_id
    assert "active_task" in second.used_context


@pytest.mark.parametrize("link", [{"conversation_turn_id": "stale"}, {"state_version": 99}])
def test_stale_free_turn_rejected_without_advancing_state(runtime, link):
    log, calls = runtime
    main.process_chat_message(ChatRequest(message="тариф", session_id="s"))
    client = TestClient(main.app)
    result = client.post("/api/chat/message", json={"message": "а дальше?", "session_id": "s", **link})
    assert result.status_code == 409
    assert log.load_dialogue_state("s").version == 1
    assert len(calls) == 1


def test_competing_workers_only_one_owns_turn(tmp_path):
    path = str(tmp_path / "workers.sqlite3")
    workers = [DialogLogger(path) for _ in range(4)]
    with ThreadPoolExecutor(max_workers=4) as pool:
        tokens = list(pool.map(lambda worker: worker.acquire_dialogue_turn("s"), workers))
    assert sum(token is not None for token in tokens) == 1
    workers[0].release_dialogue_turn("s", "wrong-token")
    assert workers[0].acquire_dialogue_turn("s") is None
    workers[0].release_dialogue_turn("s", next(token for token in tokens if token))
    assert workers[1].acquire_dialogue_turn("s")


def test_in_progress_request_returns_conflict(runtime):
    log, calls = runtime
    lease = log.acquire_dialogue_turn("s")
    with pytest.raises(HTTPException) as exc:
        main.process_chat_message(ChatRequest(message="тариф", session_id="s"))
    assert exc.value.status_code == 409
    assert not calls
    log.release_dialogue_turn("s", lease)


def test_pipeline_exception_releases_lease(runtime, monkeypatch):
    log, _ = runtime
    def fail(_):
        raise ValueError("test pipeline failure")
    monkeypatch.setattr(main, "_process_chat_message", fail)
    with pytest.raises(ValueError):
        main.process_chat_message(ChatRequest(message="тариф", session_id="s"))
    assert log.acquire_dialogue_turn("s")


def test_expired_owner_cannot_overwrite_new_state(tmp_path):
    log = DialogLogger(str(tmp_path / "lease.sqlite3"))
    lease = log.acquire_dialogue_turn("s")
    with log._connect() as conn:
        conn.execute("UPDATE dialogue_leases SET expires_at = '2000-01-01' WHERE session_id = 's'")
    assert log.acquire_dialogue_turn("s") != lease
    response = ChatResponse(session_id="s", message_id="1", answer="ответ", intent="unknown", role="guest", needs_ticket=False)
    with pytest.raises(RuntimeError, match="dialogue_lease_lost"):
        log.save_response_state(response, {}, prepare_turn("тариф", DialogueState(), "guest"), lease)
    assert log.get_response_state("s") is None


def test_identity_change_does_not_inherit_other_users_entities(runtime):
    log, calls = runtime
    state = active(subject="authorized:another-user")
    with log._connect() as conn:
        conn.execute("INSERT INTO dialogue_states VALUES (?, ?)", ("s", state.model_dump_json()))
    main.process_chat_message(ChatRequest(message="а дальше?", session_id="s"))
    assert calls[-1]["transition"] == "new"
    assert not calls[-1]["understanding"]["entities"]


def test_toggle_does_not_resurrect_old_context(runtime, monkeypatch):
    log, calls = runtime
    main.process_chat_message(ChatRequest(message="что дает премиальный тариф", session_id="s"))
    # A legacy turn advances response state but intentionally leaves dialogue data untouched.
    log.save_response_state(ChatResponse(session_id="s", message_id="legacy", answer="ответ",
        intent="support", role="guest", needs_ticket=False), {})
    main.process_chat_message(ChatRequest(message="а дальше?", session_id="s"))
    assert calls[-1]["transition"] == "new"


def test_pending_requests_and_state_are_written_together(runtime):
    log, _ = runtime
    result = main.process_chat_message(ChatRequest(message="что дает премиальный тариф, перезвоните мне", session_id="s"))
    assert result.pending_requests == ["перезвоните мне"]
    assert log.load_dialogue_state("s").pending_requests == result.pending_requests
    assert log.get_response_state("s")["response"]["state_version"] == result.state_version


def test_missing_status_identifier_is_collected_across_workers(tmp_path, monkeypatch):
    import base64
    import hashlib
    import hmac
    import time
    from backend.app.bot.scenario_engine import load_scenarios
    from backend.app.integrations.status_provider import StatusResult
    log = DialogLogger(str(tmp_path / "status.sqlite3"))
    monkeypatch.setattr(main, "logger", log)
    monkeypatch.setattr(main.settings, "dialogue_state_enabled", True)
    secret = "dialogue-test-only-secret"
    monkeypatch.setattr(main.settings, "trusted_context_secret", secret)
    body = base64.urlsafe_b64encode(json.dumps({"iss": main.settings.trusted_context_issuer,
        "sub": "user", "exp": int(time.time()) + 60, "scopes": ["status:read"]}).encode()).decode().rstrip("=")
    signature = base64.urlsafe_b64encode(hmac.new(secret.encode(), body.encode(), hashlib.sha256).digest()).decode().rstrip("=")
    token = body + "." + signature
    action = next(a for s in load_scenarios() for a in main._scenario_actions(s.scenario_id)
                  if a.type == "fetch_status" and a.payload.get("kind") != "tariff")
    log.save_response_state(ChatResponse(session_id="s", message_id="prior", answer="ответ", intent="bidding",
        role="authorized", needs_ticket=False, actions=[action]), {})
    calls = []
    class Provider:
        def fetch(self, kind, user_id, reference_id, token):
            calls.append((user_id, reference_id))
            return StatusResult(True, kind, "test-status", "", None)
    monkeypatch.setattr(main, "status_provider", Provider())
    response = main.process_chat_message(ChatRequest(message=action.label, session_id="s",
        selected_action_id=action.id, trusted_context_token=token))
    assert response.resolution == "clarified"
    assert log.load_dialogue_state("s").expected_field == "lot_id"
    assert not calls
    monkeypatch.setattr(main, "logger", DialogLogger(log.database_path))
    second = main.process_chat_message(ChatRequest(message="456", session_id="s", trusted_context_token=token,
        conversation_turn_id=response.message_id, state_version=response.state_version))
    assert second.resolution == "status"
    assert calls == [("user", "456")]


def test_guest_cannot_choose_closed_scenario_from_previous_options():
    from backend.app.bot.scenario_engine import load_scenarios
    from backend.app.bot.dialogue_understanding import resolve_choice
    closed = next(s for s in load_scenarios() if "guest" not in s.roles)
    assert resolve_choice(closed.title, [{"article_id": closed.scenario_id, "label": closed.title}], "guest") is None


def test_repeat_keeps_current_task():
    old = active()
    turn = prepare_turn(old.active.goal, old, "authorized")
    assert turn.transition == "repeat"
    assert turn.understanding.entities == old.active.entities


def test_safety_response_resets_active_context(runtime, monkeypatch):
    log, _ = runtime
    def safety(request):
        return ChatResponse(session_id=request.session_id, message_id="safety", answer="Ограничение",
            intent="safety", role="guest", needs_ticket=False)
    monkeypatch.setattr(main, "_process_chat_message", safety)
    main.process_chat_message(ChatRequest(message="ситуация", session_id="s"))
    assert log.load_dialogue_state("s").status == "idle"


def test_existing_ticket_category_flow_with_structured_state(tmp_path, monkeypatch):
    # Direct calls bypass lifespan; use the same startup warmup as HTTP workers.
    main.warm_knowledge_indexes()
    log = DialogLogger(str(tmp_path / "ticket.sqlite3"))
    monkeypatch.setattr(main, "logger", log)
    monkeypatch.setattr(main.settings, "dialogue_state_enabled", True)
    # Existing case: tests/test_manual_review_regressions.py.
    first = main.process_chat_message(ChatRequest(message="создать обращение", session_id="s"))
    assert first.resolution == "clarified"
    result = main.process_chat_message(ChatRequest(message="Оплата или возврат", session_id="s"))
    assert result.intent == "payment"
    assert result.needs_ticket
    assert "Нажмите «Создать обращение» ниже" in result.answer
    assert result.ticket_id is None


def test_unsuccessful_clarification_preserves_history_without_creating_ticket(tmp_path, monkeypatch):
    log = DialogLogger(str(tmp_path / "manual.sqlite3"))
    monkeypatch.setattr(main, "logger", log)
    monkeypatch.setattr(main.settings, "dialogue_state_enabled", True)
    state = active("нужен возврат")
    state.status = "clarifying"
    with log._connect() as conn:
        conn.execute("INSERT INTO dialogue_states VALUES (?, ?)", ("s", state.model_dump_json()))
    result = main.process_chat_message(ChatRequest(message="не знаю", session_id="s"))
    assert result.needs_ticket and result.ticket_id is None
    assert "ещё не отправлено" in result.answer
    assert log.load_dialogue_state("s").status == "manual_help"
    assert log.get_history("s")[-1]["message"] == "не знаю"
