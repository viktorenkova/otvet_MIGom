from concurrent.futures import ThreadPoolExecutor
import json
import smtplib
from urllib.error import HTTPError

import pytest
from fastapi.testclient import TestClient

from backend.app import main
from backend.app.bot.dialog_logger import DialogLogger
from backend.app.config import Settings
from backend.app.delivery.outbox import drain_outbox
from backend.app.integrations import status_provider as status
from backend.app.models.ticket import Ticket


def provider(monkeypatch, body):
    class Response:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self, limit): return json.dumps(body).encode()
    monkeypatch.setattr(status.request, "urlopen", lambda *a, **kw: Response())
    return status.InternalApiStatusProvider(Settings(internal_status_api_url="https://example.invalid"))


@pytest.mark.parametrize("body", [[], None, {}, {"status":1}, {"status":"ok", "allowed_actions":"delete"},
    {"status":"ok", "allowed_actions":[1]}, {"status":"ok", "freshness":"today"},
    {"status":"ok", "freshness":"2026-09-04T10:00:00"}, {"status":"ok", "unexpected":"x"}])
def test_invalid_status_is_not_a_business_fact(monkeypatch, body):
    result = provider(monkeypatch, body).fetch("lot", "user", "123", "token")
    assert not result.success and result.error_code == "invalid_status_payload"


def test_receipt_time_is_not_freshness(monkeypatch):
    result = provider(monkeypatch, {"status":"active"}).fetch("tariff","user","user","token")
    assert result.success and result.received_at
    assert result.freshness is None


@pytest.mark.parametrize("value", ["unknown", "", "   "])
def test_missing_status_never_becomes_success(monkeypatch,value):
    assert not provider(monkeypatch,{"status":value}).fetch("lot","user","123","token").success


@pytest.mark.parametrize("code,expected", [(401,"forbidden"),(403,"forbidden"),(404,"not_found"),(503,"upstream_error")])
def test_http_error_categories(monkeypatch,code,expected):
    def fail(*a,**kw): raise HTTPError("https://example.invalid",code,"error",None,None)
    monkeypatch.setattr(status.request,"urlopen",fail)
    result = status.InternalApiStatusProvider(Settings(internal_status_api_url="https://example.invalid")).fetch("lot","user","123","token")
    assert result.error_code == expected


def ticket(key="same", description="Проверка"):
    return Ticket(topic="Оплата", description=description, contact="test@example.invalid", session_id="s", idempotency_key=key)


def test_concurrent_repeats_create_one_ticket_and_one_outbox_record(tmp_path):
    path = str(tmp_path / "db.sqlite3")
    workers = [DialogLogger(path) for _ in range(4)]
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda log: log.save_ticket(ticket(),queue_delivery=True),workers))
    assert len({t.id for t in results}) == 1
    with workers[0]._connect() as conn:
        assert conn.execute("SELECT COUNT(*) FROM tickets").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM ticket_outbox").fetchone()[0] == 1


def test_conflicting_key_does_not_modify_original(tmp_path):
    log=DialogLogger(str(tmp_path/"db.sqlite3"))
    saved=log.save_ticket(ticket())
    with pytest.raises(ValueError,match="idempotency_key_conflict"):
        log.save_ticket(ticket(description="другая ситуация"))
    assert log.get_ticket(saved.id)["description"] == "Проверка"


def test_competing_delivery_workers_send_once(tmp_path):
    path=str(tmp_path/"db.sqlite3")
    logs=[DialogLogger(path) for _ in range(4)]
    saved=logs[0].save_ticket(ticket(),queue_delivery=True)
    calls=[]
    class Provider:
        def deliver(self,ticket):
            calls.append(ticket.id)
            ticket.status="sent_email"
            return ticket
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda log:drain_outbox(log,Provider()),logs))
    assert calls == [saved.id]
    assert logs[0].ticket_delivery(saved.id)["state"] == "accepted"
    assert drain_outbox(logs[0],Provider())["sent"] == []


def test_uncertain_smtp_failure_is_not_automatically_retried(tmp_path):
    log=DialogLogger(str(tmp_path/"db.sqlite3"))
    saved=log.save_ticket(ticket(),queue_delivery=True)
    class Provider:
        def deliver(self,ticket): raise TimeoutError("acceptance unknown")
    assert drain_outbox(log,Provider())["unknown"] == [saved.id]
    assert log.claim_delivery() is None


def test_known_smtp_rejection_can_retry_same_ticket(tmp_path):
    log=DialogLogger(str(tmp_path/"db.sqlite3"))
    saved=log.save_ticket(ticket(),queue_delivery=True)
    class Provider:
        def deliver(self,ticket): raise smtplib.SMTPAuthenticationError(535,b"rejected")
    assert drain_outbox(log,Provider())["failed"] == [saved.id]
    assert log.claim_delivery() is None
    with log._connect() as conn:
        conn.execute("UPDATE ticket_outbox SET next_attempt_at='2000-01-01'")
    assert log.claim_delivery()["ticket"].id == saved.id


def test_abandoned_delivery_is_unknown_and_stale_owner_cannot_finish(tmp_path):
    log=DialogLogger(str(tmp_path/"db.sqlite3"))
    saved=log.save_ticket(ticket(),queue_delivery=True)
    claim=log.claim_delivery()
    with log._connect() as conn:
        conn.execute("UPDATE ticket_outbox SET claimed_at='2000-01-01'")
    assert log.claim_delivery() is None
    assert log.ticket_delivery(saved.id)["state"] == "unknown"
    assert not log.complete_delivery(saved.id,claim["token"],"accepted")


def test_http_creation_only_queues_and_repeats_same_id(tmp_path,monkeypatch):
    log=DialogLogger(str(tmp_path/"db.sqlite3"))
    monkeypatch.setattr(main,"logger",log)
    monkeypatch.setattr(main.settings,"ticket_email_enabled",True)
    def forbidden(*args): pytest.fail("SMTP must not run in ticket request")
    monkeypatch.setattr(main.email_ticket_provider,"deliver",forbidden)
    client=TestClient(main.app)
    payload={"session_id":"s","topic":"Оплата","description":"Проверка","contact":"test@example.invalid"}
    a=client.post("/api/chat/ticket",json=payload)
    b=client.post("/api/chat/ticket",json=payload)
    assert a.status_code == b.status_code == 200
    assert a.json()["ticket_id"] == b.json()["ticket_id"]
    assert a.json()["delivery"]["state"] == "pending"


def test_overload_rejects_before_processing_and_shares_capacity(tmp_path,monkeypatch):
    log=DialogLogger(str(tmp_path/"db.sqlite3"))
    monkeypatch.setattr(main,"logger",log)
    monkeypatch.setattr(main.settings,"chat_max_concurrency",1)
    token=log.acquire_processing_slot(1)
    other=DialogLogger(log.database_path)
    assert other.acquire_processing_slot(1) is None
    response=TestClient(main.app).post("/api/chat/message",json={"message":"тариф"})
    assert response.status_code == 503 and response.headers["retry-after"] == "2"
    log.release_processing_slot(token)
    assert other.acquire_processing_slot(1)


def test_external_status_respects_remaining_deadline(monkeypatch):
    import time
    from backend.app.bot.architecture_decision import decision_context
    token=decision_context.set({"deadline":time.monotonic()-1})
    try:
        result=provider(monkeypatch,{"status":"ok"}).fetch("lot","user","123","token")
        assert result.error_code == "deadline_exceeded"
    finally:
        decision_context.reset(token)


def test_migration_preserves_history_without_replaying_old_sends(tmp_path):
    path = str(tmp_path / "legacy.sqlite3")
    log = DialogLogger(path)
    sent = log.save_ticket(ticket("sent"))
    failed = log.save_ticket(ticket("failed"))
    saved = log.save_ticket(ticket("saved"))
    with log._connect() as conn:
        conn.execute("DELETE FROM ticket_outbox")
        conn.execute("UPDATE tickets SET status='sent_email' WHERE id=?", (sent.id,))
        conn.execute("UPDATE tickets SET status='delivery_failed' WHERE id=?", (failed.id,))
    migrated = DialogLogger(path)
    assert migrated.ticket_delivery(sent.id)["state"] == "accepted"
    assert migrated.ticket_delivery(failed.id)["state"] == "unknown"
    assert migrated.ticket_delivery(saved.id)["state"] == "saved"
    assert migrated.claim_delivery() is None
    assert migrated.delivery_summary()["manual_check_ticket_ids"] == [failed.id]
    assert migrated.get_ticket(saved.id)["description"] == "Проверка"


def test_widget_reports_delivery_state_without_promising_contact():
    script = TestClient(main.app).get("/widget/widget.js").text
    assert "deliveryLabels[data.delivery?.state]" in script
    assert "Почтовый сервер поддержки принял обращение" in script
    assert "Сотрудник проверит данные и свяжется" not in script


def test_wording_reserves_budget_before_provider_call(tmp_path, monkeypatch):
    from backend.app.bot import answer_generator as generator
    from backend.app.bot.architecture_decision import decision_context
    from backend.app.bot.knowledge_search import get_article_by_id
    from backend.app.models.llm import LLMResult
    log = DialogLogger(str(tmp_path / "budget.sqlite3"))
    ctx = {"logger": log}
    token = decision_context.set(ctx)
    monkeypatch.setattr("backend.app.integrations.llm_provider.estimate_cost", lambda *args: .7)
    calls = []
    class Provider:
        def generate(self, request):
            assert log.reserve_llm_budget(.7, 1, 1) is None
            calls.append(request)
            return LLMResult(text=request.fallback_text, provider="mock", model="mock", task_type=request.task_type)
    monkeypatch.setattr(generator, "build_llm_provider", lambda settings: Provider())
    try:
        generated = generator.generate_answer("как начать", "bidding", "guest",
            get_article_by_id("buyer.get_started", "guest"), False,
            settings=Settings(llm_enabled=True, llm_provider="mock", llm_daily_budget_usd=1, llm_dev_budget_usd=1))
        assert calls and generated.answer and ctx["answer_budget_reservation"]
        assert calls[0].fallback_model is None
    finally:
        if ctx.get("answer_budget_reservation"):
            log.release_llm_budget(ctx["answer_budget_reservation"])
        decision_context.reset(token)
    assert log.reserve_llm_budget(.7, 1, 1)


def test_deadline_after_search_prevents_ticket_and_confident_answer(tmp_path, monkeypatch):
    from backend.app.bot.knowledge_search import KnowledgeSearchResult, get_article_by_id
    from backend.app.models.chat import ChatRequest
    log = DialogLogger(str(tmp_path / "deadline.sqlite3"))
    monkeypatch.setattr(main, "logger", log)
    monkeypatch.setattr(main, "search_knowledge_match", lambda *a, **kw:
        KnowledgeSearchResult(get_article_by_id("buyer.get_started", "guest"), 300, "high"))
    monkeypatch.setattr("backend.app.bot.processing_budget.remaining", lambda *a: 0)
    response = main.process_chat_message(ChatRequest(message="как начать участвовать в торгах", session_id="deadline",
        consent_to_ticket=True, contact="test@example.invalid"))
    assert response.confidence_level == "low" and response.resolution == "clarified"
    assert response.ticket_id is None and response.scenario_id is None
    assert log.delivery_summary()["states"] == {}
