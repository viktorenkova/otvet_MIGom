from concurrent.futures import ThreadPoolExecutor
import json
from types import SimpleNamespace

import pytest

from backend.app import main
from backend.app.bot import knowledge_search as ks
from backend.app.bot.architecture_decision import SelectorOutput, local_decision
from backend.app.bot.answer_contracts import get_answer_contract, verify_answer
from backend.app.bot.answer_generator import generate_answer
from backend.app.bot.dialog_logger import DialogLogger
from backend.app.bot.scenario_engine import load_scenarios
from backend.app.models.chat import ChatRequest, ChatResponse


@pytest.fixture
def isolated_log(tmp_path, monkeypatch):
    log = DialogLogger(str(tmp_path / "runtime.sqlite3"))
    monkeypatch.setattr(main, "logger", log)
    return log


def test_all_scenario_roles_survive_search_card_conversion():
    for scenario in load_scenarios():
        for role in ("guest", "authorized"):
            article = ks.get_article_by_id(scenario.scenario_id, role)
            assert bool(article) == (role in scenario.roles), (scenario.scenario_id, role)
            if article:
                assert set(article.roles) == set(scenario.roles)


def test_top10_never_contains_forbidden_role(monkeypatch):
    captured = set()
    class Index:
        def rank(self, message, ids, *args, **kwargs):
            captured.update(ids)
            return ()
    monkeypatch.setattr(ks, "_semantic_index", lambda: Index())
    ks.retrieve_knowledge_candidates("тариф", "guest")
    assert captured
    assert all("guest" in a.roles for a in ks.load_articles() if a.slug in captured)


def test_invented_action_never_reaches_business_logic(isolated_log, monkeypatch):
    def forbidden(*args):
        pytest.fail("invalid action reached pipeline")
    monkeypatch.setattr(main, "_process_chat_message", forbidden)
    response = main.process_chat_message(ChatRequest(message="проверить", selected_action_id="arbitrary"))
    assert response.resolution == "clarified"
    assert response.confidence_level == "low"
    assert response.scenario_id is None


def test_actions_bound_to_previous_response_and_session_across_workers(isolated_log, monkeypatch):
    scenario = next(s for s in load_scenarios() if "guest" in s.roles and any(
        not a.requires_auth for a in main._scenario_actions(s.scenario_id)))
    action = next(a for a in main._scenario_actions(scenario.scenario_id) if not a.requires_auth)
    prior = ChatResponse(session_id="one", message_id="previous", answer="ответ", intent=scenario.intent,
        role="guest", needs_ticket=False, actions=[action])
    isolated_log.save_response_state(prior, {})
    other_worker = DialogLogger(isolated_log.database_path)
    monkeypatch.setattr(main, "logger", other_worker)
    calls = []
    def run(request):
        calls.append(request.selected_action_id)
        return prior.model_copy(update={"message_id": "next", "actions": []})
    monkeypatch.setattr(main, "_process_chat_message", run)
    # A different session and a stale turn must both fail without executing it.
    assert main.process_chat_message(ChatRequest(message="x", session_id="two", selected_action_id=action.id)).resolution == "clarified"
    assert not calls
    result = main.process_chat_message(ChatRequest(message="x", session_id="one", selected_action_id=action.id, conversation_turn_id="previous"))
    assert calls == [action.id]
    assert result.state_version == 2
    main.process_chat_message(ChatRequest(message="x", session_id="one", selected_action_id=action.id, conversation_turn_id="previous"))
    assert calls == [action.id]


def test_clarification_reads_sqlite_instead_of_worker_cache(tmp_path):
    first = DialogLogger(str(tmp_path / "state.sqlite3"))
    second = DialogLogger(first.database_path)
    first.save_pending_clarification("session", [{"label": "Тариф", "article_id": "tariff.choose"}], original_message="доступ")
    assert second.get_pending_clarification_state("session")["original_message"] == "доступ"
    first.clear_pending_clarification("session")
    assert second.get_pending_clarification_state("session") is None


def test_tariff_status_uses_trusted_user_without_lot(isolated_log, monkeypatch):
    import base64
    import hashlib
    import hmac
    import time
    from backend.app.integrations.status_provider import StatusResult
    secret = "test-only-architecture-secret"
    monkeypatch.setattr(main.settings, "trusted_context_secret", secret)
    scenario = next(s for s in load_scenarios() if any(
        a.get("type") == "fetch_status" and a.get("payload", {}).get("kind") == "tariff" for a in s.actions))
    action = next(a for a in main._scenario_actions(scenario.scenario_id) if a.payload.get("kind") == "tariff")
    isolated_log.save_response_state(ChatResponse(session_id="tariff", message_id="prior", answer="тариф",
        intent=scenario.intent, role="authorized", needs_ticket=False, actions=[action]), {})
    payload = {"iss": main.settings.trusted_context_issuer, "sub": "confirmed-user",
               "exp": int(time.time()) + 60, "scopes": ["status:tariff:read"]}
    body = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=")
    signature = base64.urlsafe_b64encode(hmac.new(secret.encode(), body.encode(), hashlib.sha256).digest()).decode().rstrip("=")
    calls = []
    class Provider:
        def fetch(self, kind, user_id, reference_id, token):
            calls.append((kind, user_id, reference_id))
            return StatusResult(True, kind, "active", "", None)
    monkeypatch.setattr(main, "status_provider", Provider())
    response = main.process_chat_message(ChatRequest(message=action.label, session_id="tariff",
        selected_action_id=action.id, conversation_turn_id="prior", trusted_context_token=body + "." + signature))
    assert response.resolution == "status"
    assert calls == [("tariff", "confirmed-user", "confirmed-user")]


def test_model_failure_clarifies_without_independent_fallback(monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("query encoder unavailable")
    monkeypatch.setattr(ks, "retrieve_knowledge_candidates", fail)
    result = ks.search_knowledge_match("торги", "bidding", "guest")
    assert result.article is None
    assert result.confidence == "low"
    assert result.fallback_reason == "retrieval_or_scorer_failure"


def test_scorer_abstention_has_no_second_high_rule(monkeypatch):
    import backend.app.bot.architecture_decision as decision
    monkeypatch.setattr(decision, "get_pairwise_reranker", lambda: SimpleNamespace(available=False))
    result = local_decision("торги", [{"scenario_id": "buyer.get_started"}], "guest")
    assert result.article is None
    assert result.confidence != "high"


def test_fallback_cannot_authorize_itself():
    contract = get_answer_contract("buyer.get_started")
    invented = "Гарантируем выплату 999999 рублей завтра."
    verification = verify_answer(invented, invented, contract)
    assert not verification.passed
    assert verification.answer == contract.approved_template


def test_answer_boundary_rechecks_role_even_for_preselected_article():
    article = next(a for a in ks.load_articles() if a.roles == ("authorized",))
    answer = generate_answer("вопрос", article.intent, "guest", article, False)
    assert answer.verification_reason == "policy:scenario_access_denied"
    assert not answer.used_fact_ids


def test_published_gaps_are_executable():
    payload = json.loads((main.settings.knowledge_root / "v3_1/scenarios.json").read_text(encoding="utf-8"))
    for gap in payload["knowledge_gaps"]:
        article = ks.get_article_by_id(gap["scenario_id"], "authorized")
        answer = generate_answer(gap["question"], article.intent, "authorized", article, False)
        assert answer.answer == gap["safe_answer"]
        assert answer.verification_reason == "knowledge_gap:" + gap["gap_id"]


@pytest.mark.parametrize("value", [
    {"goal": "x", "scenario_id": "x", "missing_field": "states"},
    {"goal": "x", "scenario_id": None, "missing_field": None},
    {"goal": "x", "scenario_id": "x", "missing_field": None, "business_fact": "invented"},
])
def test_selector_rejects_invalid_schema(value):
    with pytest.raises(ValueError):
        SelectorOutput.model_validate(value)


def test_parallel_llm_calls_cannot_over_reserve_budget(tmp_path):
    path = str(tmp_path / "budget.sqlite3")
    workers = [DialogLogger(path), DialogLogger(path)]
    with ThreadPoolExecutor(max_workers=2) as pool:
        reservations = list(pool.map(lambda log: log.reserve_llm_budget(.7, 1, 1), workers))
    assert sum(r is not None for r in reservations) == 1


@pytest.mark.parametrize("payload,accepted", [
    ({"goal": "начать покупку", "scenario_id": "buyer.get_started", "missing_field": None}, True),
    ({"goal": "x", "scenario_id": "invented.id", "missing_field": None}, False),
    ({"goal": "x", "scenario_id": "buyer.get_started", "missing_field": None, "fact": "x"}, False),
])
def test_constrained_llm_selection_and_local_fallback(isolated_log, monkeypatch, payload, accepted):
    import time
    from backend.app.bot.architecture_decision import decision_context, llm_decision, clarification
    from backend.app.config import Settings
    from backend.app.models.llm import LLMResult
    bounded_settings = []
    class Provider:
        def generate(self, request):
            assert "person@example.com" not in request.prompt
            assert request.fallback_model is None
            return LLMResult(text=json.dumps(payload), provider="qwen", model="configured",
                             task_type="scenario_selection")
    def provider(settings):
        bounded_settings.append(settings)
        return Provider()
    monkeypatch.setattr("backend.app.integrations.llm_provider.build_llm_provider", provider)
    settings = Settings(llm_provider="qwen", llm_primary_model="configured",
                        llm_input_cost_per_million_usd=1, llm_output_cost_per_million_usd=1)
    local = clarification("local")
    token = decision_context.set({"logger": isolated_log, "session_id": "llm",
                                  "deadline": time.monotonic() + 5})
    try:
        answer = llm_decision("как начать person@example.com", [{"scenario_id": "buyer.get_started"}], "guest", settings, local)
        assert bool(answer.article) == accepted
        if not accepted:
            assert answer is local
        assert 0 < bounded_settings[0].llm_request_timeout_seconds <= 4
        with isolated_log._connect() as conn:
            assert conn.execute("SELECT COUNT(*) FROM llm_budget_reservations").fetchone()[0] == 0
    finally:
        decision_context.reset(token)
