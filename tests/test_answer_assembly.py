"""Scope/provenance and mutation checks; these do not certify expert answer quality."""
from dataclasses import replace
from copy import deepcopy
import json
from pathlib import Path

import pytest

from backend.app.bot import answer_assembly as assembly
from backend.app.bot.answer_generator import generate_answer
from backend.app.bot.knowledge_search import get_article_by_id
from backend.app.bot.scenario_engine import load_scenarios
from backend.app.config import Settings


def plan(message, scenario="refund.application"):
    return assembly.build_answer_plan(message, scenario, "guest")


def test_one_time_refund_keeps_amount_and_employee_confirmation():
    result = plan("где кнопка возврата разового тарифа")
    assert result.profile == "one_time"
    assert "5 000" in result.text and "подтверждает сотрудник" in result.text
    assert "Премиум" not in result.text
    assert result.documents == "omit"
    assert result.required_fact_ids == tuple(f"refund.application.fact.{i:03d}" for i in (1,6,7))


def test_premium_refund_keeps_applicant_and_details_together():
    result = plan("кто может подать заявление на возврат Премиум")
    assert result.profile == "premium"
    assert "владелец" in result.text and "реквизитами" in result.text
    assert "Разов" not in result.text and "5 000" not in result.text
    assert result.documents == "keep"


def test_multiple_financial_subjects_do_not_select_one_silently():
    result = plan("Как вернуть Разовый тариф и обеспечительный платеж Премиум?")
    assert result.profile == "published_template"


def test_price_does_not_borrow_other_amounts():
    result = plan("сколько стоит Премиум", "tariff.premium")
    assert result.profile == "payment"
    assert "один раз" in result.text and "Актуальная стоимость" in result.text
    assert not any(char.isdigit() for char in result.text)


def test_duration_keeps_termination_condition():
    result = plan("на сколько действует Премиум", "tariff.premium")
    assert result.profile == "duration"
    assert "пока не расторгнут" in result.text
    assert result.required_fact_ids == ("tariff.premium.fact.001", "tariff.premium.fact.003")


@pytest.mark.parametrize("mutation", [
    lambda text: text.replace("5 000", "10 000"),
    lambda text: text.replace("не заменяет", "заменяет"),
    lambda text: text.replace("должна быть кратна", "может быть любой, например"),
    lambda text: text.replace("сотрудник", "бот"),
    lambda text: text + " Деньги гарантированно вернут завтра.",
    lambda text: " ".join(reversed(text.split())),
])
def test_edits_cannot_pass_by_lexical_overlap(mutation):
    expected = plan("как вернуть разовый тариф")
    candidate = mutation(expected.text)
    assert candidate != expected.text
    assert not assembly.verify_plan_text(candidate, expected, expected)


def test_caller_cannot_approve_own_fragments():
    expected = plan("как вернуть разовый тариф")
    fake = replace(expected.fragments[0], text="Возврат уже доставлен.")
    fabricated = replace(expected, fragments=(fake,))
    assert not assembly.verify_plan_text(fabricated.text, fabricated, expected)


def test_every_profile_fragment_resolves_to_exact_published_source():
    samples = [("refund.application", q) for q in ("разовый", "премиум", "комиссия")]
    samples += [("tariff.premium", "сколько стоит"), ("tariff.premium", "срок"), ("tariff.one_time", "цена")]
    for scenario_id, message in samples:
        result = plan(message, scenario_id)
        for fragment in result.fragments:
            value = json.loads(Path(fragment.source_path).read_text(encoding="utf-8"))
            for key in fragment.source_pointer.split("/")[1:]:
                value = value[int(key)] if isinstance(value, list) else value[key]
            assert value == fragment.text
            assert len(fragment.source_sha256) == 64
            assert fragment.attribution
            assert fragment.primary_evidence_verified is False


def test_source_revision_change_disables_old_profile(monkeypatch):
    policy = deepcopy(assembly.assembly_policy())
    policy["scenarios"]["refund.application"]["source_version"] = "outdated"
    monkeypatch.setattr(assembly, "assembly_policy", lambda: policy)
    assert plan("разовый").reason == "profile_source_version_changed"


def test_missing_fact_does_not_omit_its_condition(monkeypatch):
    policy = deepcopy(assembly.assembly_policy())
    policy["scenarios"]["refund.application"]["profiles"][0]["facts"].append(999)
    monkeypatch.setattr(assembly, "assembly_policy", lambda: policy)
    assert plan("разовый").reason == "profile_fact_contract_mismatch"


def test_whole_template_fallback_covers_all_active_scenarios_and_roles():
    for scenario in load_scenarios():
        for role in ("guest", "authorized"):
            result = assembly.build_answer_plan("", scenario.scenario_id, role)
            assert bool(result) == (role in scenario.roles)
            if result:
                assert result.profile == "published_template"
                assert result.text == assembly.get_answer_contract(scenario.scenario_id).approved_template


def test_generator_uses_exact_assembly_and_never_calls_wording_llm(monkeypatch):
    import backend.app.bot.answer_generator as generator
    def forbidden(*args):
        pytest.fail("wording provider must not run during extractive assembly")
    monkeypatch.setattr(generator, "build_llm_provider", forbidden)
    article = get_article_by_id("refund.application", "guest")
    answer = generate_answer("вернуть разовый тариф", article.intent, "guest", article, True,
        settings=Settings(answer_assembly_enabled=True, llm_enabled=True, architecture_experiment=False))
    assert answer.answer == plan("вернуть разовый тариф").text
    assert answer.document_policy == "omit"
    assert answer.llm_result is None


def test_unavailable_policy_fails_without_unverified_legacy_override(monkeypatch):
    def failed():
        raise ValueError("bad policy")
    monkeypatch.setattr(assembly, "assembly_policy", failed)
    article = get_article_by_id("refund.application", "guest")
    answer = generate_answer("вернуть разовый тариф", article.intent, "guest", article, True,
        settings=Settings(answer_assembly_enabled=True))
    assert not answer.verification_passed
    assert answer.document_policy == "omit"
    assert "Не удалось проверить" in answer.answer


def test_runtime_documents_and_offered_ticket_match_selected_refund(tmp_path, monkeypatch):
    from backend.app import main
    from backend.app.bot.dialog_logger import DialogLogger
    from backend.app.bot.knowledge_search import KnowledgeSearchResult
    from fastapi.testclient import TestClient
    log = DialogLogger(str(tmp_path / "http.sqlite3"))
    monkeypatch.setattr(main, "logger", log)
    monkeypatch.setattr(main.settings, "answer_assembly_enabled", True)
    article = get_article_by_id("refund.application", "guest")
    monkeypatch.setattr(main, "search_knowledge_match", lambda *a, **kw: KnowledgeSearchResult(article, 300, "high"))
    response = TestClient(main.app).post("/api/chat/message", json={"message": "форма возврата разового тарифа", "session_id": "one"})
    assert response.status_code == 200
    data = response.json()
    assert data["template_links"] == []
    assert data["attachments"] == []
    assert "Премиум" not in data["answer"]
    assert data["ticket_id"] is None
    if data["needs_ticket"]:
        assert data["action_result"] == {"offered": True, "created": False, "delivery": "not_requested"}
    with log._connect() as conn:
        trace = json.loads(conn.execute("SELECT trace_json FROM decision_traces WHERE message_id = ?", (data["message_id"],)).fetchone()[0])
    assert trace["answer_plan"]["profile"] == "one_time"
    assert trace["answer_plan"]["fragments"]


def test_legacy_template_migration_preserves_text_and_does_not_approve_it():
    import ast
    import subprocess
    from backend.app.bot.runtime_templates import runtime_templates
    original = subprocess.check_output(["git", "show", "4b9bbd0:backend/app/bot/answer_generator.py"]).decode("utf-8")
    tree = ast.parse(original)
    defaults = next(n for n in tree.body if isinstance(n, ast.AnnAssign) and getattr(n.target,"id","")=="DEFAULT_ANSWERS")
    registry = runtime_templates()
    assert registry["defaults"] == ast.literal_eval(defaults.value)
    original_texts = {n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value,str)}
    assert len(registry["exceptions"]) == 11
    for item in registry["exceptions"].values():
        assert item["text"] in original_texts
        assert item["review_status"] == "pending"
