from backend.tools.run_local_shadow import expand_records, pii_leaks, redact_for_shadow, run_shadow, summarize


def test_shadow_redaction_removes_contacts_and_identifiers() -> None:
    value = redact_for_shadow(
        "Клиент: Иван Иванов, +7 999 123-45-67, user@example.com, лот 1234567"
    )
    assert "Иван Иванов" not in value
    assert "user@example.com" not in value
    assert "123-45-67" not in value
    assert "1234567" not in value
    assert pii_leaks(value) == []


def test_expand_records_creates_exact_replay_population() -> None:
    source = [{"id": "one", "text": "вопрос", "role": "", "source": "input.json"}]
    expanded = expand_records(source, 1000)
    assert len(expanded) == 1000
    assert expanded[0]["event_id"] == "shadow-000001"
    assert expanded[-1]["event_id"] == "shadow-001000"


def test_local_gate_requires_volume_transport_and_privacy() -> None:
    decision = {
        "scenario_id": "support.contact",
        "intent": "support",
        "confidence": "high",
        "fallback_reason": "",
        "clarifies": False,
        "needs_ticket": False,
    }
    row = {
        "baseline": decision,
        "candidate": decision,
        "baseline_error": "",
        "candidate_error": "",
        "pii_leaks": [],
        "baseline_latency_ms": 1.0,
        "candidate_latency_ms": 1.1,
    }
    report = summarize([row] * 1000, [], 1000)
    assert report["local_plumbing_gate"]["passed"] is True
    assert "expert answer quality on real shadow traffic" in report["not_measured"]


def test_mixed_role_replay_alternates_missing_roles(monkeypatch) -> None:
    decision = type("Decision", (), {
        "article": None,
        "confidence": "low",
        "fallback_reason": "none",
        "clarifying_question": "Уточните",
    })()
    monkeypatch.setattr("backend.tools.run_local_shadow.classify_intent", lambda _: "unknown")
    monkeypatch.setattr("backend.tools.run_local_shadow.HybridSearchProvider.search", lambda *args: decision)
    monkeypatch.setattr("backend.tools.run_local_shadow.search_knowledge_match", lambda *args: decision)
    records = [
        {"id": str(index), "text": "вопрос", "role": "", "source": "input.json"}
        for index in range(4)
    ]
    assert [item["role"] for item in run_shadow(records, "mixed")] == [
        "guest", "authorized", "guest", "authorized"
    ]
