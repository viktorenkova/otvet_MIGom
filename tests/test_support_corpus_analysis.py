from backend.tools.analyze_support_corpus import analyze, render_markdown


def _row(message_id: str, text: str, source: str = "messages.html") -> dict:
    return {
        "source": source,
        "source_message_id": message_id,
        "conversation_id": f"case-{message_id}",
        "message_kind": "candidate",
        "text_redacted": text,
    }


def test_support_analysis_keeps_topic_and_callback_as_separate_signals():
    report = analyze(
        [
            _row("1", "Выиграл лот, документы не получил, просит перезвонить"),
            _row("2", "Вопрос по оплате лота и комиссии"),
            _row("3", "Аккаунт заблокирован, не может войти", "messages2.html"),
        ]
    )
    themes = {item["theme"]: item for item in report["themes"]}
    assert themes["callback_requested"]["message_count"] == 1
    assert themes["won_lot_next_step"]["message_count"] == 1
    assert themes["documents_contract"]["message_count"] == 1
    assert themes["payment_lot"]["message_count"] == 1
    assert themes["commission"]["message_count"] == 1
    assert themes["account_blocked"]["message_count"] == 1
    assert report["publication_allowed"] is False
    assert {item["theme"] for item in report["scenario_backlog"]} >= {
        "documents_contract",
        "payment_lot",
        "account_blocked",
    }
    assert "Выиграл лот" not in render_markdown(report)
