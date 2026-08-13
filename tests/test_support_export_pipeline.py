import json

from backend.tools.knowledge_pipeline import (
    _privacy_audit,
    _ingestion_summary,
    _write_normalized,
    clean_support_text,
    load_messages,
    normalize_html_messages,
    redact_freeform_entities,
    redact,
)


def _word_telegram_fixture() -> str:
    return """
    <html><body><div class="WordSection1">
      <div id="message-1"><div><p>Support group created group</p></div></div>
      <div id="message100">
        <div><div><div><p>АК</p></div></div></div>
        <div>
          <div title="01.04.2026 10:00:00 UTC+03:00"><p>10:00</p></div>
          <div><p>Анна Консультант</p></div>
          <div><p>Клиент: Иван Иванов, +7 999 123-45-67. Лот 123456, как получить договор? https://example.test/a</p></div>
        </div>
      </div>
      <div id="message101">
        <div>
          <div title="01.04.2026 10:01:00 UTC+03:00"><p>10:01</p></div>
          <div><p>Анна Консультант, ставка не отображается в карточке лота.</p></div>
        </div>
      </div>
    </div></body></html>
    """


def test_word_saved_telegram_html_is_parsed_and_redacted(tmp_path):
    source = tmp_path / "messages.html"
    source.write_text(_word_telegram_fixture(), encoding="utf-16")

    normalized = normalize_html_messages(source)
    candidates = load_messages(source)

    assert len(normalized) == 3
    assert len(candidates) == 2
    assert candidates[0].source_message_id == "message100"
    assert candidates[0].conversation_id == "messages:message100"
    assert candidates[1].conversation_id == "messages:message100"
    assert candidates[0].speaker_key == candidates[1].speaker_key == "speaker-001"
    assert candidates[0].created_at == "2026-04-01T10:00:00"
    assert "contract.receive" not in candidates[0].text
    assert "Иван" not in candidates[0].text
    assert "Анна Консультант" not in candidates[1].text
    assert "example.test" not in candidates[0].text
    assert "123456" not in candidates[0].text
    assert "[phone]" not in candidates[0].text
    assert "[name]" not in candidates[0].text
    assert "[url]" in candidates[0].text
    assert set(candidates[1].categories) >= {"lot_vehicle_info", "bid_auction"}
    assert _privacy_audit(candidates) == {
        "email_matches": 0,
        "phone_matches": 0,
        "url_matches": 0,
        "telegram_handle_matches": 0,
        "obfuscated_contact_matches": 0,
        "labelled_name_matches": 0,
        "long_identifier_matches": 0,
    }
    summary = _ingestion_summary(normalized)
    assert summary["source_messages"] == 3
    assert summary["candidate_messages"] == 2
    assert summary["message_kinds"] == {"candidate": 2, "system": 1}
    assert summary["date_start"] == "2026-04-01T10:00:00"
    assert summary["date_end"] == "2026-04-01T10:01:00"


def test_normalized_export_does_not_write_author_names(tmp_path):
    source = tmp_path / "messages.html"
    output = tmp_path / "normalized.jsonl"
    source.write_text(_word_telegram_fixture(), encoding="utf-16")

    normalized = normalize_html_messages(source)
    _write_normalized(normalized, output)
    payload = output.read_text(encoding="utf-8")

    assert "Анна Консультант" not in payload
    assert "Иван Иванов" not in payload
    assert all(json.loads(line)["speaker_key"] != "Анна Консультант" for line in payload.splitlines())


def test_redact_covers_support_export_identifiers():
    value = redact(
        "email test@example.com, @username, www.example.test, "
        "обращение 1234567, телефон 89991234567"
    )
    assert value == "email [email], [telegram], [url] обращение [identifier], телефон [phone]"


def test_internal_routing_boilerplate_is_removed_from_analysis_text():
    value = clean_support_text(
        "Для Алексея: [phone] Иван соед., вопрос по выигранному лоту, "
        "попросил связаться с менеджером"
    )
    assert value == "вопрос по выигранному лоту"


def test_name_after_masked_phone_is_removed():
    value = clean_support_text(
        "Обращение действующего клиента: [phone] Иван Иванов просит вернуть средства"
    )
    assert "Иван" not in value
    assert value.endswith("просит вернуть средства")


def test_freeform_names_are_removed_but_public_product_name_is_kept():
    value = redact_freeform_entities(
        "Нужна доверенность от Илоны Петровой для получения в Мигторг. Следующий шаг неизвестен."
    )
    assert "Илоны" not in value
    assert "Петровой" not in value
    assert "Мигторг" in value
