from backend.app.bot.pii_redaction import detected_pii_kinds, redact_for_external_llm


def test_external_llm_redaction_covers_supported_private_identifiers() -> None:
    raw = (
        "Клиент: Иванов Иван, телефон +7 999 123-45-67, ivan@example.com, @private_user. "
        "Адрес: Москва, улица Ленина, дом 10. VIN XTA210990Y2765432, "
        "госномер А123ВС77, договор AB-12345 и паспорт 4510 123456."
    )
    redacted = redact_for_external_llm(raw)
    assert "Иванов" not in redacted
    assert "999" not in redacted
    assert "example.com" not in redacted
    assert "private_user" not in redacted
    assert "Ленина" not in redacted
    assert "XTA210990Y2765432" not in redacted
    assert "А123ВС77" not in redacted
    assert "AB-12345" not in redacted
    assert "4510" not in redacted
    assert detected_pii_kinds(redacted) == ()


def test_external_llm_redaction_preserves_business_question() -> None:
    redacted = redact_for_external_llm("Когда продавец подтвердит передачу лота 123456?")
    assert "Когда продавец подтвердит передачу" in redacted
    assert "123456" not in redacted
