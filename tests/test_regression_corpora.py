from __future__ import annotations

import json

from backend.tools.audit_regression_corpora import (
    DEFAULT_LEAKAGE_BASELINE,
    leakage_snapshot,
    validate_locked_corpora,
)


def test_regression_corpora_are_immutable() -> None:
    assert validate_locked_corpora() == []


def test_no_new_regression_phrase_is_copied_into_routing_content() -> None:
    expected = json.loads(DEFAULT_LEAKAGE_BASELINE.read_text(encoding="utf-8"))
    assert leakage_snapshot() == expected
