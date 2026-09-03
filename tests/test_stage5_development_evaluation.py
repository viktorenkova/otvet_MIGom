from backend.app.bot.trusted_context import verify_trusted_context_token
from backend.tools.evaluate_stage5_development import (
    LOCAL_TRUSTED_CONTEXT_SECRET,
    _actual_action,
    _metric,
    _trusted_context_token,
    summarize,
)


def _result(expected: str, actual: str, handoff: bool | None = None) -> dict:
    return {
        "checks": {
            "transport_ok": True,
            "action_ok": expected == actual,
            "forbidden_content_ok": True,
            "support_handoff_ok": handoff,
        },
        "diagnostics": {
            "expected_action": expected,
            "actual_action": actual,
            "confidence_level": "high",
            "required_information_overlap_pct": 50.0,
        },
    }


def test_actual_action_prioritizes_support_and_clarification() -> None:
    assert _actual_action({"action": "answer", "actions": [{"type": "open_ticket"}]}) == "answer"
    assert _actual_action({"action": "create_ticket", "resolution": "escalated"}) == "support"
    assert _actual_action({"action": "answer", "scenario_id": "support.callback"}) == "support"
    assert _actual_action({"action": "clarify", "resolution": "clarified"}) == "clarify"
    assert _actual_action({"action": "answer", "resolution": "answered"}) == "answer"


def test_metric_excludes_non_applicable_checks() -> None:
    assert _metric([_result("answer", "answer", None), _result("support", "answer", False)], "support_handoff_ok") == {
        "passed": 0,
        "total": 1,
        "rate_pct": 0.0,
        "wilson_95_lower_pct": 0.0,
    }


def test_summary_marks_development_limitations() -> None:
    source = {"single_count": 1, "dialogue_count": 1, "dialogue_turn_count": 1}
    report = summarize(source, [_result("answer", "answer"), _result("support", "answer", False)])
    assert report["status"] == "development_diagnostic_not_blind_release_gate"
    assert report["metrics"]["action_accuracy"]["rate_pct"] == 50.0
    assert report["failure_counts"]["action"] == 1


def test_authorized_development_token_is_valid_and_short_lived() -> None:
    context = verify_trusted_context_token(
        _trusted_context_token(LOCAL_TRUSTED_CONTEXT_SECRET),
        LOCAL_TRUSTED_CONTEXT_SECRET,
    )
    assert context.user_id == "stage5-development-evaluator"
