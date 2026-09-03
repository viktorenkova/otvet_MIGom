from backend.app.bot import routing_v3


def test_domain_token_repair_reuses_cached_damerau_result(monkeypatch) -> None:
    """Repeated corpus tokens must not re-run the expensive fuzzy comparison."""
    original_distance = routing_v3._damerau_distance
    calls = 0

    def counted_distance(left: str, right: str) -> int:
        nonlocal calls
        calls += 1
        return original_distance(left, right)

    monkeypatch.setattr(routing_v3, "_damerau_distance", counted_distance)
    routing_v3._repair_domain_token.cache_clear()
    try:
        first = routing_v3._repair_domain_token("почтвоой")
        first_call_count = calls
        second = routing_v3._repair_domain_token("почтвоой")

        assert first == second
        assert first_call_count > 0
        assert calls == first_call_count
    finally:
        routing_v3._repair_domain_token.cache_clear()
