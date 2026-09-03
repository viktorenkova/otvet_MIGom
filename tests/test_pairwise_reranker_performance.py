from backend.app.bot import pairwise_reranker


def test_atomic_titles_do_not_apply_user_input_recovery(monkeypatch) -> None:
    def unexpected_user_normalizer(_: str) -> str:
        raise AssertionError("reviewed scenario titles must not use user-input recovery")

    monkeypatch.setattr(pairwise_reranker, "routing_normalize", unexpected_user_normalizer)
    pairwise_reranker._atomic_titles.cache_clear()
    try:
        assert pairwise_reranker._atomic_titles()
    finally:
        pairwise_reranker._atomic_titles.cache_clear()


def test_pairwise_bundle_can_be_warmed_before_user_traffic() -> None:
    reranker = pairwise_reranker.PairwiseScenarioReranker()

    assert reranker.warm() is reranker.available
