from backend.app.bot.pairwise_reranker import BASE_FEATURE_COUNT, PairwiseScenarioReranker


def test_pairwise_bundle_has_stable_scenario_columns() -> None:
    reranker = PairwiseScenarioReranker()
    assert reranker.available, reranker.error
    assert reranker.bundle is not None
    assert reranker.bundle["feature_count"] == (
        BASE_FEATURE_COUNT + len(reranker.bundle["feature_scenario_ids"])
    )
