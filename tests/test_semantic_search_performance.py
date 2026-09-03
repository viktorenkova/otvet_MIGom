from types import SimpleNamespace

from backend.app.bot import knowledge_search, semantic_search


def test_semantic_index_does_not_apply_user_typo_recovery_to_reviewed_articles(monkeypatch) -> None:
    def unexpected_user_normalizer(_: str) -> str:
        raise AssertionError("reviewed KB text must not use user-input typo recovery")

    monkeypatch.setattr(semantic_search, "routing_normalize", unexpected_user_normalizer)
    article = SimpleNamespace(
        slug="documents.test",
        intent="documents",
        title="Документы",
        problem="Как получить документы",
        user_phrases=("как получить документы",),
        trigger_phrases=(),
        keywords=("документы",),
        search_document="Документы по сделке доступны после подготовки.",
        user_answer="",
    )

    index = semantic_search.TfidfSemanticIndex([article], {})

    assert index.article_ids == ("documents.test",)


def test_prepared_kb_articles_do_not_use_user_input_recovery(monkeypatch) -> None:
    def unexpected_user_normalizer(_: str) -> str:
        raise AssertionError("reviewed KB text must not use user-input recovery")

    # Loading static rules itself may normalize short policy phrases. Cache
    # that one-time loader first; this test targets preparation of article
    # bodies, which used to account for the cold-start bottleneck.
    knowledge_search.load_articles()
    monkeypatch.setattr(knowledge_search, "normalize_matching_text", unexpected_user_normalizer)
    knowledge_search._prepared_articles.cache_clear()
    try:
        assert knowledge_search._prepared_articles()
    finally:
        knowledge_search._prepared_articles.cache_clear()
