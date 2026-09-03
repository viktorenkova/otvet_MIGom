from __future__ import annotations

from dataclasses import dataclass
import hashlib
import logging
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable

from backend.app.bot.routing_v3 import _index_normalize, routing_normalize
from backend.app.bot.text_processing import normalize_text


@dataclass(frozen=True)
class SemanticMatch:
    article_id: str
    similarity: float
    margin: float
    candidate_article_ids: tuple[str, ...] = ()
    lexical_similarity: float = 0.0
    dense_similarity: float = 0.0
    dense_available: bool = False


@dataclass(frozen=True)
class SemanticCandidate:
    article_id: str
    score: float
    lexical_score: float
    char_score: float
    word_score: float
    dense_score: float
    dense_similarity: float
    intent_boost: float
    dense_available: bool


class TfidfSemanticIndex:
    """Local character n-gram vectors; no model download or external API is used."""

    def __init__(self, articles: Iterable[Any], config: dict[str, Any]) -> None:
        import numpy as np
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.articles = tuple(articles)
        self.article_ids = tuple(str(article.slug) for article in self.articles)
        self.article_intents = tuple(str(article.intent) for article in self.articles)
        self.article_positions = {
            article_id: position
            for position, article_id in enumerate(self.article_ids)
        }
        ngram_min = max(2, int(config.get("char_ngram_min", 2)))
        ngram_max = max(ngram_min, int(config.get("char_ngram_max", 5)))
        self.intent_boost = max(0.0, float(config.get("intent_boost", 0.025)))
        self.char_weight = max(0.0, float(config.get("lexical_char_weight", 0.65)))
        self.word_weight = max(0.0, float(config.get("lexical_word_weight", 0.35)))
        lexical_total = self.char_weight + self.word_weight
        if lexical_total <= 0:
            self.char_weight, self.word_weight, lexical_total = 1.0, 0.0, 1.0
        self.char_weight /= lexical_total
        self.word_weight /= lexical_total
        self.char_vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(ngram_min, ngram_max),
            lowercase=False,
            min_df=1,
            sublinear_tf=True,
            dtype=np.float32,
        )
        self.word_vectorizer = TfidfVectorizer(
            preprocessor=_index_normalize,
            analyzer="word",
            ngram_range=(1, 2),
            min_df=1,
            sublinear_tf=True,
            dtype=np.float32,
        )
        documents = [self._article_document(article) for article in self.articles]
        self.char_matrix = self.char_vectorizer.fit_transform(documents)
        self.word_matrix = self.word_vectorizer.fit_transform(documents)

    def similarity_channels(self, message: str):
        query = routing_normalize(message)
        if not query:
            return None
        char_query = self.char_vectorizer.transform([query])
        word_query = self.word_vectorizer.transform([query])
        char_values = (self.char_matrix @ char_query.T).toarray().ravel()
        word_values = (self.word_matrix @ word_query.T).toarray().ravel()
        combined = char_values * self.char_weight + word_values * self.word_weight
        return char_values, word_values, combined

    def similarities(self, message: str):
        channels = self.similarity_channels(message)
        return channels[2] if channels is not None else None

    def _article_document(self, article: Any) -> str:
        search_document = str(getattr(article, "search_document", "") or "")
        parts = [
            *([str(article.title)] * 4),
            *([str(article.problem)] * 3),
            *([str(item) for item in article.user_phrases] * 2),
            *([str(item) for item in article.trigger_phrases] * 2),
            *[str(item) for item in article.keywords],
            search_document or str(article.user_answer or ""),
        ]
        return _index_normalize(" ".join(part for part in parts if part))

    def search(
        self,
        message: str,
        candidate_ids: set[str],
        intent: str,
    ) -> SemanticMatch | None:
        if not candidate_ids:
            return None
        similarities = self.similarities(message)
        if similarities is None:
            return None
        ranked: list[tuple[float, float, int]] = []
        for article_id in candidate_ids:
            position = self.article_positions.get(article_id)
            if position is None:
                continue
            similarity = float(similarities[position])
            adjusted = similarity
            if intent != "unknown" and self.article_intents[position] in {intent, "unknown"}:
                adjusted += self.intent_boost
            ranked.append((adjusted, similarity, position))
        if not ranked:
            return None

        ranked.sort(key=lambda item: item[0], reverse=True)
        best_adjusted, best_similarity, best_position = ranked[0]
        second_adjusted = ranked[1][0] if len(ranked) > 1 else 0.0
        candidate_article_ids: list[str] = []
        candidate_intents: set[str] = set()
        for _, _, position in ranked:
            candidate_intent = self.article_intents[position]
            if candidate_intent in candidate_intents:
                continue
            candidate_intents.add(candidate_intent)
            candidate_article_ids.append(self.article_ids[position])
            if len(candidate_article_ids) == 2:
                break
        return SemanticMatch(
            article_id=self.article_ids[best_position],
            similarity=best_similarity,
            margin=max(0.0, best_adjusted - second_adjusted),
            candidate_article_ids=tuple(candidate_article_ids),
            lexical_similarity=best_similarity,
        )


class MultilingualHybridSemanticIndex:
    """Hybrid lexical+dense retrieval with a safe lexical fallback.

    Character TF-IDF remains useful for typos, identifiers and exact wording.
    A multilingual E5 embedding model supplies the semantic recall that TF-IDF
    cannot provide for colloquial Russian and paraphrases. If the optional
    model is unavailable, startup and search continue with TF-IDF only.
    """

    def __init__(self, articles: Iterable[Any], config: dict[str, Any]) -> None:
        import numpy as np

        self._np = np
        self.config = config
        self.articles = tuple(articles)
        self.article_ids = tuple(str(article.slug) for article in self.articles)
        self.article_intents = tuple(str(article.intent) for article in self.articles)
        self.article_positions = {
            article_id: position
            for position, article_id in enumerate(self.article_ids)
        }
        self.lexical = TfidfSemanticIndex(self.articles, config)
        self.intent_boost = max(0.0, float(config.get("intent_boost", 0.025)))
        self.lexical_weight = max(0.0, float(config.get("lexical_weight", 0.30)))
        self.dense_weight = max(0.0, float(config.get("dense_weight", 0.70)))
        total_weight = self.lexical_weight + self.dense_weight
        if total_weight <= 0:
            self.lexical_weight, self.dense_weight, total_weight = 1.0, 0.0, 1.0
        self.lexical_weight /= total_weight
        self.dense_weight /= total_weight
        self.dense_floor = float(config.get("dense_similarity_floor", 0.78))
        self.model = None
        self.dense_matrix = None
        self.dense_error = ""
        dense_env = os.getenv("SEMANTIC_DENSE_ENABLED")
        dense_enabled = bool(config.get("dense_enabled", False)) if dense_env is None else dense_env.casefold() in {
            "1", "true", "yes", "on",
        }
        if dense_enabled:
            self._load_dense_index()

    def _article_document(self, article: Any) -> str:
        # Repeating metadata makes short, user-like fields dominate long answers.
        search_document = str(getattr(article, "search_document", "") or "")
        parts = [
            *([str(article.title)] * 3),
            str(article.problem)[:400],
            *[str(item) for item in article.user_phrases[:8]],
            *[str(item) for item in article.trigger_phrases[:5]],
            *[str(item) for item in article.keywords[:12]],
            (search_document or str(article.user_answer or ""))[:800],
        ]
        normalized = normalize_text(" ".join(part for part in parts if part))
        return normalized[:max(500, int(self.config.get("dense_document_chars", 1400)))]

    def _load_dense_index(self) -> None:
        model_name = str(self.config.get("dense_model") or "intfloat/multilingual-e5-small")
        allow_download = os.getenv("SEMANTIC_MODEL_ALLOW_DOWNLOAD", "true").casefold() in {
            "1", "true", "yes", "on",
        }
        try:
            from sentence_transformers import SentenceTransformer

            model_kwargs = {"device": str(self.config.get("dense_device") or "cpu")}
            try:
                # Prefer an already provisioned model. Besides being faster,
                # this prevents an unnecessary Hub metadata request on every
                # restart in network-restricted environments.
                self.model = SentenceTransformer(
                    model_name,
                    local_files_only=True,
                    **model_kwargs,
                )
            except Exception:
                if not allow_download:
                    raise
                self.model = SentenceTransformer(model_name, **model_kwargs)
            passage_prefix = str(self.config.get("dense_passage_prefix") or "passage: ")
            documents = [
                passage_prefix + self._article_document(article)
                for article in self.articles
            ]
            self.dense_matrix = self._load_cached_embeddings(model_name, documents)
            if self.dense_matrix is None:
                self.dense_matrix = self.model.encode(
                    documents,
                    batch_size=max(1, int(self.config.get("dense_batch_size", 32))),
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
                self._save_cached_embeddings(model_name, documents, self.dense_matrix)
        except Exception as exc:  # optional dependency/model must not take down support
            self.model = None
            self.dense_matrix = None
            self.dense_error = f"{type(exc).__name__}: {exc}"
            logging.getLogger(__name__).warning(
                "Dense semantic search unavailable; using TF-IDF fallback: %s",
                self.dense_error,
            )

    def _cache_path(self, model_name: str, documents: list[str]) -> Path:
        cache_root = os.getenv("SEMANTIC_CACHE_DIR", "").strip()
        root = Path(cache_root) if cache_root else Path(tempfile.gettempdir()) / "migtorg-semantic-cache"
        fingerprint = hashlib.sha256(
            (model_name + "\0" + "\n".join(documents)).encode("utf-8")
        ).hexdigest()[:20]
        return root / f"articles-{fingerprint}.npz"

    def _load_cached_embeddings(self, model_name: str, documents: list[str]):
        path = self._cache_path(model_name, documents)
        try:
            if not path.is_file():
                return None
            with self._np.load(path, allow_pickle=False) as cached:
                matrix = cached["embeddings"]
            if len(matrix) != len(documents):
                return None
            return matrix
        except Exception:
            return None

    def _save_cached_embeddings(self, model_name: str, documents: list[str], matrix) -> None:
        path = self._cache_path(model_name, documents)
        temporary = path.with_suffix(".tmp.npz")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._np.savez_compressed(temporary, embeddings=matrix)
            os.replace(temporary, path)
        except Exception as exc:
            logging.getLogger(__name__).warning("Could not persist semantic cache: %s", exc)

    def _dense_similarities(self, message: str):
        if self.model is None or self.dense_matrix is None:
            return None
        query_prefix = str(self.config.get("dense_query_prefix") or "query: ")
        vector = self.model.encode(
            [query_prefix + normalize_text(message)],
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]
        return self.dense_matrix @ vector

    def dense_similarities_many(self, messages: list[str]):
        """Encode evaluation/offline query batches without changing scoring."""
        if self.model is None or self.dense_matrix is None or not messages:
            return None
        query_prefix = str(self.config.get("dense_query_prefix") or "query: ")
        cache_path = self._cache_path(
            str(self.config.get("dense_model") or "intfloat/multilingual-e5-small") + "-queries",
            messages,
        )
        try:
            if cache_path.is_file():
                with self._np.load(cache_path, allow_pickle=False) as cached:
                    similarities = cached["similarities"]
                if similarities.shape == (len(messages), len(self.article_ids)):
                    return similarities
        except Exception:
            pass
        vectors = self.model.encode(
            [query_prefix + normalize_text(message) for message in messages],
            batch_size=max(1, int(self.config.get("dense_batch_size", 32))),
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        similarities = vectors @ self.dense_matrix.T
        temporary = cache_path.with_suffix(".tmp.npz")
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            self._np.savez_compressed(temporary, similarities=similarities)
            os.replace(temporary, cache_path)
        except Exception as exc:
            logging.getLogger(__name__).warning("Could not persist dense query cache: %s", exc)
        return similarities

    def _normalized_dense(self, similarities):
        denominator = max(1e-6, 1.0 - self.dense_floor)
        return self._np.clip((similarities - self.dense_floor) / denominator, 0.0, 1.0)

    def search(
        self,
        message: str,
        candidate_ids: set[str],
        intent: str,
    ) -> SemanticMatch | None:
        if not candidate_ids:
            return None
        ranked_candidates = self.rank(message, candidate_ids, intent)
        if not ranked_candidates:
            return None
        best = ranked_candidates[0]
        second_score = ranked_candidates[1].score if len(ranked_candidates) > 1 else 0.0
        candidate_article_ids: list[str] = []
        seen_intents: set[str] = set()
        for candidate in ranked_candidates:
            position = self.article_positions[candidate.article_id]
            candidate_intent = self.article_intents[position]
            if candidate_intent in seen_intents:
                continue
            seen_intents.add(candidate_intent)
            candidate_article_ids.append(self.article_ids[position])
            if len(candidate_article_ids) == 3:
                break
        return SemanticMatch(
            article_id=best.article_id,
            similarity=best.lexical_score * self.lexical_weight + best.dense_score * self.dense_weight if best.dense_available else best.lexical_score,
            margin=max(0.0, best.score - second_score),
            candidate_article_ids=tuple(candidate_article_ids),
            lexical_similarity=best.lexical_score,
            dense_similarity=best.dense_similarity,
            dense_available=best.dense_available,
        )

    def rank(
        self,
        message: str,
        candidate_ids: set[str],
        intent: str = "unknown",
        top_k: int | None = None,
        dense_similarities=None,
        lexical_weight: float | None = None,
        dense_weight: float | None = None,
    ) -> tuple[SemanticCandidate, ...]:
        """Return candidates only, with an auditable per-channel score trace."""
        if not candidate_ids:
            return ()
        channels = self.lexical.similarity_channels(message)
        if channels is None:
            return ()
        char_values, word_values, lexical = channels
        dense = dense_similarities if dense_similarities is not None else self._dense_similarities(message)
        dense_available = dense is not None
        dense_scores = self._normalized_dense(dense) if dense_available else None
        effective_lexical_weight = self.lexical_weight if lexical_weight is None else max(0.0, lexical_weight)
        effective_dense_weight = self.dense_weight if dense_weight is None else max(0.0, dense_weight)
        effective_total = effective_lexical_weight + effective_dense_weight
        if effective_total <= 0:
            effective_lexical_weight, effective_dense_weight, effective_total = 1.0, 0.0, 1.0
        effective_lexical_weight /= effective_total
        effective_dense_weight /= effective_total
        combined = (
            effective_lexical_weight * lexical + effective_dense_weight * dense_scores
            if dense_available
            else lexical
        )
        ranked: list[SemanticCandidate] = []
        for article_id in candidate_ids:
            position = self.article_positions.get(article_id)
            if position is None:
                continue
            boost = (
                self.intent_boost
                if intent != "unknown" and self.article_intents[position] in {intent, "unknown"}
                else 0.0
            )
            ranked.append(
                SemanticCandidate(
                    article_id=article_id,
                    score=float(combined[position]) + boost,
                    lexical_score=float(lexical[position]),
                    char_score=float(char_values[position]),
                    word_score=float(word_values[position]),
                    dense_score=float(dense_scores[position]) if dense_available else 0.0,
                    dense_similarity=float(dense[position]) if dense_available else 0.0,
                    intent_boost=boost,
                    dense_available=dense_available,
                )
            )
        ranked.sort(key=lambda item: (item.score, item.article_id), reverse=True)
        return tuple(ranked[:top_k] if top_k is not None else ranked)
