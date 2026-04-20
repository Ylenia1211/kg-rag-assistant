from __future__ import annotations

from functools import lru_cache
from typing import Protocol

from sentence_transformers import SentenceTransformer

from app.core.config import get_settings


class EmbeddingServiceError(Exception):
    pass


class EmbeddingProvider(Protocol):
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        ...


class DummyEmbeddingProvider:
    def __init__(self, vector_size: int) -> None:
        self.vector_size = vector_size

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        embeddings: list[list[float]] = []
        for text in texts:
            base = float(len(text)) if text else 0.0
            vector = [0.0] * self.vector_size
            if self.vector_size > 0:
                vector[0] = base
            embeddings.append(vector)
        return embeddings


@lru_cache(maxsize=1)
def load_sentence_transformer(model_name: str) -> SentenceTransformer:
    return SentenceTransformer(model_name)


class LocalSentenceTransformerProvider:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        try:
            model = load_sentence_transformer(self.model_name)
            vectors = model.encode(texts, normalize_embeddings=True)
            return vectors.tolist()
        except Exception as exc:
            raise EmbeddingServiceError(f"Local embedding generation failed: {exc}") from exc


def get_embedding_provider() -> EmbeddingProvider:
    settings = get_settings()
    llm_provider = getattr(settings, "llm_provider", "dummy")

    if llm_provider == "local":
        return LocalSentenceTransformerProvider(
            model_name=settings.local_embedding_model,
        )

    return DummyEmbeddingProvider(vector_size=settings.qdrant_vector_size)


def embed_texts(texts: list[str], provider: EmbeddingProvider | None = None) -> list[list[float]]:
    if not texts:
        return []

    active_provider = provider or get_embedding_provider()
    embeddings = active_provider.embed_texts(texts)

    if len(embeddings) != len(texts):
        raise EmbeddingServiceError("Embedding provider returned a mismatched number of vectors")

    return embeddings