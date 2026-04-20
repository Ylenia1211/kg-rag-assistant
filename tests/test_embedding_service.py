from __future__ import annotations

import pytest

import app.services.embedding_service as embedding_module


class FakeProvider:
    def __init__(self, embeddings):
        self._embeddings = embeddings

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return self._embeddings


def test_dummy_embedding_provider_returns_fixed_size_vectors():
    provider = embedding_module.DummyEmbeddingProvider(vector_size=4)

    embeddings = provider.embed_texts(["hello", ""])

    assert embeddings == [
        [5.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ]


def test_embed_texts_returns_empty_list_for_empty_input():
    assert embedding_module.embed_texts([]) == []


def test_embed_texts_uses_custom_provider():
    provider = FakeProvider([[0.1, 0.2], [0.3, 0.4]])

    embeddings = embedding_module.embed_texts(["a", "b"], provider=provider)

    assert embeddings == [[0.1, 0.2], [0.3, 0.4]]


def test_embed_texts_raises_on_mismatched_vector_count():
    provider = FakeProvider([[0.1, 0.2]])

    with pytest.raises(
        embedding_module.EmbeddingServiceError,
        match="Embedding provider returned a mismatched number of vectors",
    ):
        embedding_module.embed_texts(["a", "b"], provider=provider)


def test_get_embedding_provider_uses_qdrant_vector_size(monkeypatch):
    class FakeSettings:
        qdrant_vector_size = 8

    monkeypatch.setattr(embedding_module, "get_settings", lambda: FakeSettings())

    provider = embedding_module.get_embedding_provider()

    assert isinstance(provider, embedding_module.DummyEmbeddingProvider)
    assert provider.vector_size == 8