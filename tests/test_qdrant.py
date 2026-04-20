from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import app.core.qdrant as qdrant_module
from qdrant_client.http import models


def test_get_qdrant_client_uses_settings(monkeypatch):
    fake_client = object()
    factory = Mock(return_value=fake_client)

    monkeypatch.setattr(qdrant_module, "QdrantClient", factory)
    monkeypatch.setattr(
        qdrant_module,
        "get_settings",
        lambda: SimpleNamespace(
            qdrant_url="http://localhost:6333",
            qdrant_auth_kwargs=lambda: {"api_key": "abc123"},
        ),
    )

    qdrant_module.get_qdrant_client.cache_clear()
    client = qdrant_module.get_qdrant_client()

    assert client is fake_client
    factory.assert_called_once_with(url="http://localhost:6333", api_key="abc123")

    qdrant_module.get_qdrant_client.cache_clear()


def test_ensure_qdrant_collection_skips_existing(monkeypatch):
    fake_client = Mock()
    fake_client.get_collections.return_value = SimpleNamespace(
        collections=[SimpleNamespace(name="documents")]
    )

    monkeypatch.setattr(
        qdrant_module,
        "get_settings",
        lambda: SimpleNamespace(qdrant_collection="documents", qdrant_vector_size=1536),
    )
    monkeypatch.setattr(qdrant_module, "get_qdrant_client", lambda: fake_client)

    qdrant_module.ensure_qdrant_collection()

    fake_client.create_collection.assert_not_called()


def test_ensure_qdrant_collection_creates_missing(monkeypatch):
    fake_client = Mock()
    fake_client.get_collections.return_value = SimpleNamespace(collections=[])

    monkeypatch.setattr(
        qdrant_module,
        "get_settings",
        lambda: SimpleNamespace(qdrant_collection="documents", qdrant_vector_size=1536),
    )
    monkeypatch.setattr(qdrant_module, "get_qdrant_client", lambda: fake_client)

    qdrant_module.ensure_qdrant_collection()

    fake_client.create_collection.assert_called_once()
    kwargs = fake_client.create_collection.call_args.kwargs

    assert kwargs["collection_name"] == "documents"
    assert isinstance(kwargs["vectors_config"], models.VectorParams)
    assert kwargs["vectors_config"].size == 1536
    assert kwargs["vectors_config"].distance == models.Distance.COSINE


def test_qdrant_healthcheck_returns_true(monkeypatch):
    fake_client = Mock()
    fake_client.get_collections.return_value = SimpleNamespace(collections=[])

    monkeypatch.setattr(qdrant_module, "get_qdrant_client", lambda: fake_client)

    assert qdrant_module.qdrant_healthcheck() is True


def test_qdrant_healthcheck_returns_false_on_error(monkeypatch):
    fake_client = Mock()
    fake_client.get_collections.side_effect = RuntimeError("boom")

    monkeypatch.setattr(qdrant_module, "get_qdrant_client", lambda: fake_client)

    assert qdrant_module.qdrant_healthcheck() is False