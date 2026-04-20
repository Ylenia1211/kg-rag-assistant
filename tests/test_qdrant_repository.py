from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

from qdrant_client.http import models

import app.repositories.qdrant_repository as repo_module


def make_record() -> repo_module.QdrantChunkRecord:
    return repo_module.QdrantChunkRecord(
        document_id="doc-1",
        chunk_id=1,
        text="hello world",
        embedding=[0.1, 0.2, 0.3],
        start_char=0,
        end_char=11,
        source_filename="note.txt",
    )


def test_build_qdrant_points_maps_records_to_payload(monkeypatch):
    monkeypatch.setattr(repo_module, "uuid4", lambda: "fixed-id")

    points = repo_module.build_qdrant_points([make_record()])

    assert len(points) == 1
    point = points[0]
    assert isinstance(point, models.PointStruct)
    assert point.id == "fixed-id"
    assert point.vector == [0.1, 0.2, 0.3]
    assert point.payload == {
        "document_id": "doc-1",
        "chunk_id": 1,
        "text": "hello world",
        "start_char": 0,
        "end_char": 11,
        "source_filename": "note.txt",
    }


def test_upsert_chunk_records_returns_zero_for_empty_input():
    assert repo_module.upsert_chunk_records([]) == 0


def test_upsert_chunk_records_calls_client(monkeypatch):
    fake_client = Mock()
    monkeypatch.setattr(repo_module, "get_qdrant_client", lambda: fake_client)
    monkeypatch.setattr(
        repo_module,
        "get_settings",
        lambda: SimpleNamespace(qdrant_collection="documents"),
    )
    monkeypatch.setattr(repo_module, "build_qdrant_points", lambda records: ["p1", "p2"])

    count = repo_module.upsert_chunk_records([make_record(), make_record()])

    assert count == 2
    fake_client.upsert.assert_called_once_with(
        collection_name="documents",
        points=["p1", "p2"],
        wait=True,
    )


def test_search_similar_chunks_uses_default_limit(monkeypatch):
    fake_client = Mock()
    fake_client.query_points.return_value = SimpleNamespace(
        points=[
            SimpleNamespace(id="1", score=0.9, payload={"text": "hello"}),
            SimpleNamespace(id="2", score=0.8, payload={"text": "world"}),
        ]
    )

    monkeypatch.setattr(repo_module, "get_qdrant_client", lambda: fake_client)
    monkeypatch.setattr(
        repo_module,
        "get_settings",
        lambda: SimpleNamespace(qdrant_collection="documents", top_k_vector=5),
    )

    rows = repo_module.search_similar_chunks([0.1, 0.2, 0.3])

    fake_client.query_points.assert_called_once_with(
        collection_name="documents",
        query=[0.1, 0.2, 0.3],
        limit=5,
        with_payload=True,
    )
    assert rows == [
        {"id": "1", "score": 0.9, "payload": {"text": "hello"}},
        {"id": "2", "score": 0.8, "payload": {"text": "world"}},
    ]


def test_search_similar_chunks_respects_explicit_limit(monkeypatch):
    fake_client = Mock()
    fake_client.query_points.return_value = SimpleNamespace(points=[])

    monkeypatch.setattr(repo_module, "get_qdrant_client", lambda: fake_client)
    monkeypatch.setattr(
        repo_module,
        "get_settings",
        lambda: SimpleNamespace(qdrant_collection="documents", top_k_vector=5),
    )

    repo_module.search_similar_chunks([0.1, 0.2], limit=2)

    fake_client.query_points.assert_called_once_with(
        collection_name="documents",
        query=[0.1, 0.2],
        limit=2,
        with_payload=True,
    )