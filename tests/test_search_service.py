from __future__ import annotations

import app.services.search_service as search_module


def test_search_knowledge_base_returns_vector_graph_entities_and_answer(monkeypatch):
    monkeypatch.setattr(search_module, "embed_texts", lambda texts: [[0.1, 0.2, 0.3]])
    monkeypatch.setattr(
        search_module,
        "search_similar_chunks",
        lambda query_vector, limit=None: [
            {
                "id": "p1",
                "score": 0.9,
                "payload": {"document_id": "doc-1", "chunk_id": 1, "text": "hello"},
            },
            {
                "id": "p2",
                "score": 0.8,
                "payload": {"document_id": "doc-2", "chunk_id": 1, "text": "world"},
            },
        ],
    )

    def fake_get_document_chunks(document_id: str):
        return [
            {
                "document_id": document_id,
                "chunk_id": 1,
                "text": f"chunk for {document_id}",
                "start_char": 0,
                "end_char": 10,
            }
        ]

    def fake_get_related_entities(document_id: str):
        return [
            {
                "document_id": document_id,
                "entity_id": f"entity:{document_id}",
                "entity_name": f"Entity {document_id}",
                "entity_type": "Concept",
            }
        ]

    monkeypatch.setattr(search_module, "get_document_chunks", fake_get_document_chunks)
    monkeypatch.setattr(search_module, "get_related_entities", fake_get_related_entities)
    monkeypatch.setattr(
        search_module,
        "generate_answer",
        lambda question, vector_results, graph_results: "risposta finale",
    )

    result = search_module.search_knowledge_base("hello", limit=2)

    assert result.question == "hello"
    assert result.answer == "risposta finale"
    assert len(result.vector_results) == 2
    assert result.graph_results == [
        {
            "document_id": "doc-1",
            "chunk_id": 1,
            "text": "chunk for doc-1",
            "start_char": 0,
            "end_char": 10,
        },
        {
            "document_id": "doc-2",
            "chunk_id": 1,
            "text": "chunk for doc-2",
            "start_char": 0,
            "end_char": 10,
        },
    ]
    assert result.entity_results == [
        {
            "document_id": "doc-1",
            "entity_id": "entity:doc-1",
            "entity_name": "Entity doc-1",
            "entity_type": "Concept",
        },
        {
            "document_id": "doc-2",
            "entity_id": "entity:doc-2",
            "entity_name": "Entity doc-2",
            "entity_type": "Concept",
        },
    ]


def test_search_knowledge_base_deduplicates_graph_and_entity_fetches(monkeypatch):
    monkeypatch.setattr(search_module, "embed_texts", lambda texts: [[0.1, 0.2]])
    monkeypatch.setattr(
        search_module,
        "search_similar_chunks",
        lambda query_vector, limit=None: [
            {"id": "p1", "score": 0.9, "payload": {"document_id": "doc-1"}},
            {"id": "p2", "score": 0.8, "payload": {"document_id": "doc-1"}},
        ],
    )

    chunk_calls = []
    entity_calls = []

    def fake_get_document_chunks(document_id: str):
        chunk_calls.append(document_id)
        return []

    def fake_get_related_entities(document_id: str):
        entity_calls.append(document_id)
        return []

    monkeypatch.setattr(search_module, "get_document_chunks", fake_get_document_chunks)
    monkeypatch.setattr(search_module, "get_related_entities", fake_get_related_entities)
    monkeypatch.setattr(search_module, "generate_answer", lambda question, vector_results, graph_results: "ok")

    result = search_module.search_knowledge_base("hello")

    assert result.answer == "ok"
    assert result.vector_results[0]["id"] == "p1"
    assert chunk_calls == ["doc-1"]
    assert entity_calls == ["doc-1"]