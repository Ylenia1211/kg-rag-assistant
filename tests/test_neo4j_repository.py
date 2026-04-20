from __future__ import annotations

import app.repositories.neo4j_repository as repo_module


def make_record() -> repo_module.Neo4jDocumentGraphRecord:
    return repo_module.Neo4jDocumentGraphRecord(
        document_id="doc-1",
        filename="note.txt",
        content_type="text/plain",
        text_length=11,
        chunks=[
            repo_module.Neo4jChunkNode(
                chunk_id=1,
                text="hello worl",
                start_char=0,
                end_char=10,
            ),
            repo_module.Neo4jChunkNode(
                chunk_id=2,
                text="rld",
                start_char=8,
                end_char=11,
            ),
        ],
        entities=[
            repo_module.Neo4jEntityNode(
                entity_id="technology:neo4j",
                name="Neo4j",
                entity_type="Technology",
            ),
            repo_module.Neo4jEntityNode(
                entity_id="concept:rag",
                name="RAG",
                entity_type="Concept",
            ),
        ],
        relations=[
            repo_module.Neo4jRelationEdge(
                source_entity_id="concept:rag",
                target_entity_id="technology:neo4j",
                relation_type="RELATED_TO",
            )
        ],
    )


def test_upsert_document_graph_runs_document_chunk_entity_and_relation_queries(monkeypatch):
    calls: list[tuple[str, dict]] = []

    def fake_run_cypher(query: str, parameters: dict | None = None):
        calls.append((query, parameters or {}))
        return []

    monkeypatch.setattr(repo_module, "run_cypher", fake_run_cypher)

    count = repo_module.upsert_document_graph(make_record())

    assert count == 2
    assert len(calls) == 6

    assert "MERGE (d:Document {id: $document_id})" in calls[0][0]
    assert "MERGE (c:Chunk {id: $chunk_node_id})" in calls[1][0]
    assert "MERGE (c:Chunk {id: $chunk_node_id})" in calls[2][0]
    assert "MERGE (e:Entity {id: $entity_id})" in calls[3][0]
    assert "MERGE (e:Entity {id: $entity_id})" in calls[4][0]
    assert "MERGE (source)-[:RELATED_TO {type: $relation_type}]->(target)" in calls[5][0]


def test_get_document_chunks_returns_rows(monkeypatch):
    expected = [
        {
            "document_id": "doc-1",
            "chunk_id": 1,
            "text": "hello worl",
            "start_char": 0,
            "end_char": 10,
        }
    ]

    def fake_run_cypher(query: str, parameters: dict | None = None):
        assert "MATCH (d:Document {id: $document_id})-[:HAS_CHUNK]->(c:Chunk)" in query
        assert parameters == {"document_id": "doc-1"}
        return expected

    monkeypatch.setattr(repo_module, "run_cypher", fake_run_cypher)

    rows = repo_module.get_document_chunks("doc-1")

    assert rows == expected


def test_get_related_entities_returns_rows(monkeypatch):
    expected = [
        {
            "document_id": "doc-1",
            "entity_id": "technology:neo4j",
            "entity_name": "Neo4j",
            "entity_type": "Technology",
        }
    ]

    def fake_run_cypher(query: str, parameters: dict | None = None):
        assert "MATCH (d:Document {id: $document_id})-[:MENTIONS]->(e:Entity)" in query
        assert parameters == {"document_id": "doc-1"}
        return expected

    monkeypatch.setattr(repo_module, "run_cypher", fake_run_cypher)

    rows = repo_module.get_related_entities("doc-1")

    assert rows == expected