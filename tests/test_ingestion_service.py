from __future__ import annotations

from types import SimpleNamespace

import pytest

import app.services.ingestion_service as ingestion_module
from app.repositories.neo4j_repository import Neo4jDocumentGraphRecord
from app.repositories.qdrant_repository import QdrantChunkRecord
from app.services.embedding_service import EmbeddingServiceError
from app.services.text_chunker import TextChunk


def test_build_qdrant_records_maps_chunks_and_embeddings():
    chunks = [
        TextChunk(chunk_id=1, text="hello", start_char=0, end_char=5),
        TextChunk(chunk_id=2, text="world", start_char=6, end_char=11),
    ]
    embeddings = [
        [0.1, 0.2],
        [0.3, 0.4],
    ]

    records = ingestion_module.build_qdrant_records(
        document_id="doc-1",
        filename="note.txt",
        chunks=chunks,
        embeddings=embeddings,
    )

    assert records == [
        QdrantChunkRecord(
            document_id="doc-1",
            chunk_id=1,
            text="hello",
            embedding=[0.1, 0.2],
            start_char=0,
            end_char=5,
            source_filename="note.txt",
        ),
        QdrantChunkRecord(
            document_id="doc-1",
            chunk_id=2,
            text="world",
            embedding=[0.3, 0.4],
            start_char=6,
            end_char=11,
            source_filename="note.txt",
        ),
    ]


def test_build_qdrant_records_raises_on_mismatch():
    chunks = [TextChunk(chunk_id=1, text="hello", start_char=0, end_char=5)]
    embeddings = []

    with pytest.raises(EmbeddingServiceError, match="Chunk and embedding counts do not match"):
        ingestion_module.build_qdrant_records(
            document_id="doc-1",
            filename="note.txt",
            chunks=chunks,
            embeddings=embeddings,
        )


def test_build_neo4j_graph_record_maps_chunks_entities_and_relations():
    chunks = [
        TextChunk(chunk_id=1, text="hello", start_char=0, end_char=5),
    ]

    entities = [
        SimpleNamespace(entity_id="technology:neo4j", name="Neo4j", entity_type="Technology"),
    ]
    relations = [
        SimpleNamespace(
            source_entity_id="concept:rag",
            target_entity_id="technology:neo4j",
            relation_type="RELATED_TO",
        )
    ]

    record = ingestion_module.build_neo4j_graph_record(
        document_id="doc-1",
        filename="note.txt",
        content_type="text/plain",
        text_length=11,
        chunks=chunks,
        entities=entities,
        relations=relations,
    )

    assert isinstance(record, Neo4jDocumentGraphRecord)
    assert record.document_id == "doc-1"
    assert len(record.chunks) == 1
    assert len(record.entities) == 1
    assert len(record.relations) == 1
    assert record.entities[0].name == "Neo4j"
    assert record.relations[0].relation_type == "RELATED_TO"


def test_ingest_document_success(monkeypatch):
    monkeypatch.setattr(
        ingestion_module,
        "get_settings",
        lambda: SimpleNamespace(chunk_size=10, chunk_overlap=2),
    )
    monkeypatch.setattr(ingestion_module, "parse_document", lambda filename, content: "hello Neo4j RAG")
    monkeypatch.setattr(
        ingestion_module,
        "chunk_text",
        lambda text, chunk_size, chunk_overlap: [
            TextChunk(chunk_id=1, text="hello Neo4", start_char=0, end_char=10),
            TextChunk(chunk_id=2, text="4j RAG", start_char=8, end_char=14),
        ],
    )
    monkeypatch.setattr(
        ingestion_module,
        "extract_entities",
        lambda text: [
            SimpleNamespace(entity_id="technology:neo4j", name="Neo4j", entity_type="Technology"),
            SimpleNamespace(entity_id="concept:rag", name="RAG", entity_type="Concept"),
        ],
    )
    monkeypatch.setattr(
        ingestion_module,
        "infer_relations",
        lambda entities: [
            SimpleNamespace(
                source_entity_id="concept:rag",
                target_entity_id="technology:neo4j",
                relation_type="RELATED_TO",
            )
        ],
    )
    monkeypatch.setattr(
        ingestion_module,
        "embed_texts",
        lambda texts: [[1.0, 0.0], [2.0, 0.0]],
    )
    monkeypatch.setattr(ingestion_module, "uuid4", lambda: "doc-123")
    monkeypatch.setattr(ingestion_module, "upsert_chunk_records", lambda records: len(records))
    monkeypatch.setattr(ingestion_module, "upsert_document_graph", lambda record: len(record.chunks))

    result = ingestion_module.ingest_document("note.txt", "text/plain", b"hello Neo4j RAG")

    assert result.filename == "note.txt"
    assert result.parsed is True
    assert result.document_id == "doc-123"
    assert result.stored_chunk_count == 2
    assert result.graph_chunk_count == 2
    assert result.entity_count == 2
    assert result.relation_count == 1


def test_ingest_document_returns_parsing_error(monkeypatch):
    from app.services.document_parser import DocumentParsingError

    monkeypatch.setattr(
        ingestion_module,
        "get_settings",
        lambda: SimpleNamespace(chunk_size=10, chunk_overlap=2),
    )

    def fake_parse_document(filename, content):
        raise DocumentParsingError("Unable to parse PDF file")

    monkeypatch.setattr(ingestion_module, "parse_document", fake_parse_document)

    result = ingestion_module.ingest_document("sample.pdf", "application/pdf", b"%PDF")

    assert result.parsed is False
    assert result.document_id is None
    assert result.stored_chunk_count == 0
    assert result.graph_chunk_count == 0
    assert result.entity_count == 0
    assert result.relation_count == 0