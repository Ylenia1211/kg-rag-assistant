from __future__ import annotations

from dataclasses import dataclass
from uuid import uuid4

from app.core.config import get_settings
from app.repositories.neo4j_repository import (
    Neo4jChunkNode,
    Neo4jDocumentGraphRecord,
    Neo4jEntityNode,
    Neo4jRelationEdge,
    upsert_document_graph,
)
from app.repositories.qdrant_repository import QdrantChunkRecord, upsert_chunk_records
from app.services.document_parser import DocumentParsingError, parse_document
from app.services.embedding_service import EmbeddingServiceError, embed_texts
from app.services.entity_extraction_service import extract_entities, infer_relations
from app.services.text_chunker import TextChunk, TextChunkingError, chunk_text

PREVIEW_LENGTH = 300


@dataclass(slots=True)
class IngestionResult:
    filename: str
    content_type: str
    size_bytes: int
    parsed: bool
    text_length: int
    preview: str
    error: str | None
    chunks: list[TextChunk]
    document_id: str | None
    stored_chunk_count: int
    graph_chunk_count: int
    entity_count: int
    relation_count: int


def build_qdrant_records(
    document_id: str,
    filename: str,
    chunks: list[TextChunk],
    embeddings: list[list[float]],
) -> list[QdrantChunkRecord]:
    if len(chunks) != len(embeddings):
        raise EmbeddingServiceError("Chunk and embedding counts do not match")

    records: list[QdrantChunkRecord] = []
    for chunk, embedding in zip(chunks, embeddings, strict=True):
        records.append(
            QdrantChunkRecord(
                document_id=document_id,
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                embedding=embedding,
                start_char=chunk.start_char,
                end_char=chunk.end_char,
                source_filename=filename,
            )
        )
    return records


def build_neo4j_graph_record(
    document_id: str,
    filename: str,
    content_type: str,
    text_length: int,
    chunks: list[TextChunk],
    entities,
    relations,
) -> Neo4jDocumentGraphRecord:
    return Neo4jDocumentGraphRecord(
        document_id=document_id,
        filename=filename,
        content_type=content_type,
        text_length=text_length,
        chunks=[
            Neo4jChunkNode(
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                start_char=chunk.start_char,
                end_char=chunk.end_char,
            )
            for chunk in chunks
        ],
        entities=[
            Neo4jEntityNode(
                entity_id=entity.entity_id,
                name=entity.name,
                entity_type=entity.entity_type,
            )
            for entity in entities
        ],
        relations=[
            Neo4jRelationEdge(
                source_entity_id=relation.source_entity_id,
                target_entity_id=relation.target_entity_id,
                relation_type=relation.relation_type,
            )
            for relation in relations
        ],
    )


def ingest_document(filename: str, content_type: str, content: bytes) -> IngestionResult:
    settings = get_settings()
    size_bytes = len(content)

    try:
        extracted_text = parse_document(filename, content)
    except DocumentParsingError as exc:
        return IngestionResult(
            filename=filename,
            content_type=content_type,
            size_bytes=size_bytes,
            parsed=False,
            text_length=0,
            preview="",
            error=str(exc),
            chunks=[],
            document_id=None,
            stored_chunk_count=0,
            graph_chunk_count=0,
            entity_count=0,
            relation_count=0,
        )

    try:
        chunks = chunk_text(
            extracted_text,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
    except TextChunkingError as exc:
        return IngestionResult(
            filename=filename,
            content_type=content_type,
            size_bytes=size_bytes,
            parsed=False,
            text_length=len(extracted_text),
            preview=extracted_text[:PREVIEW_LENGTH],
            error=str(exc),
            chunks=[],
            document_id=None,
            stored_chunk_count=0,
            graph_chunk_count=0,
            entity_count=0,
            relation_count=0,
        )

    entities = extract_entities(extracted_text)
    relations = infer_relations(entities)

    if not chunks:
        return IngestionResult(
            filename=filename,
            content_type=content_type,
            size_bytes=size_bytes,
            parsed=True,
            text_length=len(extracted_text),
            preview=extracted_text[:PREVIEW_LENGTH],
            error=None,
            chunks=[],
            document_id=str(uuid4()),
            stored_chunk_count=0,
            graph_chunk_count=0,
            entity_count=len(entities),
            relation_count=len(relations),
        )

    try:
        embeddings = embed_texts([chunk.text for chunk in chunks])
        document_id = str(uuid4())

        qdrant_records = build_qdrant_records(
            document_id=document_id,
            filename=filename,
            chunks=chunks,
            embeddings=embeddings,
        )
        stored_chunk_count = upsert_chunk_records(qdrant_records)

        neo4j_record = build_neo4j_graph_record(
            document_id=document_id,
            filename=filename,
            content_type=content_type,
            text_length=len(extracted_text),
            chunks=chunks,
            entities=entities,
            relations=relations,
        )
        graph_chunk_count = upsert_document_graph(neo4j_record)

    except EmbeddingServiceError as exc:
        return IngestionResult(
            filename=filename,
            content_type=content_type,
            size_bytes=size_bytes,
            parsed=False,
            text_length=len(extracted_text),
            preview=extracted_text[:PREVIEW_LENGTH],
            error=str(exc),
            chunks=chunks,
            document_id=None,
            stored_chunk_count=0,
            graph_chunk_count=0,
            entity_count=len(entities),
            relation_count=len(relations),
        )

    return IngestionResult(
        filename=filename,
        content_type=content_type,
        size_bytes=size_bytes,
        parsed=True,
        text_length=len(extracted_text),
        preview=extracted_text[:PREVIEW_LENGTH],
        error=None,
        chunks=chunks,
        document_id=document_id,
        stored_chunk_count=stored_chunk_count,
        graph_chunk_count=graph_chunk_count,
        entity_count=len(entities),
        relation_count=len(relations),
    )