from __future__ import annotations

from dataclasses import dataclass

from app.core.neo4j import run_cypher


@dataclass(slots=True)
class Neo4jChunkNode:
    chunk_id: int
    text: str
    start_char: int
    end_char: int


@dataclass(slots=True)
class Neo4jEntityNode:
    entity_id: str
    name: str
    entity_type: str


@dataclass(slots=True)
class Neo4jRelationEdge:
    source_entity_id: str
    target_entity_id: str
    relation_type: str


@dataclass(slots=True)
class Neo4jDocumentGraphRecord:
    document_id: str
    filename: str
    content_type: str
    text_length: int
    chunks: list[Neo4jChunkNode]
    entities: list[Neo4jEntityNode]
    relations: list[Neo4jRelationEdge]


def upsert_document_graph(record: Neo4jDocumentGraphRecord) -> int:
    run_cypher(
        """
        MERGE (d:Document {id: $document_id})
        SET d.filename = $filename,
            d.content_type = $content_type,
            d.text_length = $text_length
        """,
        {
            "document_id": record.document_id,
            "filename": record.filename,
            "content_type": record.content_type,
            "text_length": record.text_length,
        },
    )

    for chunk in record.chunks:
        run_cypher(
            """
            MERGE (c:Chunk {id: $chunk_node_id})
            SET c.chunk_id = $chunk_id,
                c.text = $text,
                c.start_char = $start_char,
                c.end_char = $end_char,
                c.document_id = $document_id
            WITH c
            MATCH (d:Document {id: $document_id})
            MERGE (d)-[:HAS_CHUNK]->(c)
            """,
            {
                "chunk_node_id": f"{record.document_id}:{chunk.chunk_id}",
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
                "start_char": chunk.start_char,
                "end_char": chunk.end_char,
                "document_id": record.document_id,
            },
        )

    for entity in record.entities:
        run_cypher(
            """
            MERGE (e:Entity {id: $entity_id})
            SET e.name = $name,
                e.entity_type = $entity_type
            WITH e
            MATCH (d:Document {id: $document_id})
            MERGE (d)-[:MENTIONS]->(e)
            """,
            {
                "entity_id": entity.entity_id,
                "name": entity.name,
                "entity_type": entity.entity_type,
                "document_id": record.document_id,
            },
        )

    for relation in record.relations:
        run_cypher(
            """
            MATCH (source:Entity {id: $source_entity_id})
            MATCH (target:Entity {id: $target_entity_id})
            MERGE (source)-[:RELATED_TO {type: $relation_type}]->(target)
            """,
            {
                "source_entity_id": relation.source_entity_id,
                "target_entity_id": relation.target_entity_id,
                "relation_type": relation.relation_type,
            },
        )

    return len(record.chunks)
    

def get_document_chunks(document_id: str) -> list[dict]:
    return run_cypher(
        """
        MATCH (d:Document {id: $document_id})-[:HAS_CHUNK]->(c:Chunk)
        RETURN d.id AS document_id,
               c.chunk_id AS chunk_id,
               c.text AS text,
               c.start_char AS start_char,
               c.end_char AS end_char
        ORDER BY c.chunk_id ASC
        """,
        {"document_id": document_id},
    )


def get_related_entities(document_id: str) -> list[dict]:
    return run_cypher(
        """
        MATCH (d:Document {id: $document_id})-[:MENTIONS]->(e:Entity)
        RETURN d.id AS document_id,
               e.id AS entity_id,
               e.name AS entity_name,
               e.entity_type AS entity_type
        ORDER BY e.name ASC
        """,
        {"document_id": document_id},
    )