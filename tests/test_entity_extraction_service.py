from __future__ import annotations

from app.services.entity_extraction_service import (
    extract_entities,
    infer_relations,
    slugify,
)


def test_slugify_normalizes_text():
    assert slugify("Knowledge Graph") == "knowledge-graph"
    assert slugify("Neo4j!") == "neo4j"


def test_extract_entities_finds_known_entities():
    text = "Questo progetto usa Neo4j, Qdrant, FastAPI e Streamlit per fare RAG con KG."

    entities = extract_entities(text)
    entity_names = sorted(entity.name.lower() for entity in entities)

    assert "neo4j" in entity_names
    assert "qdrant" in entity_names
    assert "fastapi" in entity_names
    assert "streamlit" in entity_names
    assert "rag" in entity_names
    assert "kg" in entity_names


def test_extract_entities_deduplicates_entities():
    text = "Neo4j è usato. Neo4j compare due volte."

    entities = extract_entities(text)

    neo4j_entities = [entity for entity in entities if entity.name.lower() == "neo4j"]
    assert len(neo4j_entities) == 1


def test_infer_relations_creates_basic_links():
    text = "Usiamo RAG con Neo4j e Qdrant. Usiamo anche KG con Neo4j."
    entities = extract_entities(text)

    relations = infer_relations(entities)
    triples = {(r.source_entity_id, r.target_entity_id, r.relation_type) for r in relations}

    assert ("concept:rag", "technology:neo4j", "RELATED_TO") in triples
    assert ("concept:rag", "technology:qdrant", "RELATED_TO") in triples
    assert ("concept:kg", "technology:neo4j", "RELATED_TO") in triples