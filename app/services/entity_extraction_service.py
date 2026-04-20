from __future__ import annotations

from dataclasses import dataclass
import re


@dataclass(slots=True)
class ExtractedEntity:
    entity_id: str
    name: str
    entity_type: str


@dataclass(slots=True)
class ExtractedRelation:
    source_entity_id: str
    target_entity_id: str
    relation_type: str


ENTITY_PATTERNS: list[tuple[str, str]] = [
    (r"\bNeo4j\b", "Technology"),
    (r"\bQdrant\b", "Technology"),
    (r"\bFastAPI\b", "Technology"),
    (r"\bStreamlit\b", "Technology"),
    (r"\bOpenAI\b", "Organization"),
    (r"\bRAG\b", "Concept"),
    (r"\bKG\b", "Concept"),
    (r"\bKnowledge Graph\b", "Concept"),
]


def slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    return value.strip("-")


def extract_entities(text: str) -> list[ExtractedEntity]:
    found: dict[str, ExtractedEntity] = {}

    for pattern, entity_type in ENTITY_PATTERNS:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            raw_name = match.group(0)
            canonical_name = raw_name.strip()
            entity_id = f"{entity_type.lower()}:{slugify(canonical_name)}"

            if entity_id not in found:
                found[entity_id] = ExtractedEntity(
                    entity_id=entity_id,
                    name=canonical_name,
                    entity_type=entity_type,
                )

    return list(found.values())


def infer_relations(entities: list[ExtractedEntity]) -> list[ExtractedRelation]:
    relations: list[ExtractedRelation] = []

    ids_by_name = {entity.name.lower(): entity for entity in entities}

    if "rag" in ids_by_name and "neo4j" in ids_by_name:
        relations.append(
            ExtractedRelation(
                source_entity_id=ids_by_name["rag"].entity_id,
                target_entity_id=ids_by_name["neo4j"].entity_id,
                relation_type="RELATED_TO",
            )
        )

    if "rag" in ids_by_name and "qdrant" in ids_by_name:
        relations.append(
            ExtractedRelation(
                source_entity_id=ids_by_name["rag"].entity_id,
                target_entity_id=ids_by_name["qdrant"].entity_id,
                relation_type="RELATED_TO",
            )
        )

    if "kg" in ids_by_name and "neo4j" in ids_by_name:
        relations.append(
            ExtractedRelation(
                source_entity_id=ids_by_name["kg"].entity_id,
                target_entity_id=ids_by_name["neo4j"].entity_id,
                relation_type="RELATED_TO",
            )
        )

    return relations