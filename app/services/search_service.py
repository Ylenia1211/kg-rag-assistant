from __future__ import annotations

from dataclasses import dataclass

from app.repositories.neo4j_repository import get_document_chunks, get_related_entities
from app.repositories.qdrant_repository import search_similar_chunks
from app.services.answer_service import generate_answer
from app.services.embedding_service import embed_texts


@dataclass(slots=True)
class SearchServiceResult:
    question: str
    answer: str
    vector_results: list[dict]
    graph_results: list[dict]
    entity_results: list[dict]


def search_knowledge_base(question: str, limit: int | None = None) -> SearchServiceResult:
    query_vector = embed_texts([question])[0]
    vector_results = search_similar_chunks(query_vector=query_vector, limit=limit)

    graph_results: list[dict] = []
    entity_results: list[dict] = []
    seen_document_ids: set[str] = set()

    for item in vector_results:
        payload = item.get("payload", {})
        document_id = payload.get("document_id")
        if not document_id or document_id in seen_document_ids:
            continue

        seen_document_ids.add(document_id)
        graph_results.extend(get_document_chunks(document_id))
        entity_results.extend(get_related_entities(document_id))

    answer = generate_answer(
        question=question,
        vector_results=vector_results,
        graph_results=graph_results,
    )

    return SearchServiceResult(
        question=question,
        answer=answer,
        vector_results=vector_results,
        graph_results=graph_results,
        entity_results=entity_results,
    )