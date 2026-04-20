from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from qdrant_client.http import models

from app.core.config import get_settings
from app.core.qdrant import get_qdrant_client


@dataclass(slots=True)
class QdrantChunkRecord:
    document_id: str
    chunk_id: int
    text: str
    embedding: list[float]
    start_char: int
    end_char: int
    source_filename: str


def build_qdrant_points(records: list[QdrantChunkRecord]) -> list[models.PointStruct]:
    points: list[models.PointStruct] = []

    for record in records:
        points.append(
            models.PointStruct(
                id=str(uuid4()),
                vector=record.embedding,
                payload={
                    "document_id": record.document_id,
                    "chunk_id": record.chunk_id,
                    "text": record.text,
                    "start_char": record.start_char,
                    "end_char": record.end_char,
                    "source_filename": record.source_filename,
                },
            )
        )

    return points


def upsert_chunk_records(records: list[QdrantChunkRecord]) -> int:
    if not records:
        return 0

    settings = get_settings()
    client = get_qdrant_client()
    points = build_qdrant_points(records)

    client.upsert(
        collection_name=settings.qdrant_collection,
        points=points,
        wait=True,
    )
    return len(points)


def search_similar_chunks(query_vector: list[float], limit: int | None = None) -> list[dict[str, Any]]:
    settings = get_settings()
    client = get_qdrant_client()
    search_limit = limit or settings.top_k_vector

    response = client.query_points(
        collection_name=settings.qdrant_collection,
        query=query_vector,
        limit=search_limit,
        with_payload=True,
    )

    points = getattr(response, "points", response)

    rows: list[dict[str, Any]] = []
    for item in points:
        rows.append(
            {
                "id": str(item.id),
                "score": item.score,
                "payload": item.payload or {},
            }
        )

    return rows