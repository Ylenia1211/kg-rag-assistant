from __future__ import annotations

from functools import lru_cache

from qdrant_client import QdrantClient
from qdrant_client.http import models

from app.core.config import get_settings
from app.core.logging import logger


@lru_cache(maxsize=1)
def get_qdrant_client() -> QdrantClient:
    settings = get_settings()
    return QdrantClient(
        url=settings.qdrant_url,
        **settings.qdrant_auth_kwargs(),
    )


def ensure_qdrant_collection() -> None:
    settings = get_settings()
    client = get_qdrant_client()

    collections = client.get_collections()
    existing_names = {collection.name for collection in collections.collections}

    if settings.qdrant_collection in existing_names:
        logger.info("Qdrant collection already exists: %s", settings.qdrant_collection)
        return

    client.create_collection(
        collection_name=settings.qdrant_collection,
        vectors_config=models.VectorParams(
            size=settings.qdrant_vector_size,
            distance=models.Distance.COSINE,
        ),
    )
    logger.info("Created Qdrant collection: %s", settings.qdrant_collection)


def qdrant_healthcheck() -> bool:
    client = get_qdrant_client()
    try:
        client.get_collections()
        return True
    except Exception:
        logger.exception("Qdrant healthcheck failed")
        return False
