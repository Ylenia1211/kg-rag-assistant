from __future__ import annotations

from functools import lru_cache
from typing import Any

from neo4j import Driver, GraphDatabase

from app.core.config import get_settings
from app.core.logging import logger


@lru_cache(maxsize=1)
def get_neo4j_driver() -> Driver:
    settings = get_settings()
    driver = GraphDatabase.driver(
        settings.neo4j_uri,
        auth=settings.neo4j_auth(),
    )
    return driver


def verify_neo4j_connectivity() -> bool:
    driver = get_neo4j_driver()
    try:
        driver.verify_connectivity()
        return True
    except Exception:
        logger.exception("Neo4j connectivity check failed")
        return False


def run_cypher(query: str, parameters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    settings = get_settings()
    driver = get_neo4j_driver()

    with driver.session(database=settings.neo4j_database) as session:
        result = session.run(query, parameters or {})
        return [record.data() for record in result]


def ensure_neo4j_constraints() -> None:
    statements = [
        "CREATE CONSTRAINT document_id_unique IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE",
        "CREATE CONSTRAINT chunk_id_unique IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE",
        "CREATE CONSTRAINT entity_id_unique IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE",
    ]

    for statement in statements:
        run_cypher(statement)

    logger.info("Ensured Neo4j constraints")


def close_neo4j_driver() -> None:
    try:
        driver = get_neo4j_driver()
        driver.close()
    finally:
        get_neo4j_driver.cache_clear()