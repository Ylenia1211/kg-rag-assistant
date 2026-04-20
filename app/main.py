from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.core.logging import configure_logging
from app.core.neo4j import (
    close_neo4j_driver,
    ensure_neo4j_constraints,
    verify_neo4j_connectivity,
)
from app.core.qdrant import ensure_qdrant_collection, qdrant_healthcheck

from app.api.routes.documents import router as documents_router
from app.api.routes.search import router as search_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    ensure_qdrant_collection()
    ensure_neo4j_constraints()
    yield
    close_neo4j_driver()


app = FastAPI(
    title="KG RAG Assistant",
    version="0.1.0",
    lifespan=lifespan,
)
app.include_router(documents_router)
app.include_router(search_router)


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "KG RAG Assistant API"}


@app.get("/health")
def health() -> dict:
    qdrant_ok = qdrant_healthcheck()
    neo4j_ok = verify_neo4j_connectivity()
    ok = qdrant_ok and neo4j_ok

    return {
        "status": "ok" if ok else "degraded",
        "services": {
            "qdrant": qdrant_ok,
            "neo4j": neo4j_ok,
        },
    }