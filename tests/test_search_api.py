from __future__ import annotations

from fastapi.testclient import TestClient


def test_search_endpoint_returns_answer_and_results(monkeypatch):
    import app.main as main_module
    import app.api.routes.search as search_module
    from app.services.search_service import SearchServiceResult

    monkeypatch.setattr(main_module, "configure_logging", lambda: None)
    monkeypatch.setattr(main_module, "ensure_qdrant_collection", lambda: None)
    monkeypatch.setattr(main_module, "ensure_neo4j_constraints", lambda: None)
    monkeypatch.setattr(main_module, "close_neo4j_driver", lambda: None)

    monkeypatch.setattr(
        search_module,
        "search_knowledge_base",
        lambda question, limit=None: SearchServiceResult(
            question=question,
            answer="Il progetto usa Neo4j, Qdrant, FastAPI e Streamlit.",
            vector_results=[
                {
                    "id": "p1",
                    "score": 0.91,
                    "payload": {
                        "document_id": "doc-1",
                        "chunk_id": 1,
                        "text": "Questo progetto usa Neo4j, Qdrant, FastAPI e Streamlit.",
                    },
                }
            ],
            graph_results=[
                {
                    "document_id": "doc-1",
                    "chunk_id": 1,
                    "text": "Questo progetto usa Neo4j, Qdrant, FastAPI e Streamlit.",
                    "start_char": 0,
                    "end_char": 58,
                }
            ],
            entity_results=[
                {
                    "document_id": "doc-1",
                    "entity_id": "technology:neo4j",
                    "entity_name": "Neo4j",
                    "entity_type": "Technology",
                },
                {
                    "document_id": "doc-1",
                    "entity_id": "technology:qdrant",
                    "entity_name": "Qdrant",
                    "entity_type": "Technology",
                },
            ],
        ),
    )

    with TestClient(main_module.app) as client:
        response = client.post(
            "/api/v1/search",
            json={"question": "Quali tecnologie usa il progetto?", "limit": 5},
        )

    assert response.status_code == 200
    assert response.json() == {
        "question": "Quali tecnologie usa il progetto?",
        "answer": "Il progetto usa Neo4j, Qdrant, FastAPI e Streamlit.",
        "vector_results": [
            {
                "id": "p1",
                "score": 0.91,
                "payload": {
                    "document_id": "doc-1",
                    "chunk_id": 1,
                    "text": "Questo progetto usa Neo4j, Qdrant, FastAPI e Streamlit.",
                },
            }
        ],
        "graph_results": [
            {
                "document_id": "doc-1",
                "chunk_id": 1,
                "text": "Questo progetto usa Neo4j, Qdrant, FastAPI e Streamlit.",
                "start_char": 0,
                "end_char": 58,
            }
        ],
        "entity_results": [
            {
                "document_id": "doc-1",
                "entity_id": "technology:neo4j",
                "entity_name": "Neo4j",
                "entity_type": "Technology",
            },
            {
                "document_id": "doc-1",
                "entity_id": "technology:qdrant",
                "entity_name": "Qdrant",
                "entity_type": "Technology",
            },
        ],
    }


def test_search_endpoint_validates_empty_question(monkeypatch):
    import app.main as main_module

    monkeypatch.setattr(main_module, "configure_logging", lambda: None)
    monkeypatch.setattr(main_module, "ensure_qdrant_collection", lambda: None)
    monkeypatch.setattr(main_module, "ensure_neo4j_constraints", lambda: None)
    monkeypatch.setattr(main_module, "close_neo4j_driver", lambda: None)

    with TestClient(main_module.app) as client:
        response = client.post(
            "/api/v1/search",
            json={"question": "", "limit": 5},
        )

    assert response.status_code == 422


def test_search_endpoint_validates_limit(monkeypatch):
    import app.main as main_module

    monkeypatch.setattr(main_module, "configure_logging", lambda: None)
    monkeypatch.setattr(main_module, "ensure_qdrant_collection", lambda: None)
    monkeypatch.setattr(main_module, "ensure_neo4j_constraints", lambda: None)
    monkeypatch.setattr(main_module, "close_neo4j_driver", lambda: None)

    with TestClient(main_module.app) as client:
        response = client.post(
            "/api/v1/search",
            json={"question": "ciao", "limit": 50},
        )

    assert response.status_code == 422