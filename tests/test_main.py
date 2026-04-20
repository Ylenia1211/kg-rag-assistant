from __future__ import annotations

from fastapi.testclient import TestClient


def test_root_endpoint(monkeypatch):
    import app.main as main_module

    monkeypatch.setattr(main_module, "configure_logging", lambda: None)
    monkeypatch.setattr(main_module, "ensure_qdrant_collection", lambda: None)
    monkeypatch.setattr(main_module, "ensure_neo4j_constraints", lambda: None)
    monkeypatch.setattr(main_module, "close_neo4j_driver", lambda: None)

    with TestClient(main_module.app) as client:
        response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {"message": "KG RAG Assistant API"}


def test_health_ok(monkeypatch):
    import app.main as main_module

    monkeypatch.setattr(main_module, "configure_logging", lambda: None)
    monkeypatch.setattr(main_module, "ensure_qdrant_collection", lambda: None)
    monkeypatch.setattr(main_module, "ensure_neo4j_constraints", lambda: None)
    monkeypatch.setattr(main_module, "close_neo4j_driver", lambda: None)
    monkeypatch.setattr(main_module, "qdrant_healthcheck", lambda: True)
    monkeypatch.setattr(main_module, "verify_neo4j_connectivity", lambda: True)

    with TestClient(main_module.app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "services": {"qdrant": True, "neo4j": True},
    }


def test_health_degraded(monkeypatch):
    import app.main as main_module

    monkeypatch.setattr(main_module, "configure_logging", lambda: None)
    monkeypatch.setattr(main_module, "ensure_qdrant_collection", lambda: None)
    monkeypatch.setattr(main_module, "ensure_neo4j_constraints", lambda: None)
    monkeypatch.setattr(main_module, "close_neo4j_driver", lambda: None)
    monkeypatch.setattr(main_module, "qdrant_healthcheck", lambda: False)
    monkeypatch.setattr(main_module, "verify_neo4j_connectivity", lambda: True)

    with TestClient(main_module.app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "degraded",
        "services": {"qdrant": False, "neo4j": True},
    }


def test_lifespan_calls_bootstrap(monkeypatch):
    import app.main as main_module

    calls = []

    monkeypatch.setattr(main_module, "configure_logging", lambda: calls.append("logging"))
    monkeypatch.setattr(main_module, "ensure_qdrant_collection", lambda: calls.append("qdrant"))
    monkeypatch.setattr(main_module, "ensure_neo4j_constraints", lambda: calls.append("neo4j"))
    monkeypatch.setattr(main_module, "close_neo4j_driver", lambda: calls.append("close"))

    with TestClient(main_module.app):
        pass

    assert calls == ["logging", "qdrant", "neo4j", "close"]