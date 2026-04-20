from __future__ import annotations

import streamlit_app.Home as home_module


class FakeResponse:
    def __init__(self, payload: dict, should_raise: bool = False):
        self._payload = payload
        self._should_raise = should_raise

    def raise_for_status(self) -> None:
        if self._should_raise:
            raise RuntimeError("http error")

    def json(self) -> dict:
        return self._payload


def test_fetch_backend_health_success(monkeypatch):
    def fake_get(url: str, timeout: float):
        assert url == "http://localhost:8000/health"
        assert timeout == 5.0
        return FakeResponse(
            {
                "status": "ok",
                "services": {"qdrant": True, "neo4j": True},
            }
        )

    monkeypatch.setattr(home_module.httpx, "get", fake_get)
    home_module.fetch_backend_health.clear()

    payload = home_module.fetch_backend_health("http://localhost:8000")

    assert payload == {
        "reachable": True,
        "status": "ok",
        "services": {"qdrant": True, "neo4j": True},
        "error": None,
    }


def test_fetch_backend_health_failure(monkeypatch):
    def fake_get(url: str, timeout: float):
        raise RuntimeError("connection refused")

    monkeypatch.setattr(home_module.httpx, "get", fake_get)
    home_module.fetch_backend_health.clear()

    payload = home_module.fetch_backend_health("http://localhost:8000")

    assert payload["reachable"] is False
    assert payload["status"] == "offline"
    assert payload["services"] == {}
    assert "connection refused" in payload["error"]