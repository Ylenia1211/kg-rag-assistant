from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import app.core.neo4j as neo4j_module


class FakeRecord:
    def __init__(self, payload):
        self._payload = payload

    def data(self):
        return self._payload


class FakeSession:
    def __init__(self, records=None):
        self.records = records or []
        self.run_calls = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return None

    def run(self, query, parameters):
        self.run_calls.append((query, parameters))
        return self.records


class FakeDriver:
    def __init__(self, session_obj=None):
        self._session_obj = session_obj or FakeSession()
        self.closed = False
        self.verify_connectivity = Mock()

    def session(self, database):
        self.database = database
        return self._session_obj

    def close(self):
        self.closed = True


def test_get_neo4j_driver_uses_settings(monkeypatch):
    fake_driver = FakeDriver()
    factory = Mock(return_value=fake_driver)

    monkeypatch.setattr(neo4j_module, "GraphDatabase", SimpleNamespace(driver=factory))
    monkeypatch.setattr(
        neo4j_module,
        "get_settings",
        lambda: SimpleNamespace(
            neo4j_uri="bolt://localhost:7687",
            neo4j_auth=lambda: ("neo4j", "password"),
        ),
    )

    neo4j_module.get_neo4j_driver.cache_clear()
    driver = neo4j_module.get_neo4j_driver()

    assert driver is fake_driver
    factory.assert_called_once_with("bolt://localhost:7687", auth=("neo4j", "password"))

    neo4j_module.get_neo4j_driver.cache_clear()


def test_verify_neo4j_connectivity_returns_true(monkeypatch):
    fake_driver = FakeDriver()
    monkeypatch.setattr(neo4j_module, "get_neo4j_driver", lambda: fake_driver)

    assert neo4j_module.verify_neo4j_connectivity() is True
    fake_driver.verify_connectivity.assert_called_once()


def test_verify_neo4j_connectivity_returns_false_on_error(monkeypatch):
    fake_driver = FakeDriver()
    fake_driver.verify_connectivity.side_effect = RuntimeError("boom")
    monkeypatch.setattr(neo4j_module, "get_neo4j_driver", lambda: fake_driver)

    assert neo4j_module.verify_neo4j_connectivity() is False


def test_run_cypher_returns_record_data(monkeypatch):
    session = FakeSession(records=[FakeRecord({"name": "Alice"}), FakeRecord({"name": "Bob"})])
    fake_driver = FakeDriver(session_obj=session)

    monkeypatch.setattr(
        neo4j_module,
        "get_settings",
        lambda: SimpleNamespace(neo4j_database="neo4j"),
    )
    monkeypatch.setattr(neo4j_module, "get_neo4j_driver", lambda: fake_driver)

    rows = neo4j_module.run_cypher("MATCH (n) RETURN n", {"limit": 2})

    assert rows == [{"name": "Alice"}, {"name": "Bob"}]
    assert session.run_calls == [("MATCH (n) RETURN n", {"limit": 2})]
    assert fake_driver.database == "neo4j"


def test_ensure_neo4j_constraints_runs_all_statements(monkeypatch):
    calls = []

    def fake_run_cypher(query, parameters=None):
        calls.append((query, parameters))
        return []

    monkeypatch.setattr(neo4j_module, "run_cypher", fake_run_cypher)

    neo4j_module.ensure_neo4j_constraints()

    assert len(calls) == 3
    assert "document_id_unique" in calls[0][0]
    assert "chunk_id_unique" in calls[1][0]
    assert "entity_id_unique" in calls[2][0]


def test_close_neo4j_driver_closes_and_clears_cache(monkeypatch):
    fake_driver = FakeDriver()
    getter = Mock(return_value=fake_driver)
    getter.cache_clear = Mock()

    monkeypatch.setattr(neo4j_module, "get_neo4j_driver", getter)

    neo4j_module.close_neo4j_driver()

    assert fake_driver.closed is True
    getter.cache_clear.assert_called_once()