from __future__ import annotations

import importlib
import json
import logging


def test_settings_reads_environment(monkeypatch):
    monkeypatch.setenv("APP_ENV", "dev")
    monkeypatch.setenv("APP_DEBUG", "false")
    monkeypatch.setenv("QDRANT_COLLECTION", "test_documents")
    monkeypatch.setenv("TOP_K_VECTOR", "9")

    import app.core.config as config_module

    config_module.get_settings.cache_clear()
    settings = config_module.get_settings()

    assert settings.app_env == "dev"
    assert settings.app_debug is False
    assert settings.qdrant_collection == "test_documents"
    assert settings.top_k_vector == 9

    config_module.get_settings.cache_clear()


def test_qdrant_auth_kwargs(monkeypatch):
    monkeypatch.setenv("QDRANT_API_KEY", "secret-key")

    import app.core.config as config_module

    config_module.get_settings.cache_clear()
    settings = config_module.get_settings()

    assert settings.qdrant_auth_kwargs() == {"api_key": "secret-key"}

    config_module.get_settings.cache_clear()


def test_neo4j_auth_tuple(monkeypatch):
    monkeypatch.setenv("NEO4J_USERNAME", "neo4j_user")
    monkeypatch.setenv("NEO4J_PASSWORD", "neo4j_pass")

    import app.core.config as config_module

    config_module.get_settings.cache_clear()
    settings = config_module.get_settings()

    assert settings.neo4j_auth() == ("neo4j_user", "neo4j_pass")

    config_module.get_settings.cache_clear()


def test_configure_logging_local_sets_root_level_and_handler(monkeypatch):
    monkeypatch.setenv("APP_ENV", "local")
    monkeypatch.setenv("APP_DEBUG", "true")

    import app.core.config as config_module
    import app.core.logging as logging_module

    config_module.get_settings.cache_clear()
    logging_module.configure_logging._configured = False

    logging_module.configure_logging()

    root_logger = logging.getLogger()
    assert root_logger.level == logging.DEBUG
    assert len(root_logger.handlers) == 1
    assert isinstance(root_logger.handlers[0].formatter, logging.Formatter)

    config_module.get_settings.cache_clear()
    logging_module.configure_logging._configured = False
    root_logger.handlers.clear()


def test_json_formatter_outputs_json():
    from app.core.logging import JsonFormatter

    formatter = JsonFormatter()
    record = logging.LogRecord(
        name="kg_rag",
        level=logging.INFO,
        pathname=__file__,
        lineno=10,
        msg="hello world",
        args=(),
        exc_info=None,
    )

    payload = json.loads(formatter.format(record))

    assert payload["level"] == "INFO"
    assert payload["logger"] == "kg_rag"
    assert payload["message"] == "hello world"
