from __future__ import annotations

import json
import logging
from typing import Any

from app.core.config import get_settings


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, ensure_ascii=False)


def configure_logging() -> None:
    settings = get_settings()
    root_logger = logging.getLogger()

    if getattr(configure_logging, "_configured", False):
        return

    root_logger.handlers.clear()

    handler = logging.StreamHandler()
    if settings.is_local:
        formatter: logging.Formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        )
    else:
        formatter = JsonFormatter()

    handler.setFormatter(formatter)
    root_logger.addHandler(handler)
    root_logger.setLevel(logging.DEBUG if settings.app_debug else logging.INFO)

    configure_logging._configured = True


logger = logging.getLogger("kg_rag")