"""Structured JSON logging setup.

Single entry point: ``configure_logging(level)`` — call once at app startup.
"""

from __future__ import annotations

import logging
import sys

from pythonjsonlogger import jsonlogger


def configure_logging(level: str = "INFO") -> None:
    """Configure root logger with JSON output to stdout.

    Idempotent — safe to call multiple times (e.g. in tests).
    """
    root = logging.getLogger()
    root.setLevel(level.upper())

    # Remove any existing handlers so we don't duplicate output.
    for h in list(root.handlers):
        root.removeHandler(h)

    handler = logging.StreamHandler(sys.stdout)
    formatter = jsonlogger.JsonFormatter(
        "%(asctime)s %(name)s %(levelname)s %(message)s",
        rename_fields={"asctime": "timestamp", "levelname": "level"},
    )
    handler.setFormatter(formatter)
    root.addHandler(handler)

    # Quiet down noisy libraries.
    for noisy in ("uvicorn.access", "httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
