# tests/unit/test_dev_console.py
from __future__ import annotations

from src.api.dev_console import dev_console_enabled


def test_dev_console_enabled_flag(monkeypatch):
    monkeypatch.delenv("VOX_DEV_CONSOLE", raising=False)
    assert dev_console_enabled() is False
    monkeypatch.setenv("VOX_DEV_CONSOLE", "1")
    assert dev_console_enabled() is True
