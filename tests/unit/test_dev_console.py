# tests/unit/test_dev_console.py
from __future__ import annotations

from src.api.dev_console import dev_console_enabled


def test_dev_console_enabled_flag(monkeypatch):
    monkeypatch.delenv("VOX_DEV_CONSOLE", raising=False)
    assert dev_console_enabled() is False
    monkeypatch.setenv("VOX_DEV_CONSOLE", "1")
    assert dev_console_enabled() is True


# append to tests/unit/test_dev_console.py
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.dev_console import dev_router


def test_dev_voice_page_served():
    app = FastAPI()
    app.include_router(dev_router)
    client = TestClient(app)
    resp = client.get("/dev/voice")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "Voice Dev Console" in resp.text
