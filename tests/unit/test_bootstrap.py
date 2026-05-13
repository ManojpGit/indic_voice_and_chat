"""Bootstrap tests for src/main.py lifespan logic.

Validates that tenant discovery + resolver wiring happens correctly on
startup, without spinning up real Redis / Postgres.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.main import _admin_tokens_from_env, _load_tenants


def test_load_tenants_picks_up_example_yaml() -> None:
    """The shipped example tenant must load cleanly."""
    out = _load_tenants(Path("config/tenants"))
    assert "example" in out
    assert out["example"].name == "Example Telecom"


def test_load_tenants_empty_dir(tmp_path: Path) -> None:
    assert _load_tenants(tmp_path) == {}


def test_admin_tokens_from_env_empty(monkeypatch) -> None:
    monkeypatch.delenv("VOX_ADMIN_TOKENS", raising=False)
    assert _admin_tokens_from_env() == []


def test_admin_tokens_from_env_comma_separated(monkeypatch) -> None:
    monkeypatch.setenv("VOX_ADMIN_TOKENS", "tok-a, tok-b , tok-c")
    assert _admin_tokens_from_env() == ["tok-a", "tok-b", "tok-c"]


def test_admin_tokens_from_env_skips_empty_entries(monkeypatch) -> None:
    monkeypatch.setenv("VOX_ADMIN_TOKENS", "a,,b,")
    assert _admin_tokens_from_env() == ["a", "b"]
