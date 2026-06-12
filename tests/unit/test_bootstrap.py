"""Bootstrap tests for src/main.py lifespan logic.

Validates admin-token parsing. Tenant loading moved from YAML-on-boot to the
DB resolver + seed (see test_seed.py / test_db_resolver.py).
"""

from __future__ import annotations

import pytest

from src.config_tenant import discover_tenant_slugs, load_tenant
from src.main import _admin_tokens_from_env


def test_example_tenant_yaml_still_loads() -> None:
    """The shipped example tenant must still parse (it's what the seed reads)."""
    from pathlib import Path
    assert "example" in discover_tenant_slugs(Path("config/tenants"))
    assert load_tenant("example", Path("config/tenants")).name == "Example Telecom"


def test_admin_tokens_from_env_empty(monkeypatch) -> None:
    monkeypatch.delenv("VOX_ADMIN_TOKENS", raising=False)
    assert _admin_tokens_from_env() == []


def test_admin_tokens_from_env_comma_separated(monkeypatch) -> None:
    monkeypatch.setenv("VOX_ADMIN_TOKENS", "tok-a, tok-b , tok-c")
    assert _admin_tokens_from_env() == ["tok-a", "tok-b", "tok-c"]


def test_admin_tokens_from_env_skips_empty_entries(monkeypatch) -> None:
    monkeypatch.setenv("VOX_ADMIN_TOKENS", "a,,b,")
    assert _admin_tokens_from_env() == ["a", "b"]
