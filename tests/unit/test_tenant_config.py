from __future__ import annotations

from pathlib import Path

import pytest

from src.config_tenant import (
    MissingEnvError,
    TenantSettings,
    discover_tenant_slugs,
    load_all_tenants,
    load_tenant,
    merge_provider_config,
)


@pytest.fixture
def tenant_dir(tmp_path: Path) -> Path:
    d = tmp_path / "tenants"
    d.mkdir()
    return d


def _write(path: Path, body: str) -> None:
    path.write_text(body, encoding="utf-8")


def test_load_tenant_minimal(tenant_dir: Path) -> None:
    _write(tenant_dir / "acme.yaml", """
id: t_acme
slug: acme
name: Acme
""")
    t = load_tenant("acme", tenant_dir)
    assert t.id == "t_acme"
    assert t.slug == "acme"
    assert t.status == "active"
    assert t.default_language == "hi"


def test_load_tenant_full(tenant_dir: Path) -> None:
    _write(tenant_dir / "acme.yaml", """
id: t_acme
slug: acme
name: Acme Telecom
default_language: en
pipeline:
  stt: {provider: sarvam, api_key_env: ACME_SARVAM}
  llm: {provider: groq, api_key_env: ACME_GROQ}
phone_numbers: ["+918888888888", "+917777777777"]
""")
    t = load_tenant("acme", tenant_dir)
    assert t.default_language == "en"
    assert t.pipeline.stt.provider == "sarvam"
    assert t.pipeline.llm.api_key_env == "ACME_GROQ"
    assert t.phone_numbers == ["+918888888888", "+917777777777"]


def test_load_tenant_unknown_slug_raises(tenant_dir: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_tenant("missing", tenant_dir)


def test_load_tenant_rejects_non_mapping(tenant_dir: Path) -> None:
    _write(tenant_dir / "bad.yaml", "- just a list\n- not a mapping\n")
    with pytest.raises(ValueError):
        load_tenant("bad", tenant_dir)


def test_load_tenant_validation_error_wraps(tenant_dir: Path) -> None:
    _write(tenant_dir / "bad.yaml", """
id: t_x
slug: ""
name: X
""")
    with pytest.raises(ValueError):
        load_tenant("bad", tenant_dir)


def test_discover_slugs_returns_sorted(tenant_dir: Path) -> None:
    for slug in ("globex", "acme", "stark"):
        _write(tenant_dir / f"{slug}.yaml", f"id: t_{slug}\nslug: {slug}\nname: {slug}\n")
    assert discover_tenant_slugs(tenant_dir) == ["acme", "globex", "stark"]


def test_discover_slugs_empty_dir(tenant_dir: Path) -> None:
    assert discover_tenant_slugs(tenant_dir) == []


def test_discover_slugs_missing_dir(tmp_path: Path) -> None:
    assert discover_tenant_slugs(tmp_path / "does-not-exist") == []


def test_load_all_tenants(tenant_dir: Path) -> None:
    _write(tenant_dir / "acme.yaml", "id: t_acme\nslug: acme\nname: Acme\n")
    _write(tenant_dir / "globex.yaml", "id: t_globex\nslug: globex\nname: Globex\n")
    all_t = load_all_tenants(tenant_dir)
    assert set(all_t.keys()) == {"acme", "globex"}
    assert all_t["acme"].id == "t_acme"


def test_secret_resolution_success(monkeypatch, tenant_dir: Path) -> None:
    _write(tenant_dir / "acme.yaml", """
id: t_acme
slug: acme
name: Acme
pipeline:
  stt: {provider: sarvam, api_key_env: ACME_SARVAM}
""")
    monkeypatch.setenv("ACME_SARVAM", "real-key-value")
    t = load_tenant("acme", tenant_dir)
    assert t.secret(t.pipeline.stt.api_key_env) == "real-key-value"


def test_secret_resolution_missing_env_raises(monkeypatch, tenant_dir: Path) -> None:
    _write(tenant_dir / "acme.yaml", """
id: t_acme
slug: acme
name: Acme
pipeline:
  stt: {api_key_env: NEVER_SET}
""")
    monkeypatch.delenv("NEVER_SET", raising=False)
    t = load_tenant("acme", tenant_dir)
    with pytest.raises(MissingEnvError, match="NEVER_SET"):
        t.secret(t.pipeline.stt.api_key_env)


def test_secret_returns_none_when_env_name_is_none() -> None:
    t = TenantSettings(id="t_x", slug="x", name="X")
    assert t.secret(None) is None


def test_merge_provider_config_overrides_only_set_fields() -> None:
    from src.config_tenant import TenantSTTConfig

    tenant = TenantSTTConfig(provider="sarvam", api_key_env="X")
    global_layer = {"provider": "default", "language": "hi-IN", "model": "saaras:v2"}
    merged = merge_provider_config(tenant, global_layer, api_key="resolved-key")
    assert merged["provider"] == "sarvam"
    assert merged["language"] == "hi-IN"
    assert merged["model"] == "saaras:v2"
    assert merged["api_key"] == "resolved-key"
    # ``api_key_env`` is metadata, not provider config
    assert "api_key_env" not in merged


def test_tenant_dir_override_via_env(monkeypatch, tenant_dir: Path) -> None:
    _write(tenant_dir / "x.yaml", "id: t_x\nslug: x\nname: X\n")
    monkeypatch.setenv("VOX_TENANT_DIR", str(tenant_dir))
    assert discover_tenant_slugs() == ["x"]


def test_example_yaml_file_loads() -> None:
    """Sanity-check the shipped example.yaml is structurally valid."""
    t = load_tenant("example", tenant_dir=Path("config/tenants"))
    assert t.slug == "example"
    assert t.pipeline.stt.provider == "sarvam"
