"""Per-tenant settings loader.

Each tenant has a YAML file at ``config/tenants/<slug>.yaml`` that overlays
the platform defaults. API keys are referenced **by env var name** (never
raw values) so secrets never land in version control or the DB.

Schema (all sections optional — global defaults fill the gaps):

    id: t_acme
    slug: acme
    name: Acme Telecom
    status: active                  # active | suspended
    default_language: hi
    webhook_secret_env: TENANT_ACME_WEBHOOK_SECRET

    pipeline:
      stt:
        provider: sarvam
        model: saaras:v2
        api_key_env: TENANT_ACME_SARVAM_KEY
      llm:
        provider: groq
        api_key_env: TENANT_ACME_GROQ_KEY
      tts:
        provider: sarvam
        voice_id: meera
        api_key_env: TENANT_ACME_SARVAM_KEY
      telephony:
        provider: twilio
        from_number: "+918888888888"
        account_sid_env: TENANT_ACME_TWILIO_SID
        auth_token_env: TENANT_ACME_TWILIO_TOKEN

    compliance:
      calling_hours: {start: "10:00", end: "19:00"}
      dnd_check_enabled: true

    crm:
      kind: fake | salesforce | hubspot
      endpoint_env: TENANT_ACME_CRM_URL
      token_env: TENANT_ACME_CRM_TOKEN

    whatsapp:
      provider: fake | meta_cloud
      phone_id_env: TENANT_ACME_WA_PHONE_ID
      token_env: TENANT_ACME_WA_TOKEN

    phone_numbers:           # Twilio numbers tied to this tenant
      - "+918888888888"
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, Field, ValidationError


class MissingEnvError(RuntimeError):
    """Raised when a tenant YAML references an env var that isn't set."""


# --- Sub-schemas --------------------------------------------------------


class TenantSTTConfig(BaseModel):
    provider: Optional[str] = None
    model: Optional[str] = None
    language: Optional[str] = None
    confidence_threshold: Optional[float] = None
    api_key_env: Optional[str] = None


class TenantLLMConfig(BaseModel):
    provider: Optional[str] = None
    model: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    response_format: Optional[str] = None
    api_key_env: Optional[str] = None


class TenantTTSConfig(BaseModel):
    provider: Optional[str] = None
    language: Optional[str] = None
    voice_id: Optional[str] = None
    speed: Optional[float] = None
    api_key_env: Optional[str] = None


class TenantTelephonyConfig(BaseModel):
    provider: Optional[str] = None
    from_number: Optional[str] = None
    webhook_base_url: Optional[str] = None
    account_sid_env: Optional[str] = None
    auth_token_env: Optional[str] = None


class TenantVectorStoreConfig(BaseModel):
    provider: Optional[str] = None
    index_path: Optional[str] = None       # auto-namespaced if unset
    embedding_dim: Optional[int] = None


class TenantPipelineConfig(BaseModel):
    stt: TenantSTTConfig = Field(default_factory=TenantSTTConfig)
    llm: TenantLLMConfig = Field(default_factory=TenantLLMConfig)
    tts: TenantTTSConfig = Field(default_factory=TenantTTSConfig)
    telephony: TenantTelephonyConfig = Field(default_factory=TenantTelephonyConfig)
    vector_store: TenantVectorStoreConfig = Field(default_factory=TenantVectorStoreConfig)


class TenantCompliance(BaseModel):
    calling_hours_start: Optional[str] = None
    calling_hours_end: Optional[str] = None
    dnd_check_enabled: Optional[bool] = None
    ai_disclosure: Optional[bool] = None
    max_retry_attempts: Optional[int] = None
    retry_interval_hours: Optional[int] = None


class TenantCRMConfig(BaseModel):
    kind: str = "fake"
    endpoint_env: Optional[str] = None
    token_env: Optional[str] = None


class TenantWhatsAppConfig(BaseModel):
    provider: str = "fake"
    phone_id_env: Optional[str] = None
    token_env: Optional[str] = None


class TenantSettings(BaseModel):
    """Validated tenant configuration loaded from YAML."""

    id: str = Field(min_length=1)
    slug: str = Field(min_length=1, max_length=63)
    name: str = Field(min_length=1)
    status: str = "active"
    default_language: str = "hi"
    webhook_secret_env: Optional[str] = None

    pipeline: TenantPipelineConfig = Field(default_factory=TenantPipelineConfig)
    compliance: TenantCompliance = Field(default_factory=TenantCompliance)
    crm: TenantCRMConfig = Field(default_factory=TenantCRMConfig)
    whatsapp: TenantWhatsAppConfig = Field(default_factory=TenantWhatsAppConfig)
    phone_numbers: list[str] = Field(default_factory=list)

    # --- secret resolution ---------------------------------------------

    def secret(self, env_var: Optional[str]) -> Optional[str]:
        """Resolve a referenced env var name. Returns None if name is None."""
        if env_var is None:
            return None
        value = os.environ.get(env_var)
        if value is None:
            raise MissingEnvError(
                f"tenant {self.slug!r} references env var {env_var!r} which is not set"
            )
        return value


# --- Loader -------------------------------------------------------------


_TENANT_DIR_DEFAULT = Path("config/tenants")


def _resolve_dir(tenant_dir: Optional[Path]) -> Path:
    return Path(tenant_dir or os.environ.get("VOX_TENANT_DIR") or _TENANT_DIR_DEFAULT)


def load_tenant(slug: str, tenant_dir: Optional[Path] = None) -> TenantSettings:
    """Load + validate one tenant by slug."""
    base = _resolve_dir(tenant_dir)
    path = base / f"{slug}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"tenant config not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: top-level YAML must be a mapping")
    try:
        return TenantSettings(**data)
    except ValidationError as e:
        raise ValueError(f"{path}: invalid tenant config: {e}") from e


def discover_tenant_slugs(tenant_dir: Optional[Path] = None) -> list[str]:
    """Return slugs for every YAML in the tenant config directory."""
    base = _resolve_dir(tenant_dir)
    if not base.exists():
        return []
    return sorted(p.stem for p in base.glob("*.yaml"))


def load_all_tenants(tenant_dir: Optional[Path] = None) -> dict[str, TenantSettings]:
    """Load every tenant in ``config/tenants/``. Returns ``{slug: settings}``."""
    return {slug: load_tenant(slug, tenant_dir) for slug in discover_tenant_slugs(tenant_dir)}


# --- Merge with global defaults ----------------------------------------


def merge_provider_config(
    tenant_layer: BaseModel,
    global_layer: dict[str, Any],
    api_key: Optional[str] = None,
) -> dict[str, Any]:
    """Overlay tenant-set fields onto the global default dict.

    Only non-None tenant fields override globals — that's the "partial
    override" semantics promised in the plan.
    """
    out = dict(global_layer)
    for k, v in tenant_layer.model_dump().items():
        if v is not None and not k.endswith("_env"):
            out[k] = v
    if api_key is not None:
        out["api_key"] = api_key
    return out
