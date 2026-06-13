"""Register Tenant endpoint (``POST /api/v1/tenants``) — admin-authed.

A CRM/partner self-registers a tenant through this API instead of dropping a
YAML file. The body carries **provider choices** for STT/LLM/TTS/realtime (which
use the shared master keys — no keys accepted here) and the **telephony**
credentials (the only per-tenant secrets — encrypted at rest into
``tenant_secrets``). We build the same ``TenantPipelineConfig`` the YAML path
produced, persist the tenant + phone numbers + telephony secrets, issue one API
token (returned once, stored only as a hash), and refresh the live resolver.
"""

from __future__ import annotations

import logging
import re
import secrets as pysecrets
import uuid
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.deps import get_db_session
from src.auth import secrets as crypto
from src.auth.context import hash_api_token
from src.auth.middleware import require_admin
from src.config_tenant import (
    TenantPipelineConfig,
    TenantRealtimeConfig,
    TenantSTTConfig,
    TenantTelephonyConfig,
    TenantTTSConfig,
)
from src.config_tenant import TenantLLMConfig as _LLM
from src.models.tenant import Tenant, TenantApiKey, TenantPhoneNumber, TenantSecret

log = logging.getLogger(__name__)
router = APIRouter(prefix="/tenants", tags=["tenants"])

# Shared master-key env var per provider. STT/LLM/TTS/realtime resolve their key
# from these platform env vars (via TenantContext.secret's fallback to os.environ);
# they are NEVER stored per tenant. Telephony keys are the per-tenant exception.
_MASTER_KEY_ENV = {
    "sarvam": "SARVAM_API_KEY",
    "groq": "GROQ_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "gemini_live": "GEMINI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "deepgram": "DEEPGRAM_API_KEY",
}


# --- Schemas ------------------------------------------------------------


class LayerChoice(BaseModel):
    """Provider choice for one cascade layer — no keys (uses master keys)."""
    provider: str
    model: Optional[str] = None
    language: Optional[str] = None
    voice_id: Optional[str] = None  # tts only
    speed: Optional[float] = None   # tts only


class RealtimeChoice(BaseModel):
    provider: str
    model: Optional[str] = None
    voice: Optional[str] = None
    language_code: Optional[str] = None


class TelephonyConfigIn(BaseModel):
    provider: str
    from_number: Optional[str] = None
    webhook_base_url: Optional[str] = None
    # Telephony credentials — the ONLY per-tenant secrets. Encrypted at rest.
    # e.g. {"account_sid": "AC...", "auth_token": "..."}. Optional for providers
    # (Stringee) whose adapter reads its keys from the platform env directly.
    keys: dict[str, str] = Field(default_factory=dict)
    phone_numbers: list[str] = Field(default_factory=list)


class RegisterTenantRequest(BaseModel):
    name: str = Field(min_length=1)
    slug: Optional[str] = None
    timezone: str = "Asia/Kolkata"
    default_language: str = "hi"
    mode: str = Field(default="layered", pattern="^(layered|s2s)$")
    max_concurrent_calls: int = Field(default=1, ge=1)
    stt: Optional[LayerChoice] = None
    llm: Optional[LayerChoice] = None
    tts: Optional[LayerChoice] = None
    realtime: Optional[RealtimeChoice] = None
    telephony: TelephonyConfigIn


class RegisterTenantResponse(BaseModel):
    tenant_id: str
    slug: str
    api_token: str


# --- Helpers ------------------------------------------------------------


def _slugify(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return s or f"tenant-{uuid.uuid4().hex[:8]}"


def _layer_key_env(provider: Optional[str]) -> Optional[str]:
    return _MASTER_KEY_ENV.get(provider) if provider else None


# --- Route --------------------------------------------------------------


@router.post("", response_model=RegisterTenantResponse, status_code=201)
async def register_tenant(
    req: RegisterTenantRequest,
    request: Request,
    session: AsyncSession = Depends(get_db_session),
    _: None = Depends(require_admin),
) -> RegisterTenantResponse:
    slug = req.slug or _slugify(req.name)
    existing = (await session.execute(
        select(Tenant.id).where(Tenant.slug == slug)
    )).first()
    if existing is not None:
        raise HTTPException(status_code=409, detail=f"tenant slug {slug!r} already exists")

    tenant_id = f"t_{uuid.uuid4().hex[:16]}"

    # Telephony secrets keyed by synthetic names that pipeline_config references;
    # TenantContext.secret(<name>) finds them in the decrypted secrets dict.
    tel = req.telephony
    sid_env = token_env = None
    secret_rows: list[tuple[str, str]] = []
    if tel.keys:
        if not crypto.has_key():
            raise HTTPException(
                status_code=503,
                detail="VOX_SECRET_KEY is not set — cannot encrypt telephony keys",
            )
        for logical, value in tel.keys.items():
            name = f"TENANT_{slug.upper().replace('-', '_')}_{logical.upper()}"
            secret_rows.append((name, value))
            if logical in ("account_sid", "sid"):
                sid_env = name
            elif logical in ("auth_token", "token"):
                token_env = name

    pipeline = TenantPipelineConfig(
        mode=req.mode,
        stt=TenantSTTConfig(
            provider=req.stt.provider if req.stt else None,
            model=req.stt.model if req.stt else None,
            language=req.stt.language if req.stt else None,
            api_key_env=_layer_key_env(req.stt.provider) if req.stt else None,
        ),
        llm=_LLM(
            provider=req.llm.provider if req.llm else None,
            model=req.llm.model if req.llm else None,
            api_key_env=_layer_key_env(req.llm.provider) if req.llm else None,
        ),
        tts=TenantTTSConfig(
            provider=req.tts.provider if req.tts else None,
            model=req.tts.model if req.tts else None,
            language=req.tts.language if req.tts else None,
            voice_id=req.tts.voice_id if req.tts else None,
            speed=req.tts.speed if req.tts else None,
            api_key_env=_layer_key_env(req.tts.provider) if req.tts else None,
        ),
        realtime=TenantRealtimeConfig(
            provider=req.realtime.provider,
            model=req.realtime.model,
            voice=req.realtime.voice,
            language_code=req.realtime.language_code,
            api_key_env=_layer_key_env(req.realtime.provider),
        ) if req.realtime else None,
        telephony=TenantTelephonyConfig(
            provider=tel.provider,
            from_number=tel.from_number,
            webhook_base_url=tel.webhook_base_url,
            account_sid_env=sid_env,
            auth_token_env=token_env,
        ),
    )

    session.add(Tenant(
        id=tenant_id, slug=slug, name=req.name, status="active",
        timezone=req.timezone, default_language=req.default_language,
        mode=req.mode, max_concurrent_calls=req.max_concurrent_calls,
        pipeline_config=pipeline.model_dump(),
    ))
    for ph in tel.phone_numbers:
        session.add(TenantPhoneNumber(
            phone_number=ph, tenant_id=tenant_id, provider=tel.provider))
    for name, value in secret_rows:
        session.add(TenantSecret(
            tenant_id=tenant_id, name=name, value_encrypted=crypto.encrypt(value)))

    api_token = f"vox_{pysecrets.token_urlsafe(32)}"
    session.add(TenantApiKey(
        token_hash=hash_api_token(api_token), tenant_id=tenant_id, label="register"))

    await session.commit()

    # Refresh the live resolver so the new tenant resolves immediately.
    resolver = getattr(request.app.state, "tenant_resolver", None)
    if resolver is not None and hasattr(resolver, "refresh"):
        await resolver.refresh(tenant_id)
        if hasattr(request.app.state, "tenants"):
            request.app.state.tenants = resolver.loaded_settings()

    log.info("registered tenant", extra={"tenant_id": tenant_id, "slug": slug})
    return RegisterTenantResponse(tenant_id=tenant_id, slug=slug, api_token=api_token)
