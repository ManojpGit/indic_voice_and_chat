"""FastAPI app entry point.

Lifespan-based startup:
- configure structured logging
- initialize SQLAlchemy async engine + Redis pool
- discover every tenant in ``config/tenants/`` and register it on the
  in-memory ``TenantResolver``
- build the ``TenantRuntimeRegistry`` so per-tenant providers, retrievers,
  DND stores, schedulers, webhook managers, etc. are lazily wired on first
  use of each tenant

``GET /health`` probes infrastructure + reports per-tenant provider routing.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import redis.asyncio as redis_async
from fastapi import FastAPI
from sqlalchemy import text

from src.api import api_router, telephony_hooks
from src.api.dev_console import (
    dev_console_enabled,
    dev_router,
    make_browser_bridge_factory,
    set_browser_bridge_factory,
)
from src.api.dev_console import (
    ws_router as dev_ws_router,
)
from src.auth.middleware import (
    InMemoryTenantResolver,
    set_admin_tokens,
    set_tenant_resolver,
)
from src.bootstrap import (
    build_provider_registry,
    make_bridge_factory,
    make_exotel_bridge_factory,
    make_stringee_bridge_factory,
)
from src.config import Settings, get_settings
from src.config_tenant import TenantSettings, discover_tenant_slugs, load_tenant
from src.dialogue.campaign_loader import active_campaign_slug, load_campaign
from src.dialogue.context import SessionStore
from src.models.database import dispose_engine, get_engine, get_sessionmaker
from src.utils.logging import configure_logging, get_logger

log = get_logger(__name__)


def _load_tenants(tenant_dir: Path) -> dict[str, TenantSettings]:
    """Discover every YAML file in ``tenant_dir`` and load it."""
    return {slug: load_tenant(slug, tenant_dir) for slug in discover_tenant_slugs(tenant_dir)}


def _admin_tokens_from_env() -> list[str]:
    """Comma-separated admin tokens in ``VOX_ADMIN_TOKENS``. Empty if unset."""
    raw = os.environ.get("VOX_ADMIN_TOKENS", "")
    return [t.strip() for t in raw.split(",") if t.strip()]


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings: Settings = get_settings()
    configure_logging(settings.app.log_level)
    log.info("startup", extra={"app": settings.app.name, "version": settings.app.version})

    # Eagerly create engine + redis pool so missing config fails on boot, not first request.
    get_engine(settings.database.url)
    redis_client = redis_async.from_url(settings.redis.url, decode_responses=False)
    app.state.redis = redis_client
    app.state.settings = settings

    # --- Tenant discovery + auth ---------------------------------------
    tenant_dir = Path(os.environ.get("VOX_TENANT_DIR", "config/tenants"))
    tenants = _load_tenants(tenant_dir)
    resolver = InMemoryTenantResolver()
    for slug, tsettings in tenants.items():
        # Tokens for tenant API access come from env via a per-tenant scheme:
        # ``TENANT_<UPPER_SLUG>_API_TOKENS`` (comma-separated).
        env_var = f"TENANT_{slug.upper()}_API_TOKENS"
        raw_tokens = os.environ.get(env_var, "")
        tokens = [t.strip() for t in raw_tokens.split(",") if t.strip()]
        resolver.register(tsettings, plaintext_tokens=tokens)
        log.info("tenant registered", extra={"slug": slug, "tokens_count": len(tokens)})
    set_tenant_resolver(resolver)
    set_admin_tokens(_admin_tokens_from_env())
    app.state.tenants = tenants

    # --- Bridge factory: turn an inbound Twilio WS into a live agent ----
    providers = build_provider_registry(
        global_defaults={
            "stt": settings.pipeline.stt.model_dump(),
            "llm": settings.pipeline.llm.model_dump(),
            "tts": settings.pipeline.tts.model_dump(),
            "telephony": settings.pipeline.telephony.model_dump(),
            "vector_store": settings.pipeline.vector_store.model_dump(),
        },
    )
    base_session_store = SessionStore(
        redis=redis_client, ttl_seconds=settings.redis.session_ttl_seconds
    )
    campaign = load_campaign(active_campaign_slug())
    log.info(
        "campaign loaded",
        extra={"slug": active_campaign_slug(), "agent": campaign.script.agent_name,
               "slots": list(campaign.slots.specs.keys())},
    )
    telephony_hooks.set_bridge_factory(
        make_bridge_factory(
            providers=providers, session_store=base_session_store,
            script=campaign.script, slots=campaign.slots,
        )
    )
    telephony_hooks.set_exotel_bridge_factory(
        make_exotel_bridge_factory(
            providers=providers, session_store=base_session_store,
            script=campaign.script, slots=campaign.slots,
        )
    )
    telephony_hooks.set_stringee_bridge_factory(
        make_stringee_bridge_factory(
            providers=providers, script=campaign.script, slots=campaign.slots,
        )
    )
    if dev_console_enabled():
        set_browser_bridge_factory(
            make_browser_bridge_factory(
                providers=providers, script=campaign.script, slots=campaign.slots,
            )
        )
        log.info("dev console enabled at /dev/voice")
    app.state.providers = providers

    try:
        yield
    finally:
        log.info("shutdown")
        telephony_hooks.set_bridge_factory(None)
        telephony_hooks.set_exotel_bridge_factory(None)
        telephony_hooks.set_stringee_bridge_factory(None)
        set_browser_bridge_factory(None)
        await redis_client.aclose()
        await dispose_engine()
        set_tenant_resolver(None)


app = FastAPI(
    title="vox-agent",
    version="1.0.0",
    description="Vendor-agnostic agentic framework for multilingual VoiceBot + ChatBot",
    lifespan=lifespan,
)

if dev_console_enabled():
    api_router.include_router(dev_ws_router)    # WS  /api/v1/dev/voice

app.include_router(api_router)

if dev_console_enabled():
    app.include_router(dev_router)              # GET /dev/voice


@app.get("/health")
async def health() -> dict:
    """Liveness + dependency probe + per-tenant provider routing."""
    settings: Settings = app.state.settings if hasattr(app.state, "settings") else get_settings()

    redis_status = "down"
    try:
        if hasattr(app.state, "redis"):
            await app.state.redis.ping()
            redis_status = "ok"
    except Exception as e:  # noqa: BLE001
        log.warning("redis ping failed", extra={"error": str(e)})

    db_status = "down"
    try:
        sessionmaker = get_sessionmaker()
        async with sessionmaker() as session:
            await session.execute(text("SELECT 1"))
        db_status = "ok"
    except Exception as e:  # noqa: BLE001
        log.warning("db probe failed", extra={"error": str(e)})

    tenants_summary = []
    tenants: dict[str, TenantSettings] = getattr(app.state, "tenants", {})
    for slug, t in tenants.items():
        tenants_summary.append({
            "slug": slug,
            "name": t.name,
            "status": t.status,
            "providers": {
                "stt": t.pipeline.stt.provider or settings.pipeline.stt.provider,
                "llm": t.pipeline.llm.provider or settings.pipeline.llm.provider,
                "tts": t.pipeline.tts.provider or settings.pipeline.tts.provider,
                "telephony": t.pipeline.telephony.provider or settings.pipeline.telephony.provider,
                "vector_store": t.pipeline.vector_store.provider or settings.pipeline.vector_store.provider,
            },
        })

    overall = "ok" if redis_status == "ok" and db_status == "ok" else "degraded"
    return {
        "status": overall,
        "version": settings.app.version,
        "platform_defaults": {
            "stt": settings.pipeline.stt.provider,
            "llm": settings.pipeline.llm.provider,
            "tts": settings.pipeline.tts.provider,
            "telephony": settings.pipeline.telephony.provider,
            "vector_store": settings.pipeline.vector_store.provider,
        },
        "tenants": tenants_summary,
        "tenant_count": len(tenants_summary),
        "redis": redis_status,
        "db": db_status,
    }
