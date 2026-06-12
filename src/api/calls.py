"""Call Lead (async) + call status.

- ``POST /api/v1/campaigns/{id}/calls`` — place one outbound call for a lead.
  Async: it returns a ``call_id`` immediately; the outcome lands later (the
  bridge writes it at teardown). Guards: the campaign must be ``active`` and the
  tenant must be under its ``max_concurrent_calls`` cap (else 429). On success a
  ``conversations`` row is inserted (``in_progress``, keyed by the provider Call
  SID) recording the config used — mode, stt/llm/tts/realtime providers, voice,
  telephony provider — for statistics + cost.
- ``GET  /api/v1/calls/{call_id}`` — poll the call's status/outcome/cost.
"""

from __future__ import annotations

import logging
import uuid

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.call_store import count_active_calls
from src.api.deps import get_db_session
from src.auth import TenantContext, current_tenant
from src.interfaces.telephony import CallConfig
from src.models.campaign import Campaign as DbCampaign
from src.models.conversation import Conversation
from src.providers import get_telephony_provider

log = logging.getLogger(__name__)
router = APIRouter(tags=["calls"])


# --- Schemas ------------------------------------------------------------


class CallLeadRequest(BaseModel):
    to_number: str = Field(min_length=1)
    from_number: str | None = None
    voice: str | None = None
    lead_id: str | None = None


class CallLeadResponse(BaseModel):
    call_id: str
    status: str
    provider_call_sid: str


class CallStatusResponse(BaseModel):
    call_id: str
    status: str
    outcome: str | None = None
    summary: str | None = None
    notes: str | None = None
    callback_at: str | None = None
    cost: float | None = None
    duration_ms: int | None = None


# --- Routes -------------------------------------------------------------


@router.post("/campaigns/{campaign_id}/calls", response_model=CallLeadResponse, status_code=202)
async def call_lead(
    campaign_id: str,
    req: CallLeadRequest,
    session: AsyncSession = Depends(get_db_session),
    tenant: TenantContext = Depends(current_tenant),
) -> CallLeadResponse:
    campaign = await session.get(DbCampaign, campaign_id)
    if campaign is None or campaign.tenant_id != tenant.id:
        raise HTTPException(status_code=404, detail="campaign not found")
    if campaign.status != "active":
        raise HTTPException(
            status_code=409, detail=f"campaign is {campaign.status!r}, not active")

    # Enforce the per-tenant concurrency cap.
    cap = tenant.settings.max_concurrent_calls
    if await count_active_calls(session, tenant.id) >= cap:
        raise HTTPException(
            status_code=429,
            detail=f"max concurrent calls reached ({cap}); retry when a call ends")

    pipeline = tenant.settings.pipeline
    tel = pipeline.telephony
    provider = (tel.provider or "").lower()
    from_number = req.from_number or (tel.outbound_from or {}).get(provider) or tel.from_number
    if not from_number:
        raise HTTPException(status_code=400, detail="no caller-ID configured for this tenant")
    if not tel.webhook_base_url:
        raise HTTPException(status_code=400, detail="tenant telephony.webhook_base_url must be set")

    try:
        adapter = get_telephony_provider({"provider": provider})
    except Exception as e:  # noqa: BLE001 — e.g. missing credentials
        raise HTTPException(status_code=400, detail=f"telephony adapter unavailable: {e}")

    voice = req.voice or pipeline.tts.voice_id or (pipeline.realtime.voice if pipeline.realtime else None)
    cfg = CallConfig(
        to_number=req.to_number.strip(),
        from_number=from_number,
        webhook_url=tel.webhook_base_url.rstrip("/"),
    )
    try:
        call_session = await adapter.initiate_call(cfg)
    except Exception as e:  # noqa: BLE001
        log.exception("call lead failed", extra={"tenant": tenant.slug, "provider": provider})
        raise HTTPException(status_code=502, detail=f"call failed: {e}")

    call_id = f"call_{uuid.uuid4().hex[:16]}"
    realtime_provider = pipeline.realtime.provider if (pipeline.mode == "s2s" and pipeline.realtime) else None
    session.add(Conversation(
        id=call_id, tenant_id=tenant.id, campaign_id=campaign_id, lead_id=req.lead_id,
        agent_type="voicebot", channel="voice", status="in_progress",
        pipeline_config=pipeline.model_dump(), provider_call_sid=call_session.session_id,
        mode=pipeline.mode,
        stt_provider=pipeline.stt.provider, llm_provider=pipeline.llm.provider,
        tts_provider=pipeline.tts.provider, realtime_provider=realtime_provider,
        voice=voice, telephony_provider=provider,
    ))
    await session.commit()

    log.info("call lead placed", extra={
        "tenant": tenant.slug, "campaign": campaign_id, "call_id": call_id,
        "sid": call_session.session_id})
    return CallLeadResponse(
        call_id=call_id, status="in_progress", provider_call_sid=call_session.session_id)


@router.get("/calls/{call_id}", response_model=CallStatusResponse)
async def get_call(
    call_id: str,
    session: AsyncSession = Depends(get_db_session),
    tenant: TenantContext = Depends(current_tenant),
) -> CallStatusResponse:
    row = await session.get(Conversation, call_id)
    if row is None or row.tenant_id != tenant.id:
        raise HTTPException(status_code=404, detail="call not found")
    return CallStatusResponse(
        call_id=row.id, status=row.status, outcome=row.outcome,
        summary=row.summary, notes=row.notes,
        callback_at=row.callback_at.isoformat() if row.callback_at else None,
        cost=row.cost, duration_ms=row.duration_ms,
    )
