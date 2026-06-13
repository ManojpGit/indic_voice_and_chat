"""Provider cost catalog + voice list endpoints.

- ``GET  /api/v1/providers``                  — list every provider + cost/min
- ``PUT  /api/v1/providers/{kind}/{provider}`` — admin: maintain a rate
- ``GET  /api/v1/voices?provider=&language=``  — static voice roster (tenant-authed)

The cost catalog is the single source of truth read by ``GET /providers`` and the
per-call cost calculation; ``PUT`` upserts so rates can be kept current as vendor
pricing changes.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query, Request
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.deps import get_db_session
from src.auth import TenantContext
from src.auth.middleware import optional_tenant, require_admin
from src.models.tenant import ProviderCost
from src.providers.model_catalog import list_models
from src.providers.voice_catalog import list_voices

router = APIRouter(tags=["catalog"])


async def tenant_or_admin(
    request: Request, tenant: TenantContext | None = Depends(optional_tenant)
) -> None:
    """Allow a valid tenant **or** admin bearer (e.g. the cost catalog, which
    both the tenant page and the admin page list)."""
    if tenant is not None:
        return
    await require_admin(request)  # raises 401/403 if not a valid admin token


# --- Schemas ------------------------------------------------------------


class ProviderCostItem(BaseModel):
    kind: str
    provider: str
    cost_per_min: float


class ProvidersResponse(BaseModel):
    providers: list[ProviderCostItem]


class UpdateProviderCostRequest(BaseModel):
    cost_per_min: float = Field(ge=0)


class VoiceItem(BaseModel):
    voice_id: str
    gender: str | None = None


class VoicesResponse(BaseModel):
    provider: str
    language: str
    voices: list[VoiceItem]


# --- Routes -------------------------------------------------------------


@router.get("/providers", response_model=ProvidersResponse)
async def list_providers(
    session: AsyncSession = Depends(get_db_session),
    _: None = Depends(tenant_or_admin),
) -> ProvidersResponse:
    """List every provider and its current cost/min, ordered by kind then name."""
    rows = (await session.execute(
        select(ProviderCost).order_by(ProviderCost.kind, ProviderCost.provider)
    )).scalars().all()
    return ProvidersResponse(providers=[
        ProviderCostItem(kind=r.kind, provider=r.provider, cost_per_min=r.cost_per_min)
        for r in rows
    ])


@router.put("/providers/{kind}/{provider}", response_model=ProviderCostItem)
async def update_provider_cost(
    kind: str,
    provider: str,
    req: UpdateProviderCostRequest,
    session: AsyncSession = Depends(get_db_session),
    _: None = Depends(require_admin),
) -> ProviderCostItem:
    """Upsert a provider's cost/min. Admin-only. New rate is read live."""
    row = await session.get(ProviderCost, (kind, provider))
    if row is None:
        row = ProviderCost(kind=kind, provider=provider, cost_per_min=req.cost_per_min)
        session.add(row)
    else:
        row.cost_per_min = req.cost_per_min
    await session.commit()
    return ProviderCostItem(kind=kind, provider=provider, cost_per_min=req.cost_per_min)


class ModelsResponse(BaseModel):
    # kind -> provider -> [model ids]; first per list is the recommended default.
    models: dict[str, dict[str, list[str]]]


@router.get("/models", response_model=ModelsResponse)
async def get_models() -> ModelsResponse:
    """Selectable provider + model variants per kind (stt/llm/tts/s2s).

    Public reference data (model ids only) — drives the Register UI's
    provider/model dropdowns, which must populate before any token is entered.
    """
    return ModelsResponse(models=list_models())


@router.get("/voices", response_model=VoicesResponse)
async def get_voices(
    provider: str = Query(..., description="sarvam | gemini_live"),
    language: str = Query("hi-IN", description="BCP-47 language tag (TTS only)"),
) -> VoicesResponse:
    """Return the available voices for a provider (+ language for TTS)."""
    voices = list_voices(provider, language)
    return VoicesResponse(
        provider=provider,
        language=language,
        voices=[VoiceItem(**v) for v in voices],
    )
