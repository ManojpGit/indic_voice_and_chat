"""Campaign management endpoints (PRD §7.1).

In-memory campaign + lead registry for now. Phase 6+ swaps this for the
SQLAlchemy-backed repository against the Postgres tables already created
in `src/models/campaign.py`.

The orchestrator's ``run()`` is started as a background asyncio task on
``/start`` and tracked in a per-campaign ``CampaignRun`` so ``/stats`` can
report live progress and ``/pause`` can flip the campaign status.
"""

from __future__ import annotations

import asyncio
import io
import logging
import uuid
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from src.auth import TenantContext, current_tenant
from src.campaign.models import (
    CampaignStatus,
    Lead,
    LeadStatus,
    parse_leads_csv,
)
from src.campaign.models import Campaign as CampaignModel
from src.campaign.orchestrator import CampaignOrchestrator, CampaignRun

log = logging.getLogger(__name__)
router = APIRouter(prefix="/campaigns", tags=["campaigns"])


# --- DI -----------------------------------------------------------------


_orchestrator: Optional[CampaignOrchestrator] = None
_campaigns: dict[str, CampaignModel] = {}
_leads: dict[str, list[Lead]] = {}
_runs: dict[str, CampaignRun] = {}
_run_tasks: dict[str, asyncio.Task] = {}


def set_orchestrator(orchestrator: Optional[CampaignOrchestrator]) -> None:
    global _orchestrator, _campaigns, _leads, _runs, _run_tasks
    _orchestrator = orchestrator
    if orchestrator is None:
        _campaigns = {}
        _leads = {}
        _runs = {}
        _run_tasks = {}


def _require_orchestrator() -> CampaignOrchestrator:
    if _orchestrator is None:
        raise HTTPException(
            status_code=503,
            detail="campaign orchestrator not initialized",
        )
    return _orchestrator


# --- Schemas ------------------------------------------------------------


class CreateCampaignRequest(BaseModel):
    id: Optional[str] = None
    name: str = Field(min_length=1)
    config_yaml: str = ""


class UpdateCampaignRequest(BaseModel):
    name: Optional[str] = None
    config_yaml: Optional[str] = None
    status: Optional[CampaignStatus] = None


class CampaignResponse(BaseModel):
    id: str
    name: str
    status: CampaignStatus
    total_leads: int
    calls_attempted: int
    calls_answered: int
    leads_qualified: int
    created_at: datetime
    updated_at: datetime

    @classmethod
    def from_model(cls, c: CampaignModel) -> "CampaignResponse":
        return cls(
            id=c.id,
            name=c.name,
            status=c.status,
            total_leads=c.total_leads,
            calls_attempted=c.calls_attempted,
            calls_answered=c.calls_answered,
            leads_qualified=c.leads_qualified,
            created_at=c.created_at,
            updated_at=c.updated_at,
        )


class CampaignListResponse(BaseModel):
    campaigns: list[CampaignResponse]
    total: int


class LeadResponse(BaseModel):
    id: str
    phone_number: str
    name: Optional[str] = None
    status: LeadStatus
    retry_count: int
    next_retry_at: Optional[datetime] = None


class LeadsResponse(BaseModel):
    leads: list[LeadResponse]
    total: int


class LeadUploadResponse(BaseModel):
    campaign_id: str
    leads_added: int
    errors: list[dict]


class CampaignStatsResponse(BaseModel):
    id: str
    status: CampaignStatus
    total_leads: int
    calls_attempted: int
    calls_answered: int
    leads_qualified: int
    active: int
    completed: int


# --- Routes -------------------------------------------------------------


def _scoped(campaign_id: str, tenant: TenantContext) -> CampaignModel:
    """Fetch a campaign and 404 if it doesn't belong to ``tenant``."""
    campaign = _campaigns.get(campaign_id)
    if campaign is None or campaign.tenant_id != tenant.id:
        raise HTTPException(status_code=404, detail="campaign not found")
    return campaign


@router.post("", response_model=CampaignResponse)
async def create_campaign(
    req: CreateCampaignRequest, tenant: TenantContext = Depends(current_tenant)
) -> CampaignResponse:
    _require_orchestrator()
    campaign_id = req.id or f"camp_{uuid.uuid4().hex[:12]}"
    if campaign_id in _campaigns:
        raise HTTPException(status_code=409, detail=f"campaign {campaign_id!r} already exists")
    campaign = CampaignModel(
        id=campaign_id, tenant_id=tenant.id, name=req.name, config_yaml=req.config_yaml
    )
    _campaigns[campaign_id] = campaign
    _leads[campaign_id] = []
    return CampaignResponse.from_model(campaign)


@router.get("", response_model=CampaignListResponse)
async def list_campaigns(tenant: TenantContext = Depends(current_tenant)) -> CampaignListResponse:
    _require_orchestrator()
    items = [
        CampaignResponse.from_model(c)
        for c in _campaigns.values()
        if c.tenant_id == tenant.id
    ]
    return CampaignListResponse(campaigns=items, total=len(items))


@router.get("/{campaign_id}", response_model=CampaignResponse)
async def get_campaign(
    campaign_id: str, tenant: TenantContext = Depends(current_tenant)
) -> CampaignResponse:
    _require_orchestrator()
    campaign = _scoped(campaign_id, tenant)
    run = _runs.get(campaign_id)
    if run is not None:
        campaign = run.campaign
    return CampaignResponse.from_model(campaign)


@router.put("/{campaign_id}", response_model=CampaignResponse)
async def update_campaign(
    campaign_id: str, req: UpdateCampaignRequest,
    tenant: TenantContext = Depends(current_tenant),
) -> CampaignResponse:
    _require_orchestrator()
    campaign = _scoped(campaign_id, tenant)
    if req.name is not None:
        campaign.name = req.name
    if req.config_yaml is not None:
        campaign.config_yaml = req.config_yaml
    if req.status is not None:
        campaign.status = req.status
    campaign.updated_at = datetime.utcnow()
    return CampaignResponse.from_model(campaign)


@router.post("/{campaign_id}/leads", response_model=LeadUploadResponse)
async def upload_leads(
    campaign_id: str, file: UploadFile = File(...),
    tenant: TenantContext = Depends(current_tenant),
) -> LeadUploadResponse:
    _require_orchestrator()
    _scoped(campaign_id, tenant)
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="empty upload")
    try:
        leads, errors = parse_leads_csv(
            data, campaign_id=campaign_id, tenant_id=tenant.id
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    _leads[campaign_id].extend(leads)
    _campaigns[campaign_id].total_leads = len(_leads[campaign_id])
    return LeadUploadResponse(
        campaign_id=campaign_id,
        leads_added=len(leads),
        errors=[{"row": r, "reason": msg} for r, msg in errors],
    )


@router.get("/{campaign_id}/leads", response_model=LeadsResponse)
async def list_leads(
    campaign_id: str, tenant: TenantContext = Depends(current_tenant),
) -> LeadsResponse:
    _require_orchestrator()
    _scoped(campaign_id, tenant)
    items = [
        LeadResponse(
            id=lead.id,
            phone_number=lead.phone_number,
            name=lead.name,
            status=lead.status,
            retry_count=lead.retry_count,
            next_retry_at=lead.next_retry_at,
        )
        for lead in _leads.get(campaign_id, [])
    ]
    return LeadsResponse(leads=items, total=len(items))


@router.post("/{campaign_id}/start", response_model=CampaignResponse)
async def start_campaign(
    campaign_id: str, tenant: TenantContext = Depends(current_tenant),
) -> CampaignResponse:
    orch = _require_orchestrator()
    campaign = _scoped(campaign_id, tenant)
    if campaign_id in _run_tasks and not _run_tasks[campaign_id].done():
        raise HTTPException(status_code=409, detail="campaign already running")

    leads = _leads.get(campaign_id, [])
    campaign.status = CampaignStatus.ACTIVE

    async def runner() -> None:
        run = await orch.run(campaign, leads)
        _runs[campaign_id] = run

    task = asyncio.create_task(runner())
    _run_tasks[campaign_id] = task
    return CampaignResponse.from_model(campaign)


@router.post("/{campaign_id}/pause", response_model=CampaignResponse)
async def pause_campaign(
    campaign_id: str, tenant: TenantContext = Depends(current_tenant),
) -> CampaignResponse:
    _require_orchestrator()
    campaign = _scoped(campaign_id, tenant)
    campaign.status = CampaignStatus.PAUSED
    campaign.updated_at = datetime.utcnow()
    return CampaignResponse.from_model(campaign)


@router.post("/{campaign_id}/resume", response_model=CampaignResponse)
async def resume_campaign(
    campaign_id: str, tenant: TenantContext = Depends(current_tenant),
) -> CampaignResponse:
    _require_orchestrator()
    campaign = _scoped(campaign_id, tenant)
    campaign.status = CampaignStatus.ACTIVE
    campaign.updated_at = datetime.utcnow()
    return CampaignResponse.from_model(campaign)


@router.get("/{campaign_id}/stats", response_model=CampaignStatsResponse)
async def campaign_stats(
    campaign_id: str, tenant: TenantContext = Depends(current_tenant),
) -> CampaignStatsResponse:
    _require_orchestrator()
    campaign = _scoped(campaign_id, tenant)
    run = _runs.get(campaign_id)
    active = len(run.active) if run else 0
    completed = len(run.completed_leads) if run else 0
    return CampaignStatsResponse(
        id=campaign.id,
        status=campaign.status,
        total_leads=campaign.total_leads,
        calls_attempted=campaign.calls_attempted,
        calls_answered=campaign.calls_answered,
        leads_qualified=campaign.leads_qualified,
        active=active,
        completed=completed,
    )
