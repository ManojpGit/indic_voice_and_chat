"""Configuration endpoints (PRD §7.6). Phase 3+."""

from fastapi import APIRouter

router = APIRouter(prefix="/config", tags=["config"])
