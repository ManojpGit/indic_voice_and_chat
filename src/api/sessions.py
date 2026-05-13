"""Session management endpoints (PRD §7.2). Phase 3+."""

from fastapi import APIRouter

router = APIRouter(prefix="/sessions", tags=["sessions"])
