"""Conversations & analytics endpoints (PRD §7.7). Phase 3+."""

from fastapi import APIRouter

router = APIRouter(prefix="/conversations", tags=["conversations"])
