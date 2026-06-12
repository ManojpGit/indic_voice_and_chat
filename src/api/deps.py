"""Shared FastAPI dependencies for the DB-backed API routes."""

from __future__ import annotations

from typing import AsyncIterator

from sqlalchemy.ext.asyncio import AsyncSession

from src.models.database import get_sessionmaker


async def get_db_session() -> AsyncIterator[AsyncSession]:
    """Yield an async DB session for a request (one session per request)."""
    sessionmaker = get_sessionmaker()
    async with sessionmaker() as session:
        yield session
