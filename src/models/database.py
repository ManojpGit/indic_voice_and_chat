"""SQLAlchemy 2.x async engine + session factory.

Engines are created lazily so tests can supply their own URL (e.g. an
in-memory SQLite) without booting postgres.
"""

from __future__ import annotations

from typing import Optional

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""


_engine: Optional[AsyncEngine] = None
_sessionmaker: Optional[async_sessionmaker[AsyncSession]] = None


def get_engine(url: Optional[str] = None) -> AsyncEngine:
    """Return the process-wide async engine, creating it on first call."""
    global _engine, _sessionmaker
    if _engine is None:
        if url is None:
            from src.config import get_settings

            url = get_settings().database.url
        _engine = create_async_engine(url, future=True, pool_pre_ping=True)
        _sessionmaker = async_sessionmaker(_engine, expire_on_commit=False)
    return _engine


def get_sessionmaker() -> async_sessionmaker[AsyncSession]:
    if _sessionmaker is None:
        get_engine()
    assert _sessionmaker is not None  # for type checker
    return _sessionmaker


async def dispose_engine() -> None:
    """Close the engine; used on FastAPI shutdown."""
    global _engine, _sessionmaker
    if _engine is not None:
        await _engine.dispose()
        _engine = None
        _sessionmaker = None


def reset_engine_for_tests(url: str) -> AsyncEngine:
    """Reinitialize the engine against a fresh URL (test fixture helper)."""
    global _engine, _sessionmaker
    _engine = create_async_engine(url, future=True)
    _sessionmaker = async_sessionmaker(_engine, expire_on_commit=False)
    return _engine
