"""Tenant ORM tables.

A tenant is the unit of multi-tenancy: each merchant gets one ``Tenant``
row, plus N ``TenantPhoneNumber`` rows (Twilio numbers they own) and M
``TenantApiKey`` rows (bearer tokens issued for their API access).

API keys are stored hashed — only the hash hits the DB. The plaintext
token is shown once at creation time and never persisted.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import DateTime, ForeignKey, String, Text, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.models.database import Base


class Tenant(Base):
    __tablename__ = "tenants"

    id: Mapped[str] = mapped_column(String(50), primary_key=True)
    slug: Mapped[str] = mapped_column(String(63), nullable=False, unique=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    status: Mapped[str] = mapped_column(String(20), default="active")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False), server_default=func.now(), onupdate=func.now()
    )

    phone_numbers: Mapped[list["TenantPhoneNumber"]] = relationship(
        back_populates="tenant", cascade="all, delete-orphan"
    )
    api_keys: Mapped[list["TenantApiKey"]] = relationship(
        back_populates="tenant", cascade="all, delete-orphan"
    )


class TenantPhoneNumber(Base):
    """Maps an inbound phone number (Twilio number) to its owning tenant.

    Used by the Twilio voice webhook to look up *which* tenant the call
    belongs to before any further routing. Phone numbers are stored in the
    normalized form (``+E.164``); see ``src.campaign.dnd_filter.normalize_phone``.
    """

    __tablename__ = "tenant_phone_numbers"

    phone_number: Mapped[str] = mapped_column(String(32), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(
        String(50), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True
    )
    provider: Mapped[str] = mapped_column(String(20), default="twilio")
    label: Mapped[Optional[str]] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False), server_default=func.now()
    )

    tenant: Mapped[Tenant] = relationship(back_populates="phone_numbers")


class TenantApiKey(Base):
    """SHA-256 hash of a bearer token. The plaintext is never stored."""

    __tablename__ = "tenant_api_keys"
    __table_args__ = (
        UniqueConstraint("tenant_id", "label", name="uq_tenant_api_key_label"),
    )

    token_hash: Mapped[str] = mapped_column(String(64), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(
        String(50), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True
    )
    label: Mapped[str] = mapped_column(String(100), nullable=False)
    last_used_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=False))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False), server_default=func.now()
    )

    tenant: Mapped[Tenant] = relationship(back_populates="api_keys")
