"""Campaign, Lead tables (PRD §6.1)."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import JSON, DateTime, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.models.database import Base


class Campaign(Base):
    __tablename__ = "campaigns"

    id: Mapped[str] = mapped_column(String(50), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(
        String(50), ForeignKey("tenants.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    status: Mapped[str] = mapped_column(String(20), default="draft")
    config_yaml: Mapped[str] = mapped_column(Text, nullable=False)
    total_leads: Mapped[int] = mapped_column(Integer, default=0)
    calls_attempted: Mapped[int] = mapped_column(Integer, default=0)
    calls_answered: Mapped[int] = mapped_column(Integer, default=0)
    leads_qualified: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False), server_default=func.now(), onupdate=func.now()
    )

    leads: Mapped[list["Lead"]] = relationship(
        back_populates="campaign", cascade="all, delete-orphan"
    )


class Lead(Base):
    __tablename__ = "leads"

    id: Mapped[str] = mapped_column(String(50), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(
        String(50), ForeignKey("tenants.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )
    campaign_id: Mapped[Optional[str]] = mapped_column(
        String(50), ForeignKey("campaigns.id"), index=True
    )
    phone_number: Mapped[str] = mapped_column(String(20), nullable=False)
    name: Mapped[Optional[str]] = mapped_column(String(255))
    language_pref: Mapped[Optional[str]] = mapped_column(String(10))
    crm_lead_id: Mapped[Optional[str]] = mapped_column(String(100))
    extra_data: Mapped[dict] = mapped_column("metadata", JSON, default=dict)
    status: Mapped[str] = mapped_column(String(20), default="pending", index=True)
    retry_count: Mapped[int] = mapped_column(Integer, default=0)
    next_retry_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=False))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False), server_default=func.now()
    )

    campaign: Mapped[Optional[Campaign]] = relationship(back_populates="leads")
