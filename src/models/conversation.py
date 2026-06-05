"""Conversation, Turn, Event tables (PRD §6.1)."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.models.database import Base


class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[str] = mapped_column(String(50), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(
        String(50), ForeignKey("tenants.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )
    campaign_id: Mapped[Optional[str]] = mapped_column(
        String(50), ForeignKey("campaigns.id"), nullable=True
    )
    lead_id: Mapped[Optional[str]] = mapped_column(
        String(50), ForeignKey("leads.id"), nullable=True
    )
    agent_type: Mapped[str] = mapped_column(String(20), nullable=False)
    channel: Mapped[str] = mapped_column(String(20), nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False)
    disposition: Mapped[Optional[str]] = mapped_column(String(30))
    outcome: Mapped[Optional[str]] = mapped_column(String(30))
    summary: Mapped[Optional[str]] = mapped_column(Text)
    notes: Mapped[Optional[str]] = mapped_column(Text)
    callback_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=False))
    interest_level: Mapped[Optional[str]] = mapped_column(String(20))
    slots_data: Mapped[dict] = mapped_column(JSON, default=dict)
    pipeline_config: Mapped[dict] = mapped_column(JSON, nullable=False)
    duration_ms: Mapped[Optional[int]] = mapped_column(Integer)
    total_turns: Mapped[int] = mapped_column(Integer, default=0)
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False), server_default=func.now()
    )
    ended_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=False))

    turns: Mapped[list["Turn"]] = relationship(
        back_populates="conversation", cascade="all, delete-orphan"
    )
    events: Mapped[list["Event"]] = relationship(
        back_populates="conversation", cascade="all, delete-orphan"
    )


class Turn(Base):
    __tablename__ = "turns"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    conversation_id: Mapped[str] = mapped_column(
        String(50), ForeignKey("conversations.id"), nullable=False, index=True
    )
    turn_number: Mapped[int] = mapped_column(Integer, nullable=False)
    role: Mapped[str] = mapped_column(String(10), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    language: Mapped[Optional[str]] = mapped_column(String(10))
    stt_confidence: Mapped[Optional[float]] = mapped_column(Float)
    stt_latency_ms: Mapped[Optional[int]] = mapped_column(Integer)
    llm_ttft_ms: Mapped[Optional[int]] = mapped_column(Integer)
    llm_total_ms: Mapped[Optional[int]] = mapped_column(Integer)
    tts_first_chunk_ms: Mapped[Optional[int]] = mapped_column(Integer)
    tts_total_ms: Mapped[Optional[int]] = mapped_column(Integer)
    total_latency_ms: Mapped[Optional[int]] = mapped_column(Integer)
    extra_data: Mapped[dict] = mapped_column("metadata", JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False), server_default=func.now()
    )

    conversation: Mapped[Conversation] = relationship(back_populates="turns")


class Event(Base):
    __tablename__ = "events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    conversation_id: Mapped[Optional[str]] = mapped_column(
        String(50), ForeignKey("conversations.id"), index=True
    )
    event_type: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    payload: Mapped[dict] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False), server_default=func.now()
    )

    conversation: Mapped[Optional[Conversation]] = relationship(back_populates="events")
