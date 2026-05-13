"""initial schema with multi-tenant scoping

Revision ID: 0001_initial
Revises:
Create Date: 2026-05-12
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0001_initial"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # --- Tenants come first; everything else FKs to them. -----------------

    op.create_table(
        "tenants",
        sa.Column("id", sa.String(50), primary_key=True),
        sa.Column("slug", sa.String(63), nullable=False, unique=True),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("status", sa.String(20), server_default="active"),
        sa.Column("created_at", sa.DateTime, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime, server_default=sa.func.now()),
    )

    op.create_table(
        "tenant_phone_numbers",
        sa.Column("phone_number", sa.String(32), primary_key=True),
        sa.Column("tenant_id", sa.String(50),
                  sa.ForeignKey("tenants.id", ondelete="CASCADE"),
                  nullable=False),
        sa.Column("provider", sa.String(20), server_default="twilio"),
        sa.Column("label", sa.String(255)),
        sa.Column("created_at", sa.DateTime, server_default=sa.func.now()),
    )
    op.create_index("idx_tenant_phones_tenant", "tenant_phone_numbers", ["tenant_id"])

    op.create_table(
        "tenant_api_keys",
        sa.Column("token_hash", sa.String(64), primary_key=True),
        sa.Column("tenant_id", sa.String(50),
                  sa.ForeignKey("tenants.id", ondelete="CASCADE"),
                  nullable=False),
        sa.Column("label", sa.String(100), nullable=False),
        sa.Column("last_used_at", sa.DateTime),
        sa.Column("created_at", sa.DateTime, server_default=sa.func.now()),
        sa.UniqueConstraint("tenant_id", "label", name="uq_tenant_api_key_label"),
    )
    op.create_index("idx_tenant_api_keys_tenant", "tenant_api_keys", ["tenant_id"])

    # --- Campaign + lead tables, each scoped by tenant_id. ----------------

    op.create_table(
        "campaigns",
        sa.Column("id", sa.String(50), primary_key=True),
        sa.Column("tenant_id", sa.String(50),
                  sa.ForeignKey("tenants.id", ondelete="CASCADE"),
                  nullable=False),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("status", sa.String(20), server_default="draft"),
        sa.Column("config_yaml", sa.Text, nullable=False),
        sa.Column("total_leads", sa.Integer, server_default="0"),
        sa.Column("calls_attempted", sa.Integer, server_default="0"),
        sa.Column("calls_answered", sa.Integer, server_default="0"),
        sa.Column("leads_qualified", sa.Integer, server_default="0"),
        sa.Column("created_at", sa.DateTime, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime, server_default=sa.func.now()),
    )
    op.create_index("idx_campaigns_tenant", "campaigns", ["tenant_id"])

    op.create_table(
        "leads",
        sa.Column("id", sa.String(50), primary_key=True),
        sa.Column("tenant_id", sa.String(50),
                  sa.ForeignKey("tenants.id", ondelete="CASCADE"),
                  nullable=False),
        sa.Column("campaign_id", sa.String(50), sa.ForeignKey("campaigns.id")),
        sa.Column("phone_number", sa.String(20), nullable=False),
        sa.Column("name", sa.String(255)),
        sa.Column("language_pref", sa.String(10)),
        sa.Column("crm_lead_id", sa.String(100)),
        sa.Column("metadata", sa.JSON, server_default="{}"),
        sa.Column("status", sa.String(20), server_default="pending"),
        sa.Column("retry_count", sa.Integer, server_default="0"),
        sa.Column("next_retry_at", sa.DateTime),
        sa.Column("created_at", sa.DateTime, server_default=sa.func.now()),
    )
    op.create_index("idx_leads_tenant", "leads", ["tenant_id"])
    op.create_index("idx_leads_campaign", "leads", ["campaign_id", "status"])

    op.create_table(
        "conversations",
        sa.Column("id", sa.String(50), primary_key=True),
        sa.Column("tenant_id", sa.String(50),
                  sa.ForeignKey("tenants.id", ondelete="CASCADE"),
                  nullable=False),
        sa.Column("campaign_id", sa.String(50), sa.ForeignKey("campaigns.id")),
        sa.Column("lead_id", sa.String(50), sa.ForeignKey("leads.id")),
        sa.Column("agent_type", sa.String(20), nullable=False),
        sa.Column("channel", sa.String(20), nullable=False),
        sa.Column("status", sa.String(20), nullable=False),
        sa.Column("disposition", sa.String(30)),
        sa.Column("interest_level", sa.String(20)),
        sa.Column("slots_data", sa.JSON, server_default="{}"),
        sa.Column("pipeline_config", sa.JSON, nullable=False),
        sa.Column("duration_ms", sa.Integer),
        sa.Column("total_turns", sa.Integer, server_default="0"),
        sa.Column("started_at", sa.DateTime, server_default=sa.func.now()),
        sa.Column("ended_at", sa.DateTime),
    )
    op.create_index("idx_conversations_tenant", "conversations", ["tenant_id"])
    op.create_index("idx_conversations_campaign", "conversations", ["campaign_id"])

    op.create_table(
        "turns",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("conversation_id", sa.String(50),
                  sa.ForeignKey("conversations.id"), nullable=False),
        sa.Column("turn_number", sa.Integer, nullable=False),
        sa.Column("role", sa.String(10), nullable=False),
        sa.Column("content", sa.Text, nullable=False),
        sa.Column("language", sa.String(10)),
        sa.Column("stt_confidence", sa.Float),
        sa.Column("stt_latency_ms", sa.Integer),
        sa.Column("llm_ttft_ms", sa.Integer),
        sa.Column("llm_total_ms", sa.Integer),
        sa.Column("tts_first_chunk_ms", sa.Integer),
        sa.Column("tts_total_ms", sa.Integer),
        sa.Column("total_latency_ms", sa.Integer),
        sa.Column("metadata", sa.JSON, server_default="{}"),
        sa.Column("created_at", sa.DateTime, server_default=sa.func.now()),
    )
    op.create_index("idx_turns_conversation", "turns", ["conversation_id"])

    op.create_table(
        "events",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("conversation_id", sa.String(50), sa.ForeignKey("conversations.id")),
        sa.Column("event_type", sa.String(50), nullable=False),
        sa.Column("payload", sa.JSON, nullable=False),
        sa.Column("created_at", sa.DateTime, server_default=sa.func.now()),
    )
    op.create_index("idx_events_conversation", "events", ["conversation_id"])
    op.create_index("idx_events_type", "events", ["event_type"])

    # --- Benchmarks: platform-admin only, no tenant_id scoping. -----------

    op.create_table(
        "benchmark_runs",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("name", sa.String(255)),
        sa.Column("description", sa.Text),
        sa.Column("pipeline_config", sa.JSON, nullable=False),
        sa.Column("language", sa.String(10), nullable=False),
        sa.Column("dataset", sa.String(100), nullable=False),
        sa.Column("results", sa.JSON, nullable=False),
        sa.Column("created_at", sa.DateTime, server_default=sa.func.now()),
    )

    op.create_table(
        "kb_documents",
        sa.Column("id", sa.String(50), primary_key=True),
        sa.Column("tenant_id", sa.String(50),
                  sa.ForeignKey("tenants.id", ondelete="CASCADE"),
                  nullable=False),
        sa.Column("filename", sa.String(255), nullable=False),
        sa.Column("source_type", sa.String(50)),
        sa.Column("language", sa.String(10)),
        sa.Column("chunk_count", sa.Integer, server_default="0"),
        sa.Column("metadata", sa.JSON, server_default="{}"),
        sa.Column("ingested_at", sa.DateTime, server_default=sa.func.now()),
    )
    op.create_index("idx_kb_documents_tenant", "kb_documents", ["tenant_id"])


def downgrade() -> None:
    op.drop_index("idx_kb_documents_tenant", table_name="kb_documents")
    op.drop_table("kb_documents")
    op.drop_table("benchmark_runs")
    op.drop_index("idx_events_type", table_name="events")
    op.drop_index("idx_events_conversation", table_name="events")
    op.drop_table("events")
    op.drop_index("idx_turns_conversation", table_name="turns")
    op.drop_table("turns")
    op.drop_index("idx_conversations_campaign", table_name="conversations")
    op.drop_index("idx_conversations_tenant", table_name="conversations")
    op.drop_table("conversations")
    op.drop_index("idx_leads_campaign", table_name="leads")
    op.drop_index("idx_leads_tenant", table_name="leads")
    op.drop_table("leads")
    op.drop_index("idx_campaigns_tenant", table_name="campaigns")
    op.drop_table("campaigns")
    op.drop_index("idx_tenant_api_keys_tenant", table_name="tenant_api_keys")
    op.drop_table("tenant_api_keys")
    op.drop_index("idx_tenant_phones_tenant", table_name="tenant_phone_numbers")
    op.drop_table("tenant_phone_numbers")
    op.drop_table("tenants")
