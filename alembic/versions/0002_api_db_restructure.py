"""API/DB restructure: tenant config + secrets + provider costs + call stats.

Adds the columns/tables the purely-API-based, DB-backed system needs:
- tenants: timezone, default_language, mode, max_concurrent_calls, pipeline_config
- tenant_secrets (NEW): per-tenant telephony keys, encrypted at rest
- provider_costs (NEW): cost/min catalog per provider
- conversations: the outcome columns that drifted out of 0001
  (outcome/summary/notes/callback_at) + the per-call config used + cost
  + provider_call_sid (bridge correlation key)

Revision ID: 0002_api_db_restructure
Revises: 0001_initial
"""

from __future__ import annotations

from typing import Union

import sqlalchemy as sa
from alembic import op

revision: str = "0002_api_db_restructure"
down_revision: Union[str, None] = "0001_initial"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # --- tenants: config columns ---
    op.add_column("tenants", sa.Column("timezone", sa.String(64), server_default="Asia/Kolkata"))
    op.add_column("tenants", sa.Column("default_language", sa.String(10), server_default="hi"))
    op.add_column("tenants", sa.Column("mode", sa.String(20), server_default="layered"))
    op.add_column("tenants", sa.Column("max_concurrent_calls", sa.Integer, server_default="1"))
    op.add_column("tenants", sa.Column("pipeline_config", sa.JSON, nullable=True))

    # --- tenant_secrets: per-tenant telephony keys (encrypted) ---
    op.create_table(
        "tenant_secrets",
        sa.Column("tenant_id", sa.String(50),
                  sa.ForeignKey("tenants.id", ondelete="CASCADE"), primary_key=True),
        sa.Column("name", sa.String(64), primary_key=True),
        sa.Column("value_encrypted", sa.Text, nullable=False),
        sa.Column("created_at", sa.DateTime, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime, server_default=sa.func.now()),
    )

    # --- provider_costs: cost/min catalog ---
    op.create_table(
        "provider_costs",
        sa.Column("kind", sa.String(20), primary_key=True),
        sa.Column("provider", sa.String(40), primary_key=True),
        sa.Column("cost_per_min", sa.Float, nullable=False, server_default="0"),
        sa.Column("updated_at", sa.DateTime, server_default=sa.func.now()),
    )

    # --- conversations: drifted outcome columns + per-call stats + cost ---
    op.add_column("conversations", sa.Column("outcome", sa.String(30)))
    op.add_column("conversations", sa.Column("summary", sa.Text))
    op.add_column("conversations", sa.Column("notes", sa.Text))
    op.add_column("conversations", sa.Column("callback_at", sa.DateTime))
    op.add_column("conversations", sa.Column("provider_call_sid", sa.String(128)))
    op.add_column("conversations", sa.Column("mode", sa.String(20)))
    op.add_column("conversations", sa.Column("stt_provider", sa.String(30)))
    op.add_column("conversations", sa.Column("llm_provider", sa.String(30)))
    op.add_column("conversations", sa.Column("tts_provider", sa.String(30)))
    op.add_column("conversations", sa.Column("realtime_provider", sa.String(30)))
    op.add_column("conversations", sa.Column("voice", sa.String(50)))
    op.add_column("conversations", sa.Column("telephony_provider", sa.String(30)))
    op.add_column("conversations", sa.Column("cost", sa.Float))
    op.create_index("idx_conversations_provider_call_sid", "conversations", ["provider_call_sid"])


def downgrade() -> None:
    op.drop_index("idx_conversations_provider_call_sid", table_name="conversations")
    for col in ("cost", "telephony_provider", "voice", "realtime_provider", "tts_provider",
                "llm_provider", "stt_provider", "mode", "provider_call_sid", "callback_at",
                "notes", "summary", "outcome"):
        op.drop_column("conversations", col)
    op.drop_table("provider_costs")
    op.drop_table("tenant_secrets")
    for col in ("pipeline_config", "max_concurrent_calls", "mode", "default_language", "timezone"):
        op.drop_column("tenants", col)
