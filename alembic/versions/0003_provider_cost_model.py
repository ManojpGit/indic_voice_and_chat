"""Make the provider cost catalog model-level.

provider_costs gains a ``model`` column and its primary key becomes
``(kind, provider, model)`` so STT/LLM/TTS/S2S can be priced per model variant.
Existing rows get ``model = ''`` (the provider-level fallback).

Postgres can ALTER the PK in place; SQLite can't drop an unnamed PK, so we
recreate the table and copy the rows.

Revision: 0003_provider_cost_model
Down: 0002_api_db_restructure
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0003_provider_cost_model"
down_revision = "0002_api_db_restructure"
branch_labels = None
depends_on = None


def _is_sqlite() -> bool:
    return op.get_bind().dialect.name == "sqlite"


def upgrade() -> None:
    if _is_sqlite():
        op.rename_table("provider_costs", "provider_costs_old")
        op.create_table(
            "provider_costs",
            sa.Column("kind", sa.String(20), nullable=False),
            sa.Column("provider", sa.String(40), nullable=False),
            sa.Column("model", sa.String(60), nullable=False, server_default=""),
            sa.Column("cost_per_min", sa.Float, nullable=False, server_default="0"),
            sa.Column("updated_at", sa.DateTime, server_default=sa.func.now()),
            sa.PrimaryKeyConstraint("kind", "provider", "model"),
        )
        op.execute(
            "INSERT INTO provider_costs (kind, provider, model, cost_per_min, updated_at) "
            "SELECT kind, provider, '', cost_per_min, updated_at FROM provider_costs_old"
        )
        op.drop_table("provider_costs_old")
    else:
        op.add_column("provider_costs", sa.Column("model", sa.String(60), nullable=False, server_default=""))
        op.drop_constraint("provider_costs_pkey", "provider_costs", type_="primary")
        op.create_primary_key("provider_costs_pkey", "provider_costs", ["kind", "provider", "model"])


def downgrade() -> None:
    # Collapse to (kind, provider): keep only the provider-level rows.
    op.execute("DELETE FROM provider_costs WHERE model <> ''")
    if _is_sqlite():
        op.rename_table("provider_costs", "provider_costs_old")
        op.create_table(
            "provider_costs",
            sa.Column("kind", sa.String(20), nullable=False),
            sa.Column("provider", sa.String(40), nullable=False),
            sa.Column("cost_per_min", sa.Float, nullable=False, server_default="0"),
            sa.Column("updated_at", sa.DateTime, server_default=sa.func.now()),
            sa.PrimaryKeyConstraint("kind", "provider"),
        )
        op.execute(
            "INSERT INTO provider_costs (kind, provider, cost_per_min, updated_at) "
            "SELECT kind, provider, cost_per_min, updated_at FROM provider_costs_old"
        )
        op.drop_table("provider_costs_old")
    else:
        op.drop_constraint("provider_costs_pkey", "provider_costs", type_="primary")
        op.create_primary_key("provider_costs_pkey", "provider_costs", ["kind", "provider"])
        op.drop_column("provider_costs", "model")
