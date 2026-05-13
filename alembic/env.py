"""Alembic env — async-aware, sources URL from app config / env."""

from __future__ import annotations

import asyncio
from logging.config import fileConfig

from alembic import context
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

# Import models so their metadata is registered against Base.
from src.config import load_settings
from src.models import Base  # noqa: F401  (side-effect: registers all tables)

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Resolve DB URL: explicit -x url=..., else app settings.
_x = context.get_x_argument(as_dictionary=True)
db_url = _x.get("url") or config.get_main_option("sqlalchemy.url")
if not db_url:
    db_url = load_settings().database.url
config.set_main_option("sqlalchemy.url", db_url)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    context.configure(
        url=db_url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    context.configure(connection=connection, target_metadata=target_metadata)
    with context.begin_transaction():
        context.run_migrations()


async def run_migrations_online() -> None:
    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_migrations_online())
