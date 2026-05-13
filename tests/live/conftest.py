"""Live-test fixtures.

These tests hit real provider APIs and cost money on every run. They're
gated by ``VOX_LIVE_TESTS=1`` plus the presence of every required env var
(per-tenant API keys). Anything missing → automatic skip with a clear
message so it's obvious why nothing ran.

Run with:
    VOX_LIVE_TESTS=1 pytest tests/live/ -m live -v
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

from src.auth.context import TenantContext
from src.config_tenant import TenantSettings, load_tenant

# Auto-load a project-root ``.env`` so live keys can live in one
# gitignored file instead of needing to be ``export``-ed every session.
#
# ``override=True`` is deliberate here: the project-root ``tests/conftest.py``
# pre-sets dummy values (``GROQ_API_KEY=test-groq-key`` etc.) at import time
# so non-live unit tests don't blow up on missing env vars. For live tests,
# those dummies would shadow the real ``.env`` values AND defeat the
# ``${VAR}`` interpolation in the per-tenant aliases. The live conftest
# only loads when live tests are in scope, so this override never affects
# the mocked unit suite.
_ENV_FILE = Path(__file__).resolve().parents[2] / ".env"
if _ENV_FILE.exists():
    load_dotenv(_ENV_FILE, override=True)


def _live_enabled() -> bool:
    return os.environ.get("VOX_LIVE_TESTS", "").strip().lower() in {"1", "true", "yes"}


def _missing_env_vars(settings: TenantSettings) -> list[str]:
    """Collect every env var the tenant config references that isn't set."""
    candidates = [
        settings.pipeline.stt.api_key_env,
        settings.pipeline.llm.api_key_env,
        settings.pipeline.tts.api_key_env,
        settings.pipeline.telephony.account_sid_env,
        settings.pipeline.telephony.auth_token_env,
    ]
    return [c for c in candidates if c and not os.environ.get(c)]


@pytest.fixture(scope="session")
def dev_tenant() -> TenantSettings:
    if not _live_enabled():
        pytest.skip("VOX_LIVE_TESTS is not set — skipping live provider tests")
    try:
        settings = load_tenant("dev", tenant_dir=Path("config/tenants"))
    except FileNotFoundError as e:
        pytest.skip(f"dev tenant config not found: {e}")
    missing = _missing_env_vars(settings)
    if missing:
        pytest.skip(
            f"missing live-test env vars: {missing}. Set them locally then "
            f"re-run with VOX_LIVE_TESTS=1."
        )
    return settings


@pytest.fixture(scope="session")
def dev_tenant_ctx(dev_tenant: TenantSettings) -> TenantContext:
    return TenantContext(settings=dev_tenant)
