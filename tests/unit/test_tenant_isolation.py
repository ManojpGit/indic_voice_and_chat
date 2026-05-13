"""Cross-tenant state-isolation tests.

Each state-holding component must not leak between tenants.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.campaign.dnd_filter import (
    CallingHoursPolicy,
    DNDFilter,
    InMemoryDNDStore,
)
from src.campaign.scheduler import CallScheduler, RateLimitConfig
from src.dialogue.context import SessionStore
from src.integration.event_bus import Event, EventBus
from src.integration.webhooks import WebhookConfig, WebhookManager
from src.interfaces.vector_store import Document
from src.providers.vector_store.faiss_store import FAISSAdapter


# --- SessionStore -------------------------------------------------------


@pytest.mark.asyncio
async def test_session_store_isolates_tenants(fake_redis) -> None:
    """Two SessionStore instances with different tenant_ids must not see
    each other's keys even when sharing the same Redis client."""
    a = SessionStore(fake_redis, ttl_seconds=300, tenant_id="t_acme")
    g = SessionStore(fake_redis, ttl_seconds=300, tenant_id="t_globex")

    await a.set_state("s1", {"who": "acme"})
    await g.set_state("s1", {"who": "globex"})

    # Same session_id, different tenant — each sees only its own state.
    assert (await a.get_state("s1"))["who"] == "acme"
    assert (await g.get_state("s1"))["who"] == "globex"


@pytest.mark.asyncio
async def test_session_store_history_isolated(fake_redis) -> None:
    a = SessionStore(fake_redis, tenant_id="t_acme")
    g = SessionStore(fake_redis, tenant_id="t_globex")

    await a.append_history("s1", {"role": "user", "content": "acme line"})
    await g.append_history("s1", {"role": "user", "content": "globex line"})

    assert (await a.get_history("s1"))[0]["content"] == "acme line"
    assert (await g.get_history("s1"))[0]["content"] == "globex line"


@pytest.mark.asyncio
async def test_session_store_delete_only_scrubs_own_tenant(fake_redis) -> None:
    a = SessionStore(fake_redis, tenant_id="t_acme")
    g = SessionStore(fake_redis, tenant_id="t_globex")
    await a.set_state("s1", {"x": 1})
    await g.set_state("s1", {"x": 2})

    await a.delete("s1")
    assert await a.get_state("s1") is None
    assert await g.get_state("s1") == {"x": 2}


@pytest.mark.asyncio
async def test_session_store_backwards_compat_when_tenant_id_none(fake_redis) -> None:
    """Legacy callers without tenant_id keep working with un-prefixed keys."""
    store = SessionStore(fake_redis)
    await store.set_state("s1", {"x": 1})
    assert await store.get_state("s1") == {"x": 1}


# --- DND filter per tenant ---------------------------------------------


def test_dnd_filter_per_tenant_isolation() -> None:
    acme_filter = DNDFilter(InMemoryDNDStore(["+919999999991"]))
    globex_filter = DNDFilter(InMemoryDNDStore(["+919999999992"]))
    # acme's number is blocked for acme but not for globex
    assert acme_filter.is_blocked("+919999999991") is True
    assert globex_filter.is_blocked("+919999999991") is False
    # And vice versa
    assert globex_filter.is_blocked("+919999999992") is True
    assert acme_filter.is_blocked("+919999999992") is False


# --- Scheduler rate-limit per tenant -----------------------------------


def test_scheduler_rate_limit_per_tenant() -> None:
    hours = CallingHoursPolicy(start="00:00", end="23:59", skip_weekday=None)
    acme_sched = CallScheduler(hours=hours, dnd_filter=DNDFilter(InMemoryDNDStore()),
                               rate_limit=RateLimitConfig(calls_per_minute=2, max_concurrent_calls=10))
    globex_sched = CallScheduler(hours=hours, dnd_filter=DNDFilter(InMemoryDNDStore()),
                                 rate_limit=RateLimitConfig(calls_per_minute=2, max_concurrent_calls=10))
    from datetime import datetime
    from src.campaign.dnd_filter import IST

    now = datetime(2026, 5, 12, 14, tzinfo=IST)
    acme_sched.mark_attempted(now)
    acme_sched.mark_attempted(now)
    # Globex's rate window is untouched.
    from src.campaign.models import Lead, LeadStatus
    lead = Lead(id="l1", tenant_id="t_globex", phone_number="+91")
    out = globex_sched.poll(leads=[lead], active_count=0, now=now)
    assert not out.blocked_by_rate
    # Acme is throttled.
    a_out = acme_sched.poll(leads=[lead], active_count=0, now=now)
    assert a_out.blocked_by_rate is True


# --- FAISS per-tenant directories --------------------------------------


@pytest.mark.asyncio
async def test_faiss_per_tenant_index_dirs(tmp_path: Path) -> None:
    """Each tenant's FAISS adapter writes to its own subdirectory; an
    indexed doc in tenant A is not visible to tenant B."""
    a_path = tmp_path / "t_acme" / "index"
    g_path = tmp_path / "t_globex" / "index"
    a_store = FAISSAdapter({"embedding_dim": 4, "index_path": str(a_path)})
    g_store = FAISSAdapter({"embedding_dim": 4, "index_path": str(g_path)})

    await a_store.index([Document(id="d1", content="acme doc", embedding=[1.0, 0, 0, 0])])
    # globex's index is untouched.
    assert await a_store.count() == 1
    assert await g_store.count() == 0

    # And reopen each from disk to confirm files are isolated.
    a_reopen = FAISSAdapter({"embedding_dim": 4, "index_path": str(a_path)})
    g_reopen = FAISSAdapter({"embedding_dim": 4, "index_path": str(g_path)})
    assert await a_reopen.count() == 1
    assert await g_reopen.count() == 0


# --- WebhookManager per tenant -----------------------------------------


@pytest.mark.asyncio
async def test_webhook_managers_independent() -> None:
    """Two WebhookManager instances on a *shared* EventBus must not deliver
    each other's tenant-scoped events. We model this by giving each tenant
    its own EventBus (which is how the runtime registry wires it)."""
    bus_a = EventBus()
    bus_g = EventBus()
    posts_a: list[tuple[str, dict]] = []
    posts_g: list[tuple[str, dict]] = []

    async def post_a(url, body, timeout):
        posts_a.append((url, body))
        return 200

    async def post_g(url, body, timeout):
        posts_g.append((url, body))
        return 200

    mgr_a = WebhookManager(bus=bus_a, http_post=post_a,
                           config=WebhookConfig(max_attempts=1, backoff_base_s=0.0))
    mgr_g = WebhookManager(bus=bus_g, http_post=post_g,
                           config=WebhookConfig(max_attempts=1, backoff_base_s=0.0))
    mgr_a.register("https://acme.example/wh", ["*"])
    mgr_g.register("https://globex.example/wh", ["*"])

    await bus_a.publish(Event(type="call.completed", payload={"tenant": "acme"}))
    await bus_g.publish(Event(type="call.completed", payload={"tenant": "globex"}))

    assert len(posts_a) == 1 and posts_a[0][0] == "https://acme.example/wh"
    assert len(posts_g) == 1 and posts_g[0][0] == "https://globex.example/wh"
