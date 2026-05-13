from __future__ import annotations

import pytest

from src.dialogue.context import SessionStore


@pytest.mark.asyncio
async def test_state_round_trip(fake_redis):
    store = SessionStore(fake_redis, ttl_seconds=300)
    await store.set_state("s1", {"phase": "opening", "lead": "Manoj"})
    state = await store.get_state("s1")
    assert state == {"phase": "opening", "lead": "Manoj"}


@pytest.mark.asyncio
async def test_history_append_and_read(fake_redis):
    store = SessionStore(fake_redis, ttl_seconds=300)
    await store.append_history("s1", {"role": "agent", "content": "hi"})
    await store.append_history("s1", {"role": "user", "content": "namaste"})
    history = await store.get_history("s1")
    assert history == [
        {"role": "agent", "content": "hi"},
        {"role": "user", "content": "namaste"},
    ]


@pytest.mark.asyncio
async def test_slots_set_and_get(fake_redis):
    store = SessionStore(fake_redis, ttl_seconds=300)
    await store.set_slot("s1", "interest_level", "hot")
    await store.set_slot("s1", "callback_time", None)
    slots = await store.get_slots("s1")
    assert slots == {"interest_level": "hot", "callback_time": None}


@pytest.mark.asyncio
async def test_unknown_session_returns_none(fake_redis):
    store = SessionStore(fake_redis, ttl_seconds=300)
    assert await store.get_state("missing") is None
    assert await store.get_history("missing") == []
    assert await store.get_slots("missing") == {}


@pytest.mark.asyncio
async def test_delete_clears_all_keys(fake_redis):
    store = SessionStore(fake_redis, ttl_seconds=300)
    await store.set_state("s1", {"x": 1})
    await store.append_history("s1", {"role": "agent", "content": "hi"})
    await store.set_slot("s1", "k", "v")
    await store.delete("s1")
    assert await store.get_state("s1") is None
    assert await store.get_history("s1") == []
    assert await store.get_slots("s1") == {}
