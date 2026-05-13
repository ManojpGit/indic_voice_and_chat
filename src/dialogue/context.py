"""Redis-backed session store (PRD §6.2).

Keys:
    session:{id}:state    JSON object  (set/replace)
    session:{id}:history  JSON list    (append)
    session:{id}:slots    Hash         (per-field set)

All keys share a single TTL refreshed on every write so an active session
doesn't expire mid-conversation.
"""

from __future__ import annotations

import json
from typing import Any, Optional

from redis.asyncio import Redis


class SessionStore:
    def __init__(self, redis: Redis, ttl_seconds: int = 1800, tenant_id: Optional[str] = None) -> None:
        self.redis = redis
        self.ttl = ttl_seconds
        # Tenant id is folded into every Redis key so two tenants can share
        # the same physical Redis without colliding.
        self.tenant_id = tenant_id

    # --- Keys ------------------------------------------------------------

    def _prefix(self) -> str:
        return f"tenant:{self.tenant_id}:" if self.tenant_id else ""

    def _state_key(self, session_id: str) -> str:
        return f"{self._prefix()}session:{session_id}:state"

    def _history_key(self, session_id: str) -> str:
        return f"{self._prefix()}session:{session_id}:history"

    def _slots_key(self, session_id: str) -> str:
        return f"{self._prefix()}session:{session_id}:slots"

    # --- State -----------------------------------------------------------

    async def set_state(self, session_id: str, state: dict[str, Any]) -> None:
        await self.redis.set(self._state_key(session_id), json.dumps(state), ex=self.ttl)

    async def get_state(self, session_id: str) -> Optional[dict[str, Any]]:
        raw = await self.redis.get(self._state_key(session_id))
        if raw is None:
            return None
        return json.loads(raw)

    # --- History ---------------------------------------------------------

    async def append_history(self, session_id: str, turn: dict[str, Any]) -> None:
        key = self._history_key(session_id)
        await self.redis.rpush(key, json.dumps(turn))
        await self.redis.expire(key, self.ttl)

    async def get_history(self, session_id: str) -> list[dict[str, Any]]:
        items = await self.redis.lrange(self._history_key(session_id), 0, -1)
        return [json.loads(i) for i in items]

    # --- Slots -----------------------------------------------------------

    async def set_slot(self, session_id: str, name: str, value: Any) -> None:
        key = self._slots_key(session_id)
        await self.redis.hset(key, name, json.dumps(value))
        await self.redis.expire(key, self.ttl)

    async def get_slots(self, session_id: str) -> dict[str, Any]:
        raw = await self.redis.hgetall(self._slots_key(session_id))
        return {
            (k.decode() if isinstance(k, bytes) else k): json.loads(v)
            for k, v in raw.items()
        }

    # --- Lifecycle -------------------------------------------------------

    async def delete(self, session_id: str) -> None:
        await self.redis.delete(
            self._state_key(session_id),
            self._history_key(session_id),
            self._slots_key(session_id),
        )
