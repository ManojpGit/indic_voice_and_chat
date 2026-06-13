"""_BaseLiveBridge teardown must always compute the outcome.

Regression: when a call dropped abnormally, the realtime session's aclose()
could raise during teardown and short-circuit the finally block *before*
_emit_outcome ran — so the conversation row was finalized with no outcome.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.api.live_bridge_base import _BaseLiveBridge
from src.interfaces.realtime import RealtimeConfig


class _BoomSession:
    async def events(self):
        return
        yield  # noqa: unreachable — makes this an (empty) async generator

    async def aclose(self):
        raise RuntimeError("session close boom")


class _FakeAgent:
    def __init__(self) -> None:
        self.started = False
        self.hung = False
        self.state = SimpleNamespace(is_terminal=False)

    async def start(self) -> None:
        self.started = True

    async def handle_hangup(self) -> None:
        self.hung = True


class _TestBridge(_BaseLiveBridge):
    def __init__(self, **kw) -> None:
        super().__init__(**kw)
        self.outcome_called = False

    async def _inbound_loop(self) -> None:
        return

    async def _send_audio_out(self, pcm16, rate) -> None:
        pass

    async def _send_interrupt(self) -> None:
        pass

    async def _emit_outcome(self) -> None:   # spy (skip real analysis)
        self.outcome_called = True


@pytest.mark.asyncio
async def test_teardown_emits_outcome_even_if_session_close_raises():
    agent = _FakeAgent()

    async def connect(cfg):
        return _BoomSession()

    bridge = _TestBridge(agent=agent, config=RealtimeConfig(model="m"), connect_session=connect)
    await bridge._drive()

    assert agent.started
    assert bridge.outcome_called   # reached despite aclose() raising
    assert agent.hung              # teardown ran to completion
