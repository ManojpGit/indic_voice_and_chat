"""Telephony provider webhook endpoints (PRD §7.4).

Phase 3: Twilio voice webhook + Media Streams websocket.
Other providers (Exotel, Stringee) deferred to Phase 5+.
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import Response

from src.api.telephony_twilio import voice_twiml

log = logging.getLogger(__name__)

router = APIRouter(prefix="/telephony", tags=["telephony"])

# A factory for per-call agent + bridge wiring is registered at app startup
# (Phase 5+ will plug a real campaign-aware factory in here). For Phase 3
# the endpoint is wired but unused — set ``router.bridge_factory`` from
# tests or app code when needed.
_bridge_factory: Optional[callable] = None  # type: ignore[type-arg]


def set_bridge_factory(factory) -> None:
    """Register the callable that produces ``(bridge, run_coro)`` per call."""
    global _bridge_factory
    _bridge_factory = factory


@router.post("/twilio/voice", response_class=Response)
async def twilio_voice(request: Request) -> Response:
    """Twilio voice webhook — returns TwiML opening a media stream."""
    base = request.headers.get("x-forwarded-host") or request.url.netloc
    scheme = "wss" if request.url.scheme == "https" else "ws"
    stream_url = f"{scheme}://{base}/api/v1/telephony/twilio/stream"
    body = voice_twiml(stream_url)
    return Response(content=body, media_type="application/xml")


@router.websocket("/twilio/stream")
async def twilio_stream(websocket: WebSocket) -> None:
    """Twilio Media Streams websocket — bridges audio to the agent."""
    await websocket.accept()
    if _bridge_factory is None:
        log.warning("twilio stream connected but no bridge factory registered")
        await websocket.close(code=1011, reason="bridge factory unset")
        return

    bridge = _bridge_factory(websocket)
    try:
        await bridge.run()
    except WebSocketDisconnect:
        log.info("twilio stream client disconnected")
    except Exception:  # noqa: BLE001 — never let the websocket task escape
        log.exception("twilio stream bridge crashed")
    finally:
        try:
            await websocket.close()
        except Exception:  # noqa: BLE001
            pass
