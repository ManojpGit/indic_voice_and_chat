"""Telephony provider webhook endpoints (PRD §7.4).

Twilio voice webhook + Media Streams websocket, tenant-aware.

Inbound flow:
1. Twilio rings the called number and POSTs to ``/twilio/voice`` with the
   ``To`` form param.
2. The voice handler resolves the tenant from the ``To`` number, builds
   the TwiML response with a ``<Stream url="wss://.../stream?tenant=<slug>"/>``
   so the websocket leg can re-resolve the same tenant.
3. Twilio opens the WebSocket; the WS handler reads ``?tenant=`` from the
   query string and asks the registered bridge factory for a per-call
   bridge wired with the tenant's agent + provider stack.
"""

from __future__ import annotations

import logging
from typing import Awaitable, Callable, Optional

from fastapi import APIRouter, Form, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import Response

from src.api.telephony_twilio import voice_twiml
from src.auth import TenantContext
from src.auth.middleware import tenant_from_twilio_to_number, tenant_from_ws_query

log = logging.getLogger(__name__)
router = APIRouter(prefix="/telephony", tags=["telephony"])


# Factory: takes (websocket, tenant_context) -> bridge instance with .run()
BridgeFactory = Callable[[WebSocket, TenantContext], object]
_bridge_factory: Optional[BridgeFactory] = None


def set_bridge_factory(factory: Optional[BridgeFactory]) -> None:
    global _bridge_factory
    _bridge_factory = factory


@router.post("/twilio/voice", response_class=Response)
async def twilio_voice(
    request: Request,
    To: str = Form(...),
    From: Optional[str] = Form(None),
    CallSid: Optional[str] = Form(None),
) -> Response:
    """Twilio voice webhook → returns TwiML opening a tenant-aware media stream.

    Tenant is resolved by looking up the Twilio number that was dialed in
    ``tenant_phone_numbers``. The resolved slug is embedded in the WS URL
    so the stream handler can re-resolve the same tenant on connect.
    """
    tenant = await tenant_from_twilio_to_number(To)

    base = request.headers.get("x-forwarded-host") or request.url.netloc
    forwarded_proto = request.headers.get("x-forwarded-proto")
    scheme = "wss" if (forwarded_proto == "https" or request.url.scheme == "https") else "ws"
    stream_url = f"{scheme}://{base}/api/v1/telephony/twilio/stream?tenant={tenant.slug}"
    body = voice_twiml(stream_url)
    log.info(
        "twilio voice webhook",
        extra={"tenant": tenant.slug, "to": To, "from": From, "sid": CallSid},
    )
    return Response(content=body, media_type="application/xml")


@router.websocket("/twilio/stream")
async def twilio_stream(websocket: WebSocket) -> None:
    """Twilio Media Streams websocket → bridges audio to the tenant's agent."""
    await websocket.accept()
    try:
        tenant = await tenant_from_ws_query(websocket)
    except HTTPException as e:
        log.warning("twilio stream tenant resolution failed: %s", e.detail)
        await websocket.close(code=1008, reason=str(e.detail))
        return

    if _bridge_factory is None:
        log.warning("twilio stream connected but no bridge factory registered")
        await websocket.close(code=1011, reason="bridge factory unset")
        return

    bridge = _bridge_factory(websocket, tenant)
    try:
        await bridge.run()
    except WebSocketDisconnect:
        log.info("twilio stream client disconnected", extra={"tenant": tenant.slug})
    except Exception:  # noqa: BLE001
        log.exception("twilio stream bridge crashed", extra={"tenant": tenant.slug})
    finally:
        try:
            await websocket.close()
        except Exception:  # noqa: BLE001
            pass
