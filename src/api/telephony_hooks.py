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

from src.api.telephony_exotel import voicebot_xml
from src.api.telephony_twilio import voice_twiml
from src.auth import TenantContext
from src.auth.middleware import tenant_from_twilio_to_number, tenant_from_ws_query

log = logging.getLogger(__name__)
router = APIRouter(prefix="/telephony", tags=["telephony"])


# Factory: takes (websocket, tenant_context) -> bridge instance with .run()
BridgeFactory = Callable[[WebSocket, TenantContext], object]
_bridge_factory: Optional[BridgeFactory] = None
_exotel_bridge_factory: Optional[BridgeFactory] = None


def set_bridge_factory(factory: Optional[BridgeFactory]) -> None:
    """Register the Twilio bridge factory."""
    global _bridge_factory
    _bridge_factory = factory


def set_exotel_bridge_factory(factory: Optional[BridgeFactory]) -> None:
    """Register the Exotel bridge factory."""
    global _exotel_bridge_factory
    _exotel_bridge_factory = factory


@router.post("/twilio/voice", response_class=Response)
async def twilio_voice(
    request: Request,
    To: str = Form(...),
    From: Optional[str] = Form(None),
    CallSid: Optional[str] = Form(None),
    Direction: Optional[str] = Form(None),
) -> Response:
    """Twilio voice webhook → returns TwiML opening a tenant-aware media stream.

    Tenant resolution is direction-aware:
    - **inbound** (a customer dials our Twilio number): ``To`` is the
      owned number; look it up in ``tenant_phone_numbers``.
    - **outbound-api / outbound-dial** (we initiated the call via the
      orchestrator or place_test_call.py): ``To`` is the dialed destination
      (an end-user number we don't own); the tenant owns ``From`` instead.

    The resolved slug is embedded in the WS URL so the stream handler can
    re-resolve the same tenant on connect.
    """
    is_outbound = (Direction or "").startswith("outbound")
    lookup_number = From if (is_outbound and From) else To
    tenant = await tenant_from_twilio_to_number(lookup_number)

    base = request.headers.get("x-forwarded-host") or request.url.netloc
    forwarded_proto = request.headers.get("x-forwarded-proto")
    scheme = "wss" if (forwarded_proto == "https" or request.url.scheme == "https") else "ws"
    # Tenant slug goes in the URL **path** — Twilio strips query strings
    # from <Stream url=...> attributes when opening the WSS connection.
    stream_url = f"{scheme}://{base}/api/v1/telephony/twilio/stream/{tenant.slug}"
    body = voice_twiml(stream_url)
    log.info(
        "twilio voice webhook",
        extra={
            "tenant": tenant.slug, "direction": Direction,
            "to": To, "from": From, "sid": CallSid,
        },
    )
    return Response(content=body, media_type="application/xml")


@router.websocket("/twilio/stream/{tenant_slug}")
async def twilio_stream(websocket: WebSocket, tenant_slug: str) -> None:
    """Twilio Media Streams websocket → bridges audio to the tenant's agent.

    Tenant slug arrives as a path segment because Twilio strips query
    strings from ``<Stream url=...>`` attributes when establishing the
    WSS connection.
    """
    from src.auth.middleware import tenant_from_slug

    await websocket.accept()
    try:
        tenant = await tenant_from_slug(tenant_slug)
    except HTTPException as e:
        log.warning("twilio stream tenant resolution failed: %s", e.detail)
        await websocket.close(code=1008 if e.status_code == 404 else 1011, reason=str(e.detail))
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


# --- Exotel webhook + WS ----------------------------------------------------


@router.post("/exotel/voice", response_class=Response)
async def exotel_voice(
    request: Request,
    To: str = Form(...),
    From: Optional[str] = Form(None),
    CallSid: Optional[str] = Form(None),
    Direction: Optional[str] = Form(None),
) -> Response:
    """Exotel Passthru / Voicebot webhook → returns ExotelML opening a stream.

    Exotel's form params mirror Twilio's (``To``, ``From``, ``CallSid``,
    ``Direction``) so the resolution logic is identical: for outbound calls
    the tenant owns ``From``; for inbound it owns ``To``.
    """
    is_outbound = (Direction or "").startswith("outbound")
    lookup_number = From if (is_outbound and From) else To
    tenant = await tenant_from_twilio_to_number(lookup_number)

    base = request.headers.get("x-forwarded-host") or request.url.netloc
    forwarded_proto = request.headers.get("x-forwarded-proto")
    scheme = "wss" if (forwarded_proto == "https" or request.url.scheme == "https") else "ws"
    stream_url = f"{scheme}://{base}/api/v1/telephony/exotel/stream/{tenant.slug}"
    body = voicebot_xml(stream_url)
    log.info(
        "exotel voice webhook",
        extra={
            "tenant": tenant.slug, "direction": Direction,
            "to": To, "from": From, "sid": CallSid,
        },
    )
    return Response(content=body, media_type="application/xml")


@router.websocket("/exotel/stream/{tenant_slug}")
async def exotel_stream(websocket: WebSocket, tenant_slug: str) -> None:
    """Exotel Voicebot Streaming websocket → bridges audio to the agent."""
    from src.auth.middleware import tenant_from_slug

    await websocket.accept()
    try:
        tenant = await tenant_from_slug(tenant_slug)
    except HTTPException as e:
        log.warning("exotel stream tenant resolution failed: %s", e.detail)
        await websocket.close(code=1008 if e.status_code == 404 else 1011, reason=str(e.detail))
        return

    if _exotel_bridge_factory is None:
        log.warning("exotel stream connected but no bridge factory registered")
        await websocket.close(code=1011, reason="exotel bridge factory unset")
        return

    bridge = _exotel_bridge_factory(websocket, tenant)
    try:
        await bridge.run()
    except WebSocketDisconnect:
        log.info("exotel stream client disconnected", extra={"tenant": tenant.slug})
    except Exception:  # noqa: BLE001
        log.exception("exotel stream bridge crashed", extra={"tenant": tenant.slug})
    finally:
        try:
            await websocket.close()
        except Exception:  # noqa: BLE001
            pass
