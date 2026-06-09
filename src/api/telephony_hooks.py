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
from collections.abc import Callable

import httpx
from fastapi import APIRouter, Form, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, Response

from src.api.telephony_exotel import voicebot_xml
from src.api.telephony_stringee import reprompt_scco
from src.api.telephony_stringee_bridge import StringeeIvrBridge, registry
from src.api.telephony_twilio import voice_twiml
from src.auth import TenantContext
from src.auth.middleware import tenant_from_twilio_to_number

log = logging.getLogger(__name__)
router = APIRouter(prefix="/telephony", tags=["telephony"])


# Factory: takes (websocket, tenant_context) -> bridge instance with .run()
BridgeFactory = Callable[[WebSocket, TenantContext], object]
_bridge_factory: BridgeFactory | None = None
_exotel_bridge_factory: BridgeFactory | None = None


def set_bridge_factory(factory: BridgeFactory | None) -> None:
    """Register the Twilio bridge factory."""
    global _bridge_factory
    _bridge_factory = factory


def set_exotel_bridge_factory(factory: BridgeFactory | None) -> None:
    """Register the Exotel bridge factory."""
    global _exotel_bridge_factory
    _exotel_bridge_factory = factory


_stringee_bridge_factory: Callable[..., StringeeIvrBridge] | None = None


def set_stringee_bridge_factory(factory) -> None:
    """Register the Stringee IVR bridge factory."""
    global _stringee_bridge_factory
    _stringee_bridge_factory = factory


@router.post("/twilio/voice", response_class=Response)
async def twilio_voice(
    request: Request,
    To: str = Form(...),
    From: str | None = Form(None),
    CallSid: str | None = Form(None),
    Direction: str | None = Form(None),
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
    From: str | None = Form(None),
    CallSid: str | None = Form(None),
    Direction: str | None = Form(None),
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


# --- Stringee IVR webhook routes --------------------------------------------


def _stringee_base(request: Request) -> str:
    base = request.headers.get("x-forwarded-host") or request.url.netloc
    proto = request.headers.get("x-forwarded-proto")
    scheme = "https" if (proto == "https" or request.url.scheme == "https") else "http"
    return f"{scheme}://{base}/api/v1/telephony/stringee"


async def _download(url: str) -> bytes:
    async with httpx.AsyncClient(timeout=httpx.Timeout(8.0, connect=5.0)) as c:
        resp = await c.get(url)
        resp.raise_for_status()
        return resp.content


@router.post("/stringee/answer")
async def stringee_answer(request: Request):
    """Call answered -> build the call's bridge and return the opening SCCO."""
    body = await request.json()
    call_id = str(body.get("call_id") or body.get("callId") or "")
    direction = (body.get("direction") or "").lower()
    to_num, from_num = body.get("to"), body.get("from")
    lookup = from_num if (direction.startswith("outbound") and from_num) else to_num
    tenant = await tenant_from_twilio_to_number(lookup)
    if _stringee_bridge_factory is None:
        return Response(status_code=503)
    bridge = _stringee_bridge_factory(
        call_id=call_id, tenant=tenant,
        base_url=_stringee_base(request), fetch=_download,
    )
    registry.put(bridge)
    scco = await bridge.start_call()
    log.info("stringee answer", extra={"tenant": tenant.slug, "call_id": call_id})
    return JSONResponse(scco)


@router.post("/stringee/event/{tenant_slug}")
async def stringee_event(tenant_slug: str, request: Request, call_id: str):
    """Per-turn recordMessage webhook -> run a turn -> return the next SCCO.

    Note: tenant_slug is URL namespacing only; the bridge is looked up by
    call_id — tenant_slug is not validated here.
    """
    body = await request.json()
    rec_url = body.get("recording_url") or body.get("url") or body.get("link")
    bridge = registry.get(call_id)
    if bridge is None or not rec_url:
        base = _stringee_base(request)
        return JSONResponse(reprompt_scco(
            text="Maaf kijiye, dobara boliye?",
            event_url=f"{base}/event/{tenant_slug}?call_id={call_id}",
        ))
    scco = await bridge.handle_turn(recording_url=rec_url)
    return JSONResponse(scco)


@router.get("/stringee/audio/{token}")
async def stringee_audio(token: str, call_id: str | None = None):
    """Serve a hosted reply/opening WAV for Stringee's `play` to fetch."""
    if call_id:
        bridge = registry.get(call_id)
        if bridge is not None:
            wav = bridge.audio.get(token)
            if wav is not None:
                return Response(content=wav, media_type="audio/wav")
    # Stringee may fetch without our call_id query: scan live calls.
    for b in registry.iter_bridges():
        wav = b.audio.get(token)
        if wav is not None:
            return Response(content=wav, media_type="audio/wav")
    return Response(status_code=404)


@router.post("/stringee/status/{tenant_slug}")
async def stringee_status(tenant_slug: str, request: Request):
    """Lifecycle webhook: on call end, record the outcome and clean up."""
    body = await request.json()
    call_id = str(body.get("call_id") or body.get("callId") or "")
    status = (body.get("status") or "").upper()
    log.info("stringee status", extra={"call_id": call_id, "status": status})
    if status in ("ENDED", "FAILED", "NO_ANSWER", "BUSY"):
        await registry.end(call_id)
    return Response(status_code=200)
