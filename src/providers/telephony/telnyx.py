"""Telnyx Programmable Voice / Call Control adapter.

Telnyx is a US-domiciled CPaaS with direct Indian carrier interconnects.
Signup takes minutes (light ID upload, auto-approved, not the multi-day
business KYC Exotel requires) and per-minute India pricing is ~$0.012
— roughly 20x cheaper than Infobip's per-call fee.

REST surface implemented:
- ``initiate_call``  POST /v2/calls
- ``hangup``         POST /v2/calls/{call_control_id}/actions/hangup
- ``transfer``       POST /v2/calls/{call_control_id}/actions/transfer

Telnyx uses a "Call Control" model: each call gets a stable
``call_control_id`` returned from create-call, and every later action
references that ID via ``/v2/calls/{id}/actions/<verb>`` paths.

Key shape differences from Twilio/Exotel:
- **Single global base URL** (``api.telnyx.com``) — unlike Infobip's
  per-account subdomain.
- **Bearer-token auth** — ``Authorization: Bearer <key>`` (the simplest
  scheme of any of our adapters).
- **``connection_id`` is required** — points at the Voice API
  Application in your Telnyx portal that defines webhook URLs and
  codecs. This is the analogue of Twilio's TwiML App or Infobip's
  Calls Application.
- **Status comes from webhooks, not the create response** — the
  POST /v2/calls response only returns IDs (call_control_id, call_leg_id,
  call_session_id), not a state field. We map to ``ringing`` on success.

Media Streams: Telnyx's bidirectional audio runs over a WebSocket whose
framing matches Twilio's Media Streams almost exactly (μ-law @ 8kHz,
base64 JSON envelopes with ``event: start | media | stop``). When wired,
``TwilioMediaBridge`` should serve Telnyx with minimal changes — but the
streaming-start trigger lives in Telnyx's API rather than in a TwiML
``<Connect><Stream>`` directive, so the activation path is different and
we stub these methods for now.

Endpoint reference: https://developers.telnyx.com/api/call-control/dial-call
"""

from __future__ import annotations

import os
from typing import Any, AsyncIterator

import httpx

from src.interfaces.telephony import (
    CallConfig,
    CallSession,
    ITelephonyProvider,
)


TELNYX_BASE_URL = "https://api.telnyx.com"


class TelnyxAdapter(ITelephonyProvider):
    def __init__(self, config: dict[str, Any]) -> None:
        api_key = config.get("api_key") or os.environ.get("TELNYX_API_KEY")
        # connection_id is the Voice API Application id from the Telnyx portal.
        # Telnyx's API rejects POST /v2/calls without one (you can't just dial
        # in a vacuum — the connection defines codecs, webhook URLs, etc.).
        self._connection_id = (
            config.get("connection_id")
            or config.get("connectionId")
            or os.environ.get("TELNYX_CONNECTION_ID")
        )
        if not (api_key and self._connection_id):
            raise ValueError(
                "TelnyxAdapter requires api_key + connection_id "
                "(or TELNYX_API_KEY / TELNYX_CONNECTION_ID env vars). "
                "Get connection_id from Mission Control Portal → Voice → "
                "Programmable Voice → Applications."
            )
        self._api_key = api_key
        self._base_url = config.get("base_url", TELNYX_BASE_URL).rstrip("/")
        self._timeout = config.get("timeout", 30.0)

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    async def initiate_call(self, config: CallConfig) -> CallSession:
        """Place an outbound call via Telnyx Call Control.

        Unlike Twilio, Telnyx's webhook URL is not per-call — it's
        configured on the Connection (``connection_id``). So
        ``config.webhook_url`` is ignored here; the Voice Application in
        your Telnyx portal must already point at the vox-agent voice route.
        """
        body = {
            "to": config.to_number,
            "from": config.from_number,
            "connection_id": self._connection_id,
        }
        if config.timeout_seconds:
            body["timeout_secs"] = config.timeout_seconds
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(
                f"{self._base_url}/v2/calls",
                headers=self._headers(),
                json=body,
            )
            resp.raise_for_status()
            payload = resp.json()
        # Telnyx wraps everything in {"data": {...}}; the IDs live there.
        data = payload.get("data") or {}
        call_control_id = data.get("call_control_id") or data.get("call_leg_id") or ""
        # No state field in the create response — Telnyx publishes status
        # via webhook events (call.initiated, call.answered, call.hangup).
        # A successful 200 means the call is queued / ringing.
        return CallSession(
            session_id=str(call_control_id),
            status="ringing",
            to_number=config.to_number,
            from_number=config.from_number,
        )

    async def hangup(self, session_id: str) -> None:
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(
                f"{self._base_url}/v2/calls/{session_id}/actions/hangup",
                headers=self._headers(),
            )
            # 404 = call already ended / unknown id; tolerate it.
            # 422 = call already in finishing state; also benign.
            if resp.status_code >= 500:
                resp.raise_for_status()

    async def transfer(self, session_id: str, to_number: str) -> None:
        """Transfer a live call to another phone number."""
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(
                f"{self._base_url}/v2/calls/{session_id}/actions/transfer",
                headers=self._headers(),
                json={"to": to_number},
            )
            resp.raise_for_status()

    # --- Media Streams stubs --------------------------------------------
    # Telnyx's bidirectional audio runs over a WSS that frames audio
    # nearly identically to Twilio Media Streams (μ-law @ 8kHz, base64
    # JSON envelopes — event: start | media | stop). To wire it:
    #
    #   1. Trigger streaming via
    #      POST /v2/calls/{call_control_id}/actions/streaming_start
    #      with body {"stream_url": "wss://.../telnyx/stream/<tenant>",
    #                 "stream_track": "both_tracks",
    #                 "codec": "PCMU"}
    #   2. Accept the WSS at the route layer with our existing
    #      TwilioMediaBridge — the JSON event shape is compatible enough
    #      that the bridge needs only field-name adjustments (Telnyx
    #      uses ``call_control_id`` in start frames instead of ``streamSid``).
    #
    # Left as NotImplementedError until that bridge / streaming-start
    # action are wired.

    async def stream_audio_in(self, session_id: str) -> AsyncIterator[bytes]:
        raise NotImplementedError(
            "Telnyx audio streaming is triggered via the streaming_start "
            "Call Control action, then runs over a WSS that the "
            "TelnyxMediaBridge will consume (not yet built)."
        )
        if False:  # pragma: no cover
            yield b""

    async def stream_audio_out(
        self,
        session_id: str,
        audio_stream: AsyncIterator[bytes],
    ) -> None:
        raise NotImplementedError(
            "Telnyx audio streaming is triggered via the streaming_start "
            "Call Control action, then runs over a WSS that the "
            "TelnyxMediaBridge will consume (not yet built)."
        )
