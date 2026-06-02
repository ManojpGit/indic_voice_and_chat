"""Infobip Voice / Calls API adapter.

Infobip is an international CPaaS with direct +91 carrier interconnects.
Their trial accounts are usable for outbound voice to verified mobile
numbers without KYC — which is what makes them the fastest path to a
real Indian-mobile test call when Twilio's US-trial gets carrier-filtered
and Exotel's KYC hasn't cleared.

REST surface implemented here:
- ``initiate_call``  POST /calls/1/calls
- ``hangup``         POST /calls/1/calls/{callId}/hangup
- ``transfer``       POST /calls/1/calls/{callId}/transfer

Key differences from Twilio/Exotel adapters:
- **Auth header is literal ``Authorization: App <api_key>``** — not
  Basic auth, not Bearer. Easy to get wrong.
- **Base URL is per-account** — Infobip provisions a unique subdomain
  for each account (visible on the API keys page). Must be configured
  per tenant; there is no shared default.
- **``applicationId`` is required** — every call references a Calls
  Application (configured in the Infobip console). The application
  also points at the URL Infobip POSTs call events to (so this is
  also where the Voicebot Streaming WS gets wired up in Phase 2).

Media Streams (real-time bidirectional audio) goes through an Infobip
"Media-Stream" Calls Configuration: server-to-server WS with binary
PCM16 frames. Different framing from Twilio/Exotel, so the bridge is
a separate class (Phase 2 work). For now ``stream_audio_in/out`` raise
NotImplementedError pointing at that gap.

Endpoint reference: https://www.infobip.com/docs/api/channels/voice/calls
"""

from __future__ import annotations

import os
from typing import Any, AsyncIterator, Optional

import httpx

from src.interfaces.telephony import (
    CallConfig,
    CallSession,
    ITelephonyProvider,
)


# Infobip's call state enum -> our internal vocabulary.
# See https://www.infobip.com/docs/voice-and-video/calls (Call lifecycle).
_STATUS_MAP = {
    "NEW":             "ringing",
    "CALLING":         "ringing",
    "RINGING":         "ringing",
    "EARLY_MEDIA":     "ringing",
    "PRE_ESTABLISHED": "ringing",
    "ESTABLISHED":     "answered",
    "FINISHING":       "answered",
    "FINISHED":        "answered",
    "FAILED":          "failed",
    "BUSY":            "busy",
    "NO_ANSWER":       "no_answer",
    "CANCELLED":       "failed",
    "REJECTED":        "failed",
}


class InfobipAdapter(ITelephonyProvider):
    def __init__(self, config: dict[str, Any]) -> None:
        api_key = config.get("api_key") or os.environ.get("INFOBIP_API_KEY")
        # Each Infobip account has its own base URL (e.g.
        # ``https://abc123.api.infobip.com``). Visible on the API key page.
        base_url = (
            config.get("base_url")
            or os.environ.get("INFOBIP_BASE_URL")
        )
        # applicationId is required on every call create — it ties the call
        # to a Calls Application configured in the Infobip console (which
        # also points at the webhook URL Infobip POSTs call events to).
        self._application_id = (
            config.get("application_id")
            or config.get("applicationId")
            or os.environ.get("INFOBIP_APPLICATION_ID")
        )
        if not (api_key and base_url and self._application_id):
            raise ValueError(
                "InfobipAdapter requires api_key + base_url + application_id "
                "(or INFOBIP_API_KEY / INFOBIP_BASE_URL / INFOBIP_APPLICATION_ID "
                "env vars). The base URL is per-account — find it on the "
                "Infobip console's API keys page (e.g. https://<sub>.api.infobip.com)."
            )
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._timeout = config.get("timeout", 30.0)
        # ``connect_timeout`` is the per-call ring duration; Infobip default
        # is 30s. Keep ours separately so we don't conflict with httpx's
        # client-level timeout.
        self._connect_timeout = int(config.get("connect_timeout", 30))

    def _headers(self) -> dict[str, str]:
        # Infobip's auth scheme is literally ``App <key>`` — not Bearer, not Basic.
        return {
            "Authorization": f"App {self._api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    async def initiate_call(self, config: CallConfig) -> CallSession:
        """Place an outbound call.

        Infobip's Calls API does not take a per-call webhook URL the way
        Twilio does — webhooks are configured once on the parent ``application``
        and reused for every call referencing it. So we ignore
        ``config.webhook_url`` here; the caller must point their Calls
        Application at our voice route in the Infobip console.
        """
        body = {
            "applicationId": self._application_id,
            "endpoint": {
                "type": "PHONE",
                "phoneNumber": config.to_number.lstrip("+"),
            },
            "from": config.from_number.lstrip("+"),
            "connectTimeout": (
                config.timeout_seconds if config.timeout_seconds else self._connect_timeout
            ),
        }
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(
                f"{self._base_url}/calls/1/calls",
                headers=self._headers(),
                json=body,
            )
            resp.raise_for_status()
            payload = resp.json()
        call_id = payload.get("id") or payload.get("callId") or ""
        raw_state = (payload.get("state") or payload.get("callState") or "CALLING").upper()
        return CallSession(
            session_id=str(call_id),
            status=_STATUS_MAP.get(raw_state, raw_state.lower()),
            to_number=config.to_number,
            from_number=config.from_number,
        )

    async def hangup(self, session_id: str) -> None:
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(
                f"{self._base_url}/calls/1/calls/{session_id}/hangup",
                headers=self._headers(),
            )
            # 404 = call already finished; tolerate it.
            if resp.status_code >= 500:
                resp.raise_for_status()

    async def transfer(self, session_id: str, to_number: str) -> None:
        """Transfer a live call to another phone number.

        Infobip's transfer endpoint takes the same ``endpoint`` shape as
        call creation.
        """
        body = {
            "endpoint": {
                "type": "PHONE",
                "phoneNumber": to_number.lstrip("+"),
            },
        }
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(
                f"{self._base_url}/calls/1/calls/{session_id}/transfer",
                headers=self._headers(),
                json=body,
            )
            resp.raise_for_status()

    # --- Media Streams stubs --------------------------------------------
    # Infobip's real-time audio comes via a "Media-Stream" Calls Configuration:
    # the server-side WS gets raw PCM16 frames (binary WS messages, not JSON
    # envelopes like Twilio/Exotel). Wiring that into our agent needs a
    # separate ``InfobipMediaBridge`` similar to ExotelMediaBridge but with:
    #
    #   - Binary WS receive_bytes / send_bytes instead of JSON
    #   - Sample rate negotiated in the Calls Configuration (8 or 16 kHz)
    #   - No per-frame envelope — frames are just raw audio bytes
    #
    # Until that bridge exists, calling these methods is a programmer error.

    async def stream_audio_in(self, session_id: str) -> AsyncIterator[bytes]:
        raise NotImplementedError(
            "Infobip real-time audio runs through a Media-Stream Calls "
            "Configuration WebSocket — wired by InfobipMediaBridge at the "
            "FastAPI route layer, not at the adapter. Bridge not yet built."
        )
        if False:  # pragma: no cover
            yield b""

    async def stream_audio_out(
        self,
        session_id: str,
        audio_stream: AsyncIterator[bytes],
    ) -> None:
        raise NotImplementedError(
            "Infobip real-time audio runs through a Media-Stream Calls "
            "Configuration WebSocket — wired by InfobipMediaBridge at the "
            "FastAPI route layer, not at the adapter. Bridge not yet built."
        )
