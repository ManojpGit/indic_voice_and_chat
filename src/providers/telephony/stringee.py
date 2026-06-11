"""Stringee telephony adapter.

Stringee is a Vietnam-based CPaaS with carrier interconnects across
Southeast Asia and India. Their authentication model is different from
Twilio/Exotel: instead of HTTP Basic Auth with an SID + token, Stringee
issues short-lived JWT access tokens signed with your API key + secret.

REST surface implemented:
- ``initiate_call``  POST /v1/call2/callout (or /v1/call/click2call) — outbound dial
- ``hangup``         POST /v1/call2/{sid}/hangup
- ``transfer``       not natively supported by their public REST API;
                     would need their SDK or an SCC (Stringee Call Control)
                     script. Implemented as a documented NotImplementedError.

Audio Streaming:
Stringee's audio streaming runs through their proprietary client SDKs
(JS / Android / iOS / Flutter), not a documented server-side WebSocket
protocol like Twilio Media Streams. Wiring our agent into Stringee
audio therefore requires a different pattern than the Twilio/Exotel
bridges. We stub ``stream_audio_in/out`` with a clear error message so
the gap is loud and isolated. See the docstrings in
``StringeeAdapter.stream_audio_in`` for the recommended path forward.

Endpoint reference: https://developer.stringee.com/docs/api-reference
"""

from __future__ import annotations

import os
import time
from collections.abc import AsyncIterator
from typing import Any

import httpx

from src.interfaces.telephony import (
    CallConfig,
    CallSession,
    ITelephonyProvider,
)

STRINGEE_BASE_URL = "https://api.stringee.com"


def _bare_number(number: str) -> str:
    """Stringee requires bare digits — no leading '+' (E.164 with '+' is rejected
    as r:10 FROM/TO_NUMBER_INVALID_FORMAT). Strip a leading '+' and surrounding
    whitespace so callers/config may keep the '+E.164' form."""
    return (number or "").strip().lstrip("+")

# Stringee call event → our vocabulary.
_STATUS_MAP = {
    "STARTING":  "ringing",
    "RINGING":   "ringing",
    "ANSWERED":  "answered",
    "ENDED":     "answered",
    "BUSY":      "busy",
    "NO_ANSWER": "no_answer",
    "FAILED":    "failed",
}


class StringeeAdapter(ITelephonyProvider):
    def __init__(self, config: dict[str, Any]) -> None:
        api_key_sid = config.get("api_key_sid") or os.environ.get("STRINGEE_API_KEY_SID")
        api_key_secret = config.get("api_key_secret") or os.environ.get("STRINGEE_API_KEY_SECRET")
        if not (api_key_sid and api_key_secret):
            raise ValueError(
                "StringeeAdapter requires api_key_sid + api_key_secret (or "
                "STRINGEE_API_KEY_SID / STRINGEE_API_KEY_SECRET env vars)"
            )
        self._api_key_sid = api_key_sid
        self._api_key_secret = api_key_secret
        # Stringee keys are region-scoped: a key issued in (e.g.) asia-2 is
        # rejected by the global api.stringee.com with "keySid invalid". Allow
        # the regional API base to come from config or the STRINGEE_BASE_URL env
        # var (e.g. https://asia-2.api.stringee.com).
        self._base_url = (
            config.get("base_url")
            or os.environ.get("STRINGEE_BASE_URL")
            or STRINGEE_BASE_URL
        ).rstrip("/")
        self._timeout = config.get("timeout", 30.0)
        # Tests can inject a pre-built bearer; otherwise we mint a fresh JWT.
        self._token_override: str | None = config.get("access_token")

    def _make_access_token(self, ttl_seconds: int = 3600) -> str:
        """Mint a Stringee access JWT signed with HS256 (api_key_secret).

        Stringee requires the JWT *header* to carry ``cty: stringee-api;v=1``
        in addition to the standard ``alg``/``typ``. Without it the REST API
        rejects the token with HTTP 403 ``{"r": 5, "message": "keySid
        invalid"}`` even when the keySid and signature are correct.
        """
        import jwt  # PyJWT — already pulled in transitively by Twilio SDK

        now = int(time.time())
        payload = {
            "jti": f"{self._api_key_sid}-{now}",
            "iss": self._api_key_sid,
            "exp": now + ttl_seconds,
            "rest_api": True,
        }
        return jwt.encode(
            payload,
            self._api_key_secret,
            algorithm="HS256",
            headers={"cty": "stringee-api;v=1"},
        )

    def _headers(self) -> dict[str, str]:
        token = self._token_override or self._make_access_token()
        return {
            "X-STRINGEE-AUTH": token,
            "Content-Type": "application/json",
        }

    async def initiate_call(self, config: CallConfig) -> CallSession:
        """Place an outbound call via the Stringee REST API.

        Stringee's ``callout`` endpoint takes a JSON ``answer_url`` that
        returns an SCC (Stringee Call Control) script when the destination
        picks up — equivalent to TwiML.
        """
        # Stringee rejects E.164 with a leading '+' (r:10 FROM/TO_NUMBER_INVALID_
        # FORMAT) — numbers must be BARE digits (e.g. 918204268005). Normalize
        # here so callers/config can keep the '+E.164' form.
        from_number = _bare_number(config.from_number)
        to_number = _bare_number(config.to_number)
        # NOTE: do NOT send ``actions`` (not even an empty list) alongside
        # ``answer_url`` — Stringee treats a present ``actions`` as the SCCO and
        # never GETs the answer_url, so the IVR is never invoked.
        body: dict[str, Any] = {
            # ``from`` is the originating Stringee-side identity for the outbound
            # callout (per Stringee support) -> type "internal"; ``to`` is the real
            # PSTN destination -> "external".
            "from": {"type": "internal", "number": from_number, "alias": from_number},
            "to": [{"type": "external", "number": to_number, "alias": to_number}],
            "answer_url": config.webhook_url,
        }
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(
                f"{self._base_url}/v1/call2/callout",
                headers=self._headers(),
                json=body,
            )
            resp.raise_for_status()
            payload = resp.json()
        call_id = payload.get("call_id") or payload.get("callId") or ""
        raw_status = (payload.get("status") or "STARTING").upper()
        return CallSession(
            session_id=str(call_id),
            status=_STATUS_MAP.get(raw_status, raw_status.lower()),
            to_number=config.to_number,
            from_number=config.from_number,
        )

    async def hangup(self, session_id: str) -> None:
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(
                f"{self._base_url}/v1/call2/{session_id}/hangup",
                headers=self._headers(),
            )
            if resp.status_code >= 500:
                resp.raise_for_status()

    async def transfer(self, session_id: str, to_number: str) -> None:
        """Stringee doesn't expose a direct REST transfer — requires an
        SCC script update or client-side SDK action. Surface this gap
        explicitly rather than silently no-op."""
        raise NotImplementedError(
            "Stringee transfer is not supported via the public REST API. "
            "Implement via an SCC script update or the client SDK; see "
            "https://developer.stringee.com/docs/voice-api/call-control"
        )

    # --- Media Streams stubs --------------------------------------------
    # Stringee does not expose a server-side bidirectional media WebSocket
    # in the Twilio Media Streams / Exotel Voicebot Streaming style. Every
    # supported real-time audio path requires either their client SDK
    # (JS / Android / iOS / Flutter) acting as a participant, or an SCC
    # script orchestrating a conference into which our agent joins as
    # another participant. The integration shape therefore lives outside
    # this adapter; we surface the gap loudly here.
    #
    # Recommended integration path (when wired):
    #
    #   1. Provision a "bot" Stringee user account for this tenant.
    #   2. On call answer, the SCC ``answer_url`` script creates a
    #      ``conference`` and dials our bot user into it
    #      (`<connect><conference id="..."/></connect>`).
    #   3. A separate Node/Python service runs the Stringee JS/Web SDK
    #      headlessly under the bot identity, exposing local audio in/out
    #      via a WebRTC data channel or a local WS bridge.
    #   4. That bridge forwards PCM frames to our agent over the network
    #      using the same shape ExotelMediaBridge already consumes.
    #
    # Until that bridge is built, calling these methods is a programmer
    # error — the adapter raises a NotImplementedError that names the
    # missing piece so the caller can either pick a different telephony
    # provider or build the conference bridge.
    #
    # See ``docs/stringee-streaming.md`` for the long-form recipe.

    async def stream_audio_in(self, session_id: str) -> AsyncIterator[bytes]:
        raise NotImplementedError(
            "Stringee server-side audio streaming is not supported; route this "
            "call through a Stringee conference + bot-user client SDK bridge. "
            "See docs/stringee-streaming.md for the integration recipe."
        )
        if False:  # pragma: no cover - keep the generator type honest
            yield b""

    async def stream_audio_out(
        self,
        session_id: str,
        audio_stream: AsyncIterator[bytes],
    ) -> None:
        raise NotImplementedError(
            "Stringee server-side audio streaming is not supported; route this "
            "call through a Stringee conference + bot-user client SDK bridge. "
            "See docs/stringee-streaming.md for the integration recipe."
        )
