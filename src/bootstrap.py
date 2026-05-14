"""Production wiring: per-call bridge factory + provider construction.

Lives in its own module (rather than ``main.py``) so individual pieces are
unit-testable without spinning up the FastAPI lifespan.

The bridge factory is the moment of truth — it's what turns a tenant's
declared provider preferences into a live conversation. Flow:

    inbound WS connect
        -> twilio_stream() resolves tenant from ?tenant= query param
        -> calls registered bridge_factory(websocket, tenant)
        -> builds STT/LLM/TTS/scheduler/etc. for that tenant via the
           cached TenantProviders registry
        -> assembles a VoiceBotAgent with a minimal demo script
        -> wraps in a TwilioMediaBridge that:
             * calls agent.start() then agent.play_opening(sink)
             * pumps Twilio media frames into agent.handle_turn()
             * sends agent TTS audio back as μ-law frames
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from fastapi import WebSocket

from src.agents.base import AgentSession
from src.agents.state_machine import AgentStateMachine
from src.agents.voicebot import VoiceBotAgent
from src.api.telephony_twilio import TwilioBridgeConfig, TwilioMediaBridge
from src.auth.context import TenantContext
from src.auth.registry import TenantProviders
from src.dialogue.context import SessionStore
from src.dialogue.prompts import VoiceBotScript
from src.dialogue.slots import SlotSchema
from src.interfaces.llm import LLMConfig
from src.interfaces.stt import STTConfig
from src.interfaces.tts import TTSConfig
from src.pipeline.engine import PipelineConfig, PipelineEngine
from src.pipeline.vad import EndpointConfig, EnergyVAD
from src.providers import (
    get_llm_provider,
    get_stt_provider,
    get_telephony_provider,
    get_tts_provider,
    get_vector_store,
)

log = logging.getLogger(__name__)


# --- Demo script -------------------------------------------------------


DEFAULT_DEMO_SCRIPT = VoiceBotScript(
    agent_name="Priya",
    agent_role="Customer Engagement Specialist",
    company_name="Vox Demo",
    language_default="hi-IN",
    opening=(
        "Namaste! Main Priya bol rahi hoon Vox Demo se. "
        "Aapse ek choti si baat karni thi — kya aapke paas do minute hain?"
    ),
    talking_points=[
        "Vox Demo ek end-to-end AI voice agent platform hai.",
    ],
    qualifying_questions=["Aap abhi kya use kar rahe hain customer calls ke liye?"],
    objection_responses={
        "is_ai": "Haan, main ek AI assistant hoon — Vox Demo ki taraf se.",
        "busy": "Bilkul, samajh sakti hoon. Kya main baad mein call karun?",
    },
    closing={
        "positive": "Bahut accha! Dhanyavaad aapke time ke liye.",
        "negative": "Koi baat nahi. Aapka din shubh ho!",
    },
)


# --- Per-tenant runtime builders ---------------------------------------


def build_provider_registry(
    global_defaults: dict, base_vector_path: Path = Path("data/faiss"),
) -> TenantProviders:
    """One ``TenantProviders`` per process; caches per-tenant clients."""
    return TenantProviders(
        global_defaults=global_defaults,
        stt_factory=get_stt_provider,
        llm_factory=get_llm_provider,
        tts_factory=get_tts_provider,
        telephony_factory=get_telephony_provider,
        vector_store_factory=get_vector_store,
        base_vector_path=base_vector_path,
    )


# --- The bridge factory ------------------------------------------------


@dataclass
class _CallSpec:
    """Resolved per-call wiring."""

    tenant: TenantContext
    session_id: str
    lead_data: dict


def make_bridge_factory(
    providers: TenantProviders,
    session_store: SessionStore | None = None,
    bridge_config: TwilioBridgeConfig | None = None,
    script: VoiceBotScript = DEFAULT_DEMO_SCRIPT,
) -> Callable[[WebSocket, TenantContext], TwilioMediaBridge]:
    """Return a callable suitable for ``set_bridge_factory(...)``.

    The returned factory closes over the shared registries so every
    inbound call lands on the right tenant-scoped provider clients
    without rebuilding anything.
    """
    cfg = bridge_config or TwilioBridgeConfig()

    def factory(websocket: WebSocket, tenant: TenantContext) -> TwilioMediaBridge:
        # Build a fresh agent per call; provider clients are cached on the
        # registry so we don't pay reconstruction cost.
        stt = providers.get_stt(tenant)
        llm = providers.get_llm(tenant)
        tts = providers.get_tts(tenant)

        # Tenant-namespaced Redis session store (one per tenant; the same
        # instance is fine across calls since the keys carry session_id).
        store: SessionStore | None = None
        if session_store is not None:
            store = SessionStore(
                redis=session_store.redis,
                ttl_seconds=session_store.ttl,
                tenant_id=tenant.id,
            )

        # Use Sarvam/etc. defaults for the per-call configs — providers
        # have already been built with tenant overrides applied.
        pipeline_cfg = PipelineConfig(
            stt=STTConfig(language=tenant.settings.pipeline.stt.language or "hi-IN"),
            llm=LLMConfig(
                temperature=tenant.settings.pipeline.llm.temperature or 0.5,
                max_tokens=tenant.settings.pipeline.llm.max_tokens or 256,
                response_format=tenant.settings.pipeline.llm.response_format or "json",
            ),
            tts=TTSConfig(
                language=tenant.settings.pipeline.tts.language or "hi-IN",
                voice_id=tenant.settings.pipeline.tts.voice_id,
                sample_rate=16000,
            ),
        )
        engine = PipelineEngine(stt, llm, tts, pipeline_cfg)

        import uuid

        session_id = f"call_{uuid.uuid4().hex[:12]}"
        session = AgentSession(session_id=session_id)
        sm = AgentStateMachine()
        agent = VoiceBotAgent(
            session=session,
            state_machine=sm,
            slot_schema=SlotSchema(),
            script=script,
            engine=engine,
            store=store,
        )

        log.info(
            "bridge factory built call",
            extra={"tenant": tenant.slug, "session_id": session_id},
        )

        return _AgentBridge(
            websocket=websocket,
            agent=agent,
            vad=EnergyVAD(sample_rate=16000, frame_ms=30, rms_threshold=300.0),
            config=cfg,
        )

    return factory


class _AgentBridge(TwilioMediaBridge):
    """Subclass of TwilioMediaBridge that plays an opening line on connect.

    Crucial ordering: Twilio sends ``connected`` then ``start`` events on
    the WS before any media. ``_send_pcm`` needs ``self._stream_sid``,
    which is only populated when we process the ``start`` event. So
    ``play_opening`` MUST run after the start event, not before — otherwise
    the opening audio is silently dropped (``_send_pcm`` returns early
    when ``_stream_sid is None``).
    """

    async def run(self) -> None:
        import json

        await self._agent.start()
        opening_played = False
        try:
            while not self._stopped.is_set():
                raw = await self._ws.receive_text()
                msg = json.loads(raw)
                event = msg.get("event")
                if event == "connected":
                    continue
                if event == "start":
                    self._stream_sid = (
                        msg.get("start", {}).get("streamSid") or msg.get("streamSid")
                    )
                    log.info("twilio stream started", extra={"streamSid": self._stream_sid})
                    # NOW we can play the opening — Twilio is ready to receive media.
                    if not opening_played:
                        opening_played = True
                        await self._agent.play_opening(self._send_pcm)  # type: ignore[arg-type]
                        log.info("agent opening played", extra={"streamSid": self._stream_sid})
                elif event == "media":
                    await self._on_media_frame(msg["media"])
                elif event == "stop":
                    log.info("twilio stream stopped")
                    break
        finally:
            await self._agent.handle_hangup()
