# src/api/dev_console.py
"""Dev-only browser voice console (gated by VOX_DEV_CONSOLE=1).

Serves a self-contained page at ``GET /dev/voice`` and runs a
``BrowserVoiceBridge`` at ``WS /api/v1/dev/voice``. Reuses the tenant's
provider stack exactly like the telephony bridges; intended for local
dialogue-management iteration with no telephony cost.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Callable, Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from src.agents.base import AgentSession
from src.agents.state_machine import AgentStateMachine
from src.agents.voicebot import VoiceBotAgent
from src.api.browser_bridge import BrowserBridgeConfig, BrowserVoiceBridge
from src.api.gemini_live_bridge import RECORD_TURN_SIGNAL, GeminiLiveBridge
from src.auth.context import TenantContext
from src.auth.registry import TenantProviders
from src.bootstrap import DEFAULT_DEMO_SCRIPT
from src.dialogue.prompts import VoiceBotScript, build_s2s_system_instruction
from src.dialogue.slots import SlotSchema
from src.interfaces.realtime import RealtimeConfig
from src.providers.realtime.gemini_live import GeminiLiveSession
from src.interfaces.llm import LLMConfig
from src.interfaces.stt import STTConfig
from src.interfaces.tts import TTSConfig
from src.pipeline.engine import PipelineConfig, PipelineEngine
from src.pipeline.vad import EnergyVAD, SileroVAD
from src.providers import get_streaming_stt_provider

log = logging.getLogger(__name__)

_STATIC = Path(__file__).resolve().parents[2] / "static"

# WS path lives under the /api/v1 router; the page route is top-level.
ws_router = APIRouter(prefix="/dev", tags=["dev-console"])   # mounted under /api/v1
dev_router = APIRouter(tags=["dev-console"])                  # mounted at app root

# Factory: (websocket, tenant) -> BrowserVoiceBridge. Set during lifespan.
BrowserBridgeFactory = Callable[[WebSocket, TenantContext], BrowserVoiceBridge]
_browser_bridge_factory: Optional[BrowserBridgeFactory] = None


def dev_console_enabled() -> bool:
    return os.environ.get("VOX_DEV_CONSOLE", "") == "1"


def set_browser_bridge_factory(factory: Optional[BrowserBridgeFactory]) -> None:
    global _browser_bridge_factory
    _browser_bridge_factory = factory


# Factory: (websocket, tenant) -> GeminiLiveBridge (the S2S path). Set during lifespan.
LiveBridgeFactory = Callable[[WebSocket, TenantContext], GeminiLiveBridge]
_live_bridge_factory: Optional[LiveBridgeFactory] = None


def set_live_bridge_factory(factory: Optional[LiveBridgeFactory]) -> None:
    global _live_bridge_factory
    _live_bridge_factory = factory


@dev_router.get("/dev/voice")
async def dev_voice_page() -> FileResponse:
    return FileResponse(_STATIC / "dev_console.html", media_type="text/html")


@ws_router.websocket("/voice")
async def dev_voice_ws(websocket: WebSocket) -> None:
    from src.auth.middleware import tenant_from_slug

    await websocket.accept()
    if _browser_bridge_factory is None:
        await websocket.close(code=1011, reason="browser bridge factory unset")
        return
    try:
        tenant = await tenant_from_slug(
            websocket.query_params.get("tenant", "dev")
        )
    except Exception as e:  # noqa: BLE001
        log.warning("dev console tenant resolution failed: %s", e)
        await websocket.close(code=1008, reason="unknown tenant")
        return

    bridge = _browser_bridge_factory(websocket, tenant)
    try:
        await bridge.run()
    except WebSocketDisconnect:
        log.info("dev console client disconnected", extra={"tenant": tenant.slug})
    except Exception:  # noqa: BLE001
        log.exception("dev console bridge crashed", extra={"tenant": tenant.slug})
    finally:
        try:
            await websocket.close()
        except Exception:  # noqa: BLE001
            pass


@ws_router.websocket("/voice-live")
async def dev_voice_live_ws(websocket: WebSocket) -> None:
    """Speech-to-speech (Gemini Live) path. Same client; different bridge."""
    from src.auth.middleware import tenant_from_slug

    await websocket.accept()
    if _live_bridge_factory is None:
        await websocket.close(code=1011, reason="live bridge factory unset")
        return
    try:
        tenant = await tenant_from_slug(websocket.query_params.get("tenant", "dev"))
    except Exception as e:  # noqa: BLE001
        log.warning("dev console (s2s) tenant resolution failed: %s", e)
        await websocket.close(code=1008, reason="unknown tenant")
        return
    try:
        bridge = _live_bridge_factory(websocket, tenant)
    except Exception as e:  # noqa: BLE001 - e.g. tenant has no realtime config
        log.warning("dev console (s2s) bridge build failed: %s", e)
        await websocket.close(code=1011, reason="s2s not configured for tenant")
        return
    try:
        await bridge.run()
    except WebSocketDisconnect:
        log.info("dev console (s2s) client disconnected", extra={"tenant": tenant.slug})
    except Exception:  # noqa: BLE001
        log.exception("dev console (s2s) bridge crashed", extra={"tenant": tenant.slug})
    finally:
        try:
            await websocket.close()
        except Exception:  # noqa: BLE001
            pass


def _build_browser_vad():
    """VAD for the dev console: prefer Silero (robust speech/noise discrimination
    so turns end cleanly on speakers); fall back to EnergyVAD if onnxruntime or
    the model isn't available. Silero needs 32 ms / 512-sample frames at 16 kHz.
    """
    try:
        vad = SileroVAD(sample_rate=16000, frame_ms=32, threshold=0.5)
        vad._ensure_model()  # load now so a failure falls back here, not mid-call
        log.info("dev console using SileroVAD")
        return vad
    except Exception as e:  # noqa: BLE001
        log.warning("SileroVAD unavailable (%s); using EnergyVAD", e)
        return EnergyVAD(sample_rate=16000, frame_ms=30, rms_threshold=300.0)


def _build_stream_provider(tenant: TenantContext):
    """Build a streaming-STT provider from pipeline.stt_streaming, or None.

    Returns None when no streaming config is present (batch behaviour) or when
    the provider can't be constructed (e.g. missing key) — the bridge then
    falls back to batch Groq, so this never blocks a call.
    """
    cfg = getattr(tenant.settings.pipeline, "stt_streaming", None)
    if cfg is None or not getattr(cfg, "provider", None):
        return None
    try:
        merged = {
            "provider": cfg.provider,
            "model": cfg.model,
            "language": cfg.language,
            "endpointing": cfg.endpointing,
            "utterance_end_ms": cfg.utterance_end_ms,
            "api_key": tenant.secret(cfg.api_key_env) if cfg.api_key_env else None,
        }
        return get_streaming_stt_provider(merged)
    except Exception as e:  # noqa: BLE001 - never block a call on streaming setup
        log.warning("streaming STT provider unavailable (%s); using batch", e)
        return None


def make_browser_bridge_factory(
    providers: TenantProviders,
    script: VoiceBotScript = DEFAULT_DEMO_SCRIPT,
    slots: SlotSchema = SlotSchema(),
) -> BrowserBridgeFactory:
    """Build a BrowserVoiceBridge per connection, wired to the tenant stack.

    Mirrors ``src.bootstrap.make_bridge_factory`` but returns a browser bridge.
    """

    def factory(websocket: WebSocket, tenant: TenantContext) -> BrowserVoiceBridge:
        import uuid

        stt = providers.get_stt(tenant)
        llm = providers.get_llm(tenant)
        tts = providers.get_tts(tenant)
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
        session_id = f"web_{uuid.uuid4().hex[:12]}"
        # The dev console has no CRM lead, so let the page supply a test lead
        # name via the WS query string (?lead_name=...). This feeds the spoken
        # opening, the rendered opening in the prompt, and "Known lead data".
        query_params = getattr(websocket, "query_params", {}) or {}
        lead_name = (query_params.get("lead_name") or "").strip()
        lead_data = {"lead_name": lead_name, "name": lead_name} if lead_name else {}
        agent = VoiceBotAgent(
            session=AgentSession(session_id=session_id, lead_data=lead_data),
            state_machine=AgentStateMachine(),
            slot_schema=slots,
            script=script,
            engine=engine,
            store=None,
        )
        log.info("dev console built call", extra={"tenant": tenant.slug, "session_id": session_id})
        return BrowserVoiceBridge(
            websocket=websocket,
            agent=agent,
            vad=_build_browser_vad(),
            config=BrowserBridgeConfig(),
            stream_provider=_build_stream_provider(tenant),
            llm=llm,
            tenant_timezone=getattr(tenant.settings, "timezone", "Asia/Kolkata"),
        )

    return factory


def make_live_bridge_factory(
    providers: TenantProviders,
    script: VoiceBotScript = DEFAULT_DEMO_SCRIPT,
    slots: SlotSchema = SlotSchema(),
) -> LiveBridgeFactory:
    """Build a GeminiLiveBridge (S2S) per connection from pipeline.realtime."""

    def factory(websocket: WebSocket, tenant: TenantContext) -> GeminiLiveBridge:
        import uuid

        rt = getattr(tenant.settings.pipeline, "realtime", None)
        if rt is None or not getattr(rt, "provider", None):
            raise RuntimeError("tenant has no pipeline.realtime config for S2S")

        llm = providers.get_llm(tenant)
        # The agent is the same; only the bridge differs. The engine is required by
        # the constructor (the Live path doesn't synthesize via it).
        engine = PipelineEngine(
            providers.get_stt(tenant), llm, providers.get_tts(tenant),
            PipelineConfig(stt=STTConfig(), llm=LLMConfig(), tts=TTSConfig(sample_rate=16000)),
        )
        qp = getattr(websocket, "query_params", {}) or {}
        lead_name = (qp.get("lead_name") or "").strip()
        lead_data = {"lead_name": lead_name, "name": lead_name} if lead_name else {}
        session_id = f"live_{uuid.uuid4().hex[:12]}"
        agent = VoiceBotAgent(
            session=AgentSession(session_id=session_id, lead_data=lead_data),
            state_machine=AgentStateMachine(), slot_schema=slots, script=script,
            engine=engine, store=None,
        )

        # Voice: ?voice= overrides the config default (validated against allowed_voices).
        voice = (qp.get("voice") or "").strip() or rt.voice
        if rt.allowed_voices and voice not in rt.allowed_voices:
            voice = rt.voice
        key = tenant.secret(rt.api_key_env) if rt.api_key_env else None
        config = RealtimeConfig(
            model=rt.model, voice=voice, language_code=rt.language_code,
            system_instruction=build_s2s_system_instruction(script, slots, lead_data),
            tools=[RECORD_TURN_SIGNAL],
        )

        async def connect(cfg: RealtimeConfig):
            return await GeminiLiveSession.connect(cfg, api_key=key)

        log.info("dev console built S2S call", extra={
            "tenant": tenant.slug, "session_id": session_id, "voice": voice, "model": rt.model})
        return GeminiLiveBridge(
            websocket=websocket, agent=agent, config=config, connect_session=connect,
            llm=llm, tenant_timezone=getattr(tenant.settings, "timezone", "Asia/Kolkata"))

    return factory
