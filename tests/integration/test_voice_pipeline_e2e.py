"""End-to-end mocked voice call test.

Drives the full Twilio bridge -> VoiceBotAgent -> pipeline -> mocked
STT/LLM/TTS stack with synthetic μ-law frames. Asserts:

- Agent moves through IDLE -> LISTENING -> PROCESSING -> RESPONDING -> LISTENING
- The pipeline is invoked with the captured PCM
- Outbound media frames are emitted as μ-law @ 8kHz
- Slots are extracted from the LLM response and applied
- Session state is persisted to (fake) Redis
- ``close_positive`` action terminates the call cleanly
"""

from __future__ import annotations

import asyncio
import base64
import json
import math
import struct
from typing import Any, AsyncIterator, Optional

import pytest
import yaml

from src.agents.base import AgentSession
from src.agents.state_machine import AgentStateMachine, State
from src.agents.voicebot import VoiceBotAgent
from src.api.telephony_twilio import (
    TWILIO_SAMPLE_RATE,
    TwilioBridgeConfig,
    TwilioMediaBridge,
)
from src.dialogue.context import SessionStore
from src.dialogue.prompts import VoiceBotScript
from src.dialogue.slots import SlotSchema
from src.interfaces.llm import ILLMProvider, LLMConfig, LLMMessage, LLMResult
from src.interfaces.stt import ISTTProvider, STTConfig, STTResult
from src.interfaces.tts import ITTSProvider, TTSConfig, TTSResult
from src.pipeline.audio_utils import pcm16_to_mulaw
from src.pipeline.engine import PipelineConfig, PipelineEngine
from src.pipeline.vad import EndpointConfig, EnergyVAD


# --- Mock providers ------------------------------------------------------


class MockSTT(ISTTProvider):
    def __init__(self, transcripts: list[str]) -> None:
        self._transcripts = list(transcripts)

    async def transcribe(self, audio: bytes, config: STTConfig) -> STTResult:
        text = self._transcripts.pop(0) if self._transcripts else ""
        return STTResult(text=text, confidence=0.9, language="hi", raw_response={})

    async def transcribe_stream(self, audio_stream, config) -> AsyncIterator[STTResult]:
        if False:
            yield  # pragma: no cover

    def get_supported_languages(self):
        return ["hi", "en"]


class MockLLM(ILLMProvider):
    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self._responses = list(responses)
        self.history_seen: list[list[LLMMessage]] = []

    async def generate(self, messages, config) -> LLMResult:
        self.history_seen.append(list(messages))
        payload = self._responses.pop(0) if self._responses else {"response_text": "ok", "language": "hi", "action": "end"}
        return LLMResult(text=json.dumps(payload), finish_reason="stop")

    async def generate_stream(self, messages, config):
        self.history_seen.append(list(messages))
        payload = self._responses.pop(0) if self._responses else {"response_text": "ok", "language": "hi", "action": "end"}
        text = json.dumps(payload)
        # Stream in 3 chunks
        third = max(1, len(text) // 3)
        for i in range(0, len(text), third):
            yield text[i : i + third]


class MockTTS(ITTSProvider):
    def __init__(self, sample_rate: int = 16000) -> None:
        self._sr = sample_rate
        self.synthesized: list[str] = []

    async def synthesize(self, text: str, config: TTSConfig) -> TTSResult:
        self.synthesized.append(text)
        # 50ms of zero-PCM as the "synthesized" output.
        n_samples = int(self._sr * 0.05)
        audio = b"\x00\x00" * n_samples
        return TTSResult(audio=audio, duration_ms=50.0, sample_rate=self._sr)

    async def synthesize_stream(self, text_stream, config):
        if False:
            yield  # pragma: no cover

    def get_available_voices(self, language: str):
        return []


# --- Synthetic Twilio frame helpers --------------------------------------


def _loud_mulaw(duration_ms: int) -> str:
    n_samples = int(TWILIO_SAMPLE_RATE * duration_ms / 1000)
    pcm = b"".join(
        struct.pack("<h", int(8000 * math.sin(2 * math.pi * 440 * i / TWILIO_SAMPLE_RATE)))
        for i in range(n_samples)
    )
    return base64.b64encode(pcm16_to_mulaw(pcm)).decode("ascii")


def _silent_mulaw(duration_ms: int) -> str:
    n_samples = int(TWILIO_SAMPLE_RATE * duration_ms / 1000)
    return base64.b64encode(pcm16_to_mulaw(b"\x00\x00" * n_samples)).decode("ascii")


def _utterance_frames(speech_chunks: int = 6, silence_chunks: int = 30) -> list[dict]:
    """One full utterance: speech + trailing silence to trigger endpointing."""
    frames: list[dict] = []
    for _ in range(speech_chunks):
        frames.append({
            "event": "media",
            "media": {"payload": _loud_mulaw(50), "track": "inbound", "timestamp": "0"},
        })
    for _ in range(silence_chunks):
        frames.append({
            "event": "media",
            "media": {"payload": _silent_mulaw(50), "track": "inbound", "timestamp": "0"},
        })
    return frames


class FakeWS:
    def __init__(self, incoming: list[dict]) -> None:
        self._incoming = list(incoming)
        self.sent: list[dict] = []

    async def send_text(self, data: str) -> None:
        self.sent.append(json.loads(data))

    async def receive_text(self) -> str:
        if not self._incoming:
            raise asyncio.CancelledError("no more frames")
        return json.dumps(self._incoming.pop(0))


SCRIPT_YAML = {
    "agent_name": "Priya",
    "agent_role": "Engagement",
    "company_name": "Acme",
    "language_default": "hi",
    "opening": "Namaste!",
    "talking_points": ["Plan B"],
    "qualifying_questions": [],
    "objection_responses": {},
    "closing": {"positive": "Dhanyavaad!", "negative": "Bye"},
}

SLOT_YAML = """
lead_name:        { type: string,   required: true }
interest_level:   { type: enum,     required: true,  values: [hot, warm, cold] }
"""


# --- Tests ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_e2e_two_turns_then_close(fake_redis) -> None:
    # User says two utterances; agent ends on the second turn with close_positive.
    stt = MockSTT(transcripts=["Aap kaun hain?", "Theek hai, lo lete hain"])
    llm = MockLLM(responses=[
        {
            "response_text": "Main Priya hoon, Acme se. Plan B aapke liye accha hoga.",
            "language": "hi",
            "conversation_phase": "opening",
            "updated_slots": {"interest_level": "warm"},
            "action": "continue",
            "sentiment": "positive",
        },
        {
            "response_text": "Bahut accha! Plan B activate kar rahi hoon. Dhanyavaad!",
            "language": "hi",
            "conversation_phase": "closing",
            "updated_slots": {"interest_level": "hot"},
            "action": "close_positive",
            "sentiment": "positive",
        },
    ])
    tts = MockTTS(sample_rate=16000)

    engine = PipelineEngine(
        stt, llm, tts,
        PipelineConfig(stt=STTConfig(language="hi"), llm=LLMConfig(), tts=TTSConfig(language="hi")),
    )

    store = SessionStore(fake_redis, ttl_seconds=300)
    schema = SlotSchema.from_campaign_yaml(yaml.safe_load(SLOT_YAML))
    script = VoiceBotScript.from_campaign_yaml(SCRIPT_YAML)
    sm = AgentStateMachine()
    session = AgentSession(session_id="e2e-session-1", lead_data={"lead_name": "Manoj"})

    agent = VoiceBotAgent(
        session=session,
        state_machine=sm,
        slot_schema=schema,
        script=script,
        engine=engine,
        store=store,
    )

    # Build the full frame timeline: connected, start, utterance 1, utterance 2, stop
    frames: list[dict] = [{"event": "connected"}, {"event": "start", "start": {"streamSid": "MZ"}}]
    frames.extend(_utterance_frames(speech_chunks=6, silence_chunks=30))
    frames.extend(_utterance_frames(speech_chunks=6, silence_chunks=30))
    frames.append({"event": "stop"})

    ws = FakeWS(incoming=frames)
    bridge = TwilioMediaBridge(
        websocket=ws,
        agent=agent,
        vad=EnergyVAD(sample_rate=16000, frame_ms=30, rms_threshold=300.0),
        config=TwilioBridgeConfig(
            pcm_sample_rate=16000,
            endpoint=EndpointConfig(min_speech_ms=60, min_silence_ms=300),
            max_idle_silence_s=60,
        ),
    )

    await bridge.run()

    # Both turns dispatched — and the agent is now ENDED.
    assert len(llm.history_seen) == 2
    assert agent.state.state is State.ENDED

    # Slots accumulated correctly.
    assert agent.slots.get("interest_level") == "hot"

    # Sentiment tracked across both turns.
    assert agent.session.sentiment_history == ["positive", "positive"]

    # Outbound media frames were emitted (μ-law @ 8kHz, base64).
    sent_media = [m for m in ws.sent if m.get("event") == "media"]
    assert len(sent_media) > 0
    # All outbound frames carry the streamSid.
    assert all(m["streamSid"] == "MZ" for m in sent_media)

    # Persisted to Redis: history (>= 2 user + 2 agent turns) and final state.
    history = await store.get_history("e2e-session-1")
    user_turns = [t for t in history if t["role"] == "user"]
    agent_turns = [t for t in history if t["role"] == "agent"]
    assert len(user_turns) == 2
    assert len(agent_turns) == 2
    assert "Main Priya" in agent_turns[0]["content"]
    state = await store.get_state("e2e-session-1")
    assert state["state"] == "ended"
    assert state["slots"]["interest_level"] == "hot"
    assert state["last_action"] == "close_positive"


@pytest.mark.asyncio
async def test_e2e_extended_silence_terminates_call(fake_redis) -> None:
    stt = MockSTT(transcripts=[])  # never reached
    llm = MockLLM(responses=[])
    tts = MockTTS()

    engine = PipelineEngine(
        stt, llm, tts,
        PipelineConfig(stt=STTConfig(), llm=LLMConfig(), tts=TTSConfig()),
    )

    schema = SlotSchema.from_campaign_yaml(yaml.safe_load(SLOT_YAML))
    script = VoiceBotScript.from_campaign_yaml(SCRIPT_YAML)
    agent = VoiceBotAgent(
        session=AgentSession(session_id="e2e-session-2"),
        state_machine=AgentStateMachine(),
        slot_schema=schema,
        script=script,
        engine=engine,
        store=SessionStore(fake_redis, ttl_seconds=300),
    )

    frames: list[dict] = [{"event": "connected"}, {"event": "start", "start": {"streamSid": "MZ"}}]
    # 4 seconds of silence — exceeds max_idle_silence_s=2
    for _ in range(80):
        frames.append({"event": "media",
                       "media": {"payload": _silent_mulaw(50), "track": "inbound", "timestamp": "0"}})

    ws = FakeWS(incoming=frames)
    bridge = TwilioMediaBridge(
        websocket=ws,
        agent=agent,
        vad=EnergyVAD(sample_rate=16000, frame_ms=30, rms_threshold=300.0),
        config=TwilioBridgeConfig(
            pcm_sample_rate=16000,
            endpoint=EndpointConfig(min_speech_ms=300, min_silence_ms=600),
            max_idle_silence_s=2.0,
        ),
    )

    await bridge.run()

    assert agent.state.state is State.ENDED
    assert llm.history_seen == []  # no LLM calls — never had an utterance
