"""Speech-to-speech (realtime, audio-in/audio-out) provider interface.

Unlike the STT/LLM/TTS cascade, an S2S model (e.g. Gemini Live) ingests caller
audio and emits agent audio directly over one duplex session, plus side events:
transcripts (for the transcript/outcome), tool-calls (for dialogue control), and
interruption signals (native barge-in). The bridge consumes ``RealtimeEvent``s
and drives the existing agent/state-machine via those side channels.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class RealtimeTool:
    """A provider-agnostic function declaration the model may call."""
    name: str
    description: str
    parameters: dict[str, Any]  # JSON-Schema-ish: type/properties/required/enum/items


@dataclass
class RealtimeConfig:
    model: str
    voice: Optional[str] = None
    language_code: Optional[str] = None
    system_instruction: str = ""
    tools: list[RealtimeTool] = field(default_factory=list)


@dataclass
class RealtimeEvent:
    """One event from a live S2S session.

    type:
        "audio"             - ``audio`` is agent PCM16 at ``audio_rate``
        "input_transcript"  - ``text`` is (incremental) recognized caller speech
        "output_transcript" - ``text`` is (incremental) agent speech transcript
        "tool_call"         - the model called ``tool_name`` with ``tool_args`` (``tool_id``)
        "interrupted"       - the caller interrupted; flush buffered agent audio
        "turn_complete"     - the agent finished its turn
    """
    type: str
    audio: bytes = b""
    audio_rate: int = 24000
    text: str = ""
    tool_name: str = ""
    tool_args: dict[str, Any] = field(default_factory=dict)
    tool_id: str = ""


class IRealtimeSession(ABC):
    """One open duplex S2S session (one call)."""

    @abstractmethod
    async def send_audio(self, pcm16: bytes) -> None:
        """Feed one chunk of caller PCM16-LE mono @16kHz to the model."""

    @abstractmethod
    async def send_text(self, text: str) -> None:
        """Send a text turn (e.g. a kickoff to make the agent greet first)."""

    @abstractmethod
    def events(self) -> AsyncIterator[RealtimeEvent]:
        """Yield model events until the session closes."""

    @abstractmethod
    async def send_tool_response(self, *, tool_id: str, name: str, response: dict[str, Any]) -> None:
        """Acknowledge a tool-call so the model continues."""

    @abstractmethod
    async def aclose(self) -> None:
        """Close the upstream session."""
