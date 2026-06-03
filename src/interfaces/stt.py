"""Speech-to-Text provider interface (PRD §4.1)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import AsyncIterator, Optional


@dataclass
class STTResult:
    text: str
    confidence: float
    language: Optional[str] = None
    word_timestamps: Optional[list[dict]] = None
    raw_response: dict = field(default_factory=dict)


@dataclass
class STTConfig:
    language: Optional[str] = None
    model: Optional[str] = None
    sample_rate: int = 16000
    enable_timestamps: bool = False


class ISTTProvider(ABC):
    @abstractmethod
    async def transcribe(self, audio: bytes, config: STTConfig) -> STTResult:
        """Transcribe a complete audio segment."""

    @abstractmethod
    async def transcribe_stream(
        self,
        audio_stream: AsyncIterator[bytes],
        config: STTConfig,
    ) -> AsyncIterator[STTResult]:
        """Stream transcription results as audio arrives."""

    @abstractmethod
    def get_supported_languages(self) -> list[str]:
        """Return list of supported language codes."""


@dataclass
class STTStreamEvent:
    """One event from a live STT session.

    type:
        "interim"  - a partial, non-final transcript (may change)
        "final"    - a finalized transcript segment (won't change)
        "endpoint" - end of utterance; ``text`` is the full utterance transcript
    """

    type: str
    text: str
    confidence: float = 1.0
    language: Optional[str] = None


class ISTTStreamSession(ABC):
    @abstractmethod
    async def send(self, pcm16: bytes) -> None:
        """Feed one chunk of raw PCM16-LE mono audio to the recognizer."""

    @abstractmethod
    def events(self) -> AsyncIterator[STTStreamEvent]:
        """Yield recognizer events until the session is closed."""

    @abstractmethod
    async def aclose(self) -> None:
        """Flush, close the upstream connection, and cancel background tasks."""


class IStreamingSTTProvider(ABC):
    @abstractmethod
    async def open_stream(self, config: STTConfig) -> ISTTStreamSession:
        """Open a live streaming session for one utterance stream."""
