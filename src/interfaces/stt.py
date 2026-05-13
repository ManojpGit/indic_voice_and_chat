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
