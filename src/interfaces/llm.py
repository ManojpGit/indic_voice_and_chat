"""LLM provider interface (PRD §4.2)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import AsyncIterator, Optional


@dataclass
class LLMMessage:
    role: str  # "system" | "user" | "assistant"
    content: str


@dataclass
class LLMConfig:
    model: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 1024
    response_format: Optional[str] = "json"  # "json" | "text"


@dataclass
class LLMResult:
    text: str
    finish_reason: str
    usage: dict = field(default_factory=dict)
    raw_response: dict = field(default_factory=dict)


class ILLMProvider(ABC):
    @abstractmethod
    async def generate(
        self,
        messages: list[LLMMessage],
        config: LLMConfig,
    ) -> LLMResult:
        """Generate a complete response."""

    @abstractmethod
    async def generate_stream(
        self,
        messages: list[LLMMessage],
        config: LLMConfig,
    ) -> AsyncIterator[str]:
        """Stream response tokens."""
