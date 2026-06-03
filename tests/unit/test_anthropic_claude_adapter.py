from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.interfaces.llm import LLMConfig, LLMMessage
from src.providers.llm.anthropic_claude import AnthropicClaudeAdapter


def _message(text: str, stop_reason: str = "end_turn",
             input_tokens: int = 10, output_tokens: int = 5) -> SimpleNamespace:
    """Fake anthropic Message: content is a list of typed blocks."""
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)],
        stop_reason=stop_reason,
        usage=SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens),
    )


class _FakeStreamCM:
    """Mimics the SDK's ``client.messages.stream(...)`` async context manager."""

    def __init__(self, chunks: list[str]) -> None:
        self._chunks = chunks

    async def __aenter__(self):
        chunks = self._chunks

        async def _text_stream():
            for c in chunks:
                yield c

        return SimpleNamespace(text_stream=_text_stream())

    async def __aexit__(self, *exc: Any) -> bool:
        return False


def _make_client(*, create_return: Any = None,
                 stream_chunks: list[str] | None = None) -> SimpleNamespace:
    create = AsyncMock(return_value=create_return) if create_return else AsyncMock()
    if stream_chunks is not None:
        stream = MagicMock(return_value=_FakeStreamCM(stream_chunks))
    else:
        stream = MagicMock()
    return SimpleNamespace(messages=SimpleNamespace(create=create, stream=stream))


# --- generate() ---------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_prepends_prefill_and_returns_usage() -> None:
    # Model generated everything after the prefilled '{'.
    client = _make_client(create_return=_message('"response_text": "hi"}'))
    adapter = AnthropicClaudeAdapter({"client": client, "model": "claude-haiku-4-5"})

    result = await adapter.generate(
        [LLMMessage(role="user", content="hi")],
        LLMConfig(model="claude-haiku-4-5", temperature=0.4, max_tokens=128),
    )
    # The prefilled '{' is glued back on so the envelope is valid JSON.
    assert result.text == '{"response_text": "hi"}'
    assert result.finish_reason == "stop"
    assert result.usage == {"prompt_tokens": 10, "completion_tokens": 5}

    kwargs = client.messages.create.await_args.kwargs
    assert kwargs["model"] == "claude-haiku-4-5"
    assert kwargs["max_tokens"] == 128
    assert kwargs["temperature"] == 0.4
    # JSON mode forces a '{' assistant prefill as the last message.
    assert kwargs["messages"][-1] == {"role": "assistant", "content": "{"}


@pytest.mark.asyncio
async def test_generate_maps_system_role_into_system_param() -> None:
    client = _make_client(create_return=_message("ok}"))
    adapter = AnthropicClaudeAdapter({"client": client})
    await adapter.generate(
        [
            LLMMessage(role="system", content="be terse"),
            LLMMessage(role="system", content="reply in Hindi"),
            LLMMessage(role="user", content="hi"),
            LLMMessage(role="assistant", content="namaste"),
            LLMMessage(role="user", content="status?"),
        ],
        LLMConfig(),
    )
    kwargs = client.messages.create.await_args.kwargs
    assert kwargs["system"] == "be terse\n\nreply in Hindi"
    conv = kwargs["messages"]
    # System messages do not appear in the conversation list.
    assert all(m["role"] in ("user", "assistant") for m in conv)
    assert conv[0] == {"role": "user", "content": "hi"}
    assert conv[1] == {"role": "assistant", "content": "namaste"}


@pytest.mark.asyncio
async def test_text_format_does_not_prefill() -> None:
    client = _make_client(create_return=_message("plain text"))
    adapter = AnthropicClaudeAdapter({"client": client})
    result = await adapter.generate(
        [LLMMessage(role="user", content="hi")], LLMConfig(response_format="text")
    )
    assert result.text == "plain text"  # no leading '{'
    conv = client.messages.create.await_args.kwargs["messages"]
    assert conv[-1]["role"] == "user"  # no assistant prefill appended


@pytest.mark.asyncio
async def test_opus_48_omits_temperature_and_prefill() -> None:
    client = _make_client(create_return=_message('{"response_text": "x"}'))
    adapter = AnthropicClaudeAdapter({"client": client, "model": "claude-opus-4-8"})
    await adapter.generate(
        [LLMMessage(role="user", content="hi")],
        LLMConfig(model="claude-opus-4-8", response_format="json"),
    )
    kwargs = client.messages.create.await_args.kwargs
    assert "temperature" not in kwargs  # 400s on opus-4-8
    assert kwargs["messages"][-1]["role"] == "user"  # prefill 400s on opus-4-8


@pytest.mark.asyncio
async def test_finish_reason_max_tokens() -> None:
    client = _make_client(create_return=_message("trunc", stop_reason="max_tokens"))
    adapter = AnthropicClaudeAdapter({"client": client})
    result = await adapter.generate([LLMMessage(role="user", content="hi")], LLMConfig())
    assert result.finish_reason == "length"


# --- generate_stream() --------------------------------------------------


@pytest.mark.asyncio
async def test_stream_emits_prefill_then_chunks() -> None:
    client = _make_client(stream_chunks=['"response_text": "Hel', 'lo"}'])
    adapter = AnthropicClaudeAdapter({"client": client, "model": "claude-haiku-4-5"})
    tokens: list[str] = []
    async for t in adapter.generate_stream(
        [LLMMessage(role="user", content="hi")], LLMConfig(response_format="json")
    ):
        tokens.append(t)
    # First token is the prefilled brace, then the model's deltas.
    assert tokens == ["{", '"response_text": "Hel', 'lo"}']
    assert "".join(tokens) == '{"response_text": "Hello"}'


@pytest.mark.asyncio
async def test_stream_text_mode_no_prefill() -> None:
    client = _make_client(stream_chunks=["hello ", "world"])
    adapter = AnthropicClaudeAdapter({"client": client})
    tokens = []
    async for t in adapter.generate_stream(
        [LLMMessage(role="user", content="hi")], LLMConfig(response_format="text")
    ):
        tokens.append(t)
    assert tokens == ["hello ", "world"]


# --- Construction ------------------------------------------------------


@pytest.mark.asyncio
async def test_constructor_rejects_missing_key(monkeypatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(ValueError):
        AnthropicClaudeAdapter({})


# --- helpers ------------------------------------------------------------


def test_extract_text_joins_text_blocks() -> None:
    msg = SimpleNamespace(content=[
        SimpleNamespace(type="text", text="a"),
        SimpleNamespace(type="thinking", thinking="ignore me"),
        SimpleNamespace(type="text", text="b"),
    ])
    assert AnthropicClaudeAdapter._extract_text(msg) == "ab"
