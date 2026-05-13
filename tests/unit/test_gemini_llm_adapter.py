from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

from src.interfaces.llm import LLMConfig, LLMMessage
from src.providers.llm.gemini import GeminiLLMAdapter


def _response(text: str, finish_reason: str = "STOP",
              prompt_tokens: int = 10, completion_tokens: int = 5) -> SimpleNamespace:
    """Build a fake response shape mirroring google.genai's GenerateContentResponse."""
    candidate = SimpleNamespace(
        content=SimpleNamespace(parts=[SimpleNamespace(text=text)]),
        finish_reason=SimpleNamespace(name=finish_reason),
    )
    return SimpleNamespace(
        text=text,
        candidates=[candidate],
        usage_metadata=SimpleNamespace(
            prompt_token_count=prompt_tokens,
            candidates_token_count=completion_tokens,
        ),
    )


def _make_client(*, generate_return: Any = None,
                 stream_chunks: list[Any] | None = None) -> SimpleNamespace:
    generate = AsyncMock(return_value=generate_return) if generate_return else AsyncMock()

    if stream_chunks is not None:
        async def _gen():
            for c in stream_chunks:
                yield c

        async def _start_stream(**kwargs):
            return _gen()

        stream = AsyncMock(side_effect=_start_stream)
    else:
        stream = AsyncMock()

    models = SimpleNamespace(
        generate_content=generate,
        generate_content_stream=stream,
    )
    return SimpleNamespace(aio=SimpleNamespace(models=models))


# --- generate() ---------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_returns_text_and_usage() -> None:
    client = _make_client(generate_return=_response('{"x": 1}'))
    adapter = GeminiLLMAdapter({"client": client, "model": "gemini-2.0-flash"})

    result = await adapter.generate(
        [LLMMessage(role="user", content="hi")],
        LLMConfig(model="gemini-2.0-flash", temperature=0.4, max_tokens=128),
    )
    assert result.text == '{"x": 1}'
    assert result.finish_reason == "stop"
    assert result.usage == {"prompt_tokens": 10, "completion_tokens": 5}

    call_kwargs = client.aio.models.generate_content.await_args.kwargs
    assert call_kwargs["model"] == "gemini-2.0-flash"
    assert call_kwargs["config"]["temperature"] == 0.4
    assert call_kwargs["config"]["max_output_tokens"] == 128


@pytest.mark.asyncio
async def test_generate_maps_system_role_into_system_instruction() -> None:
    client = _make_client(generate_return=_response("ok"))
    adapter = GeminiLLMAdapter({"client": client})

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
    kwargs = client.aio.models.generate_content.await_args.kwargs
    assert kwargs["config"]["system_instruction"] == "be terse\n\nreply in Hindi"
    contents = kwargs["contents"]
    # System messages live in system_instruction, not in contents
    assert all(c["role"] in ("user", "model") for c in contents)
    # Assistant -> model
    assert any(c["role"] == "model" for c in contents)
    # Order preserved
    assert contents[0]["role"] == "user"
    assert contents[0]["parts"][0]["text"] == "hi"


@pytest.mark.asyncio
async def test_generate_json_format_sets_mime_type() -> None:
    client = _make_client(generate_return=_response("{}"))
    adapter = GeminiLLMAdapter({"client": client})
    await adapter.generate(
        [LLMMessage(role="user", content="hi")],
        LLMConfig(response_format="json"),
    )
    cfg = client.aio.models.generate_content.await_args.kwargs["config"]
    assert cfg["response_mime_type"] == "application/json"


@pytest.mark.asyncio
async def test_generate_text_format_omits_mime_type() -> None:
    client = _make_client(generate_return=_response("plain"))
    adapter = GeminiLLMAdapter({"client": client})
    await adapter.generate([LLMMessage(role="user", content="hi")], LLMConfig(response_format="text"))
    cfg = client.aio.models.generate_content.await_args.kwargs["config"]
    assert "response_mime_type" not in cfg


@pytest.mark.asyncio
async def test_generate_finish_reason_max_tokens() -> None:
    client = _make_client(generate_return=_response("trunc", finish_reason="MAX_TOKENS"))
    adapter = GeminiLLMAdapter({"client": client})
    result = await adapter.generate([LLMMessage(role="user", content="hi")], LLMConfig())
    assert result.finish_reason == "length"


@pytest.mark.asyncio
async def test_generate_finish_reason_safety_blocked() -> None:
    client = _make_client(generate_return=_response("", finish_reason="SAFETY"))
    adapter = GeminiLLMAdapter({"client": client})
    result = await adapter.generate([LLMMessage(role="user", content="hi")], LLMConfig())
    assert result.finish_reason == "blocked"


# --- generate_stream() --------------------------------------------------


@pytest.mark.asyncio
async def test_generate_stream_yields_text_per_chunk() -> None:
    client = _make_client(stream_chunks=[
        _response("Hel"),
        _response("lo, "),
        _response("world"),
    ])
    adapter = GeminiLLMAdapter({"client": client})

    tokens: list[str] = []
    async for t in adapter.generate_stream(
        [LLMMessage(role="user", content="hi")], LLMConfig(),
    ):
        tokens.append(t)
    assert tokens == ["Hel", "lo, ", "world"]


@pytest.mark.asyncio
async def test_generate_stream_skips_empty_chunks() -> None:
    empty = SimpleNamespace(text="", candidates=[])
    client = _make_client(stream_chunks=[empty, _response("hello"), empty])
    adapter = GeminiLLMAdapter({"client": client})
    tokens = []
    async for t in adapter.generate_stream([LLMMessage(role="user", content="hi")], LLMConfig()):
        tokens.append(t)
    assert tokens == ["hello"]


# --- Construction ------------------------------------------------------


@pytest.mark.asyncio
async def test_constructor_rejects_missing_key(monkeypatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(ValueError):
        GeminiLLMAdapter({})


# --- _extract_text fallback ---------------------------------------------


def test_extract_text_falls_back_to_candidate_parts() -> None:
    response = SimpleNamespace(
        text=None,
        candidates=[
            SimpleNamespace(
                content=SimpleNamespace(parts=[
                    SimpleNamespace(text="a"),
                    SimpleNamespace(text="b"),
                ]),
                finish_reason=SimpleNamespace(name="STOP"),
            )
        ],
    )
    assert GeminiLLMAdapter._extract_text(response) == "ab"


def test_extract_text_empty_when_no_candidates() -> None:
    assert GeminiLLMAdapter._extract_text(SimpleNamespace(text=None, candidates=[])) == ""
