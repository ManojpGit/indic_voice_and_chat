"""Gemini LLM adapter (Google Generative AI).

Wraps the official ``google-genai`` SDK. Both batch and streaming flows
are supported — streaming uses ``aio.generate_content_stream`` and yields
text deltas as they arrive.

Indic-language handling: Gemini understands Hindi/Devanagari natively, so
the adapter doesn't need to do any pre-processing. The agent's system
prompt declares the language directive (see ``build_voicebot_system_prompt``).
"""

from __future__ import annotations

import os
from typing import Any, AsyncIterator, Optional

from src.interfaces.llm import ILLMProvider, LLMConfig, LLMMessage, LLMResult


DEFAULT_MODEL = "gemini-2.0-flash"


class GeminiLLMAdapter(ILLMProvider):
    def __init__(self, config: dict[str, Any]) -> None:
        self._default_model = config.get("model") or DEFAULT_MODEL
        client = config.get("client")
        if client is not None:
            # Tests inject a fake; bypass real SDK construction.
            self._client = client
            return
        api_key = config.get("api_key") or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "GeminiLLMAdapter requires an API key (config 'api_key' or "
                "GEMINI_API_KEY env var)"
            )
        try:
            from google import genai  # type: ignore[import-not-found]
        except ImportError as e:
            raise RuntimeError(
                "GeminiLLMAdapter requires the 'google-genai' package. "
                "Install with: pip install google-genai"
            ) from e
        self._client = genai.Client(api_key=api_key)

    # --- Message conversion ---------------------------------------------

    @staticmethod
    def _to_gemini_contents(messages: list[LLMMessage]) -> tuple[Optional[str], list[dict]]:
        """Split our (role, content) messages into Gemini's shape.

        Gemini takes a separate ``system_instruction`` plus a list of
        ``contents`` entries with ``role`` in {user, model}. We map:
            our "system"    -> system_instruction (concatenated if many)
            our "user"      -> {role: "user", parts: [{text: ...}]}
            our "assistant" -> {role: "model", parts: [{text: ...}]}
        """
        system_parts: list[str] = []
        contents: list[dict] = []
        for m in messages:
            if m.role == "system":
                system_parts.append(m.content)
                continue
            role = "model" if m.role == "assistant" else "user"
            contents.append({"role": role, "parts": [{"text": m.content}]})
        system = "\n\n".join(system_parts) if system_parts else None
        return system, contents

    def _build_config(self, config: LLMConfig) -> dict[str, Any]:
        cfg: dict[str, Any] = {
            "temperature": config.temperature,
            "max_output_tokens": config.max_tokens,
        }
        if config.response_format == "json":
            cfg["response_mime_type"] = "application/json"
        return cfg

    # --- Public API ----------------------------------------------------

    async def generate(
        self,
        messages: list[LLMMessage],
        config: LLMConfig,
    ) -> LLMResult:
        system, contents = self._to_gemini_contents(messages)
        gen_config = self._build_config(config)
        if system:
            gen_config["system_instruction"] = system

        response = await self._client.aio.models.generate_content(
            model=config.model or self._default_model,
            contents=contents,
            config=gen_config,
        )
        text = self._extract_text(response)
        usage = self._extract_usage(response)
        finish_reason = self._extract_finish_reason(response)
        return LLMResult(
            text=text,
            finish_reason=finish_reason,
            usage=usage,
            raw_response=_dump(response),
        )

    async def generate_stream(
        self,
        messages: list[LLMMessage],
        config: LLMConfig,
    ) -> AsyncIterator[str]:
        system, contents = self._to_gemini_contents(messages)
        gen_config = self._build_config(config)
        if system:
            gen_config["system_instruction"] = system

        stream = await self._client.aio.models.generate_content_stream(
            model=config.model or self._default_model,
            contents=contents,
            config=gen_config,
        )
        async for chunk in stream:
            text = self._extract_text(chunk)
            if text:
                yield text

    # --- Response shape helpers (resilient to SDK changes) -------------

    @staticmethod
    def _extract_text(response: Any) -> str:
        # The SDK exposes a ``.text`` convenience accessor; fall back to
        # walking ``candidates[0].content.parts[*].text`` if that fails.
        text = getattr(response, "text", None)
        if text:
            return text
        candidates = getattr(response, "candidates", None) or []
        if not candidates:
            return ""
        content = getattr(candidates[0], "content", None)
        if content is None:
            return ""
        parts = getattr(content, "parts", None) or []
        return "".join((getattr(p, "text", "") or "") for p in parts)

    @staticmethod
    def _extract_usage(response: Any) -> dict:
        u = getattr(response, "usage_metadata", None)
        if u is None:
            return {}
        return {
            "prompt_tokens": getattr(u, "prompt_token_count", 0) or 0,
            "completion_tokens": getattr(u, "candidates_token_count", 0) or 0,
        }

    @staticmethod
    def _extract_finish_reason(response: Any) -> str:
        candidates = getattr(response, "candidates", None) or []
        if not candidates:
            return "stop"
        raw = getattr(candidates[0], "finish_reason", None)
        if raw is None:
            return "stop"
        # Gemini returns an enum or a string depending on SDK version.
        name = getattr(raw, "name", None) or str(raw)
        # Map Gemini's vocabulary to ours.
        if "STOP" in name:
            return "stop"
        if "MAX_TOKENS" in name:
            return "length"
        if "SAFETY" in name or "RECITATION" in name:
            return "blocked"
        return name.lower()


def _dump(response: Any) -> dict:
    """Best-effort serialization so the raw_response can be inspected."""
    if hasattr(response, "model_dump"):
        try:
            return response.model_dump()
        except Exception:  # noqa: BLE001
            pass
    if hasattr(response, "to_dict"):
        try:
            return response.to_dict()
        except Exception:  # noqa: BLE001
            pass
    return {"text": getattr(response, "text", "")}
