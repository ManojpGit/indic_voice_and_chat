"""System prompt builder for VoiceBot + ChatBot.

Builds the system message that goes to the LLM. The prompt:
- Identifies the agent (name, role, company)
- Sets the language and code-switching policy
- Embeds the talking points / qualifying questions / objection responses
- Lists the slots the agent should try to fill
- Specifies the structured JSON response schema (PRD §12.2 / §12.3)

Kept as a pure-Python builder rather than a templating engine so it's easy
to inspect, diff, and unit-test.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional

from src.dialogue.slots import SlotSchema


VOICEBOT_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["response_text", "language", "action"],
    "properties": {
        "response_text": {"type": "string"},
        "language": {"type": "string"},
        "conversation_phase": {
            "type": "string",
            "enum": ["opening", "pitch", "qualification", "objection", "closing"],
        },
        "updated_slots": {"type": "object"},
        "action": {
            "type": "string",
            "enum": [
                "continue",
                "clarify",
                "transfer",
                "schedule_callback",
                "send_info",
                "close_positive",
                "close_negative",
                "end",
            ],
        },
        "action_reason": {"type": "string"},
        "sentiment": {
            "type": "string",
            "enum": ["positive", "neutral", "negative", "frustrated"],
        },
        "internal_notes": {"type": "string"},
    },
}

CHATBOT_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["response_text", "language"],
    "properties": {
        "response_text": {"type": "string"},
        "language": {"type": "string"},
        "sources_used": {"type": "array", "items": {"type": "string"}},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
        "action": {
            "type": "string",
            "enum": ["none", "schedule_callback", "send_info", "create_ticket", "escalate"],
        },
        "suggested_followups": {"type": "array", "items": {"type": "string"}},
    },
}


@dataclass
class VoiceBotScript:
    agent_name: str
    agent_role: str
    company_name: str
    language_default: str = "hi"
    opening: str = ""
    talking_points: list[str] = field(default_factory=list)
    qualifying_questions: list[str] = field(default_factory=list)
    objection_responses: dict[str, str] = field(default_factory=dict)
    closing: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_campaign_yaml(cls, script: dict[str, Any]) -> "VoiceBotScript":
        return cls(
            agent_name=script.get("agent_name", "Agent"),
            agent_role=script.get("agent_role", "Customer Engagement"),
            company_name=script.get("company_name", "[Company]"),
            language_default=script.get("language_default", "hi"),
            opening=script.get("opening", ""),
            talking_points=list(script.get("talking_points") or []),
            qualifying_questions=list(script.get("qualifying_questions") or []),
            objection_responses=dict(script.get("objection_responses") or {}),
            closing=dict(script.get("closing") or {}),
        )


def build_voicebot_system_prompt(
    script: VoiceBotScript,
    schema: SlotSchema,
    lead_data: Optional[dict[str, Any]] = None,
    extra_directives: Optional[list[str]] = None,
) -> str:
    """Assemble the full system prompt for VoiceBotAgent."""
    lead_data = lead_data or {}
    parts: list[str] = []

    parts.append(
        f"You are {script.agent_name}, a {script.agent_role} at {script.company_name}. "
        f"You are on a phone call with a lead. Speak naturally as a human would on a call."
    )

    parts.append(
        f"Default language: {script.language_default}. Mirror the user's language and "
        "match their level of formality. Code-switch (Hindi/English mixing) is fine "
        "if the user does it."
    )

    if script.opening:
        parts.append("Opening line:\n" + script.opening.strip())

    if script.talking_points:
        bullets = "\n".join(f"- {p}" for p in script.talking_points)
        parts.append("Talking points:\n" + bullets)

    if script.qualifying_questions:
        bullets = "\n".join(f"- {q}" for q in script.qualifying_questions)
        parts.append("Qualifying questions to ask when natural:\n" + bullets)

    if script.objection_responses:
        bullets = "\n".join(
            f"- {tag}: {resp}" for tag, resp in script.objection_responses.items()
        )
        parts.append("Objection responses (use the tone, not verbatim):\n" + bullets)

    if script.closing:
        bullets = "\n".join(f"- {tag}: {resp}" for tag, resp in script.closing.items())
        parts.append("Closing lines:\n" + bullets)

    if schema.specs:
        slot_lines = []
        for name, spec in schema.specs.items():
            mark = "*" if spec.required else " "
            extra = (
                f" (one of: {', '.join(spec.values)})"
                if spec.values
                else f" ({spec.type.value})"
            )
            slot_lines.append(f"  {mark} {name}{extra}")
        parts.append(
            "Slots to fill (* = required). Update them via the JSON `updated_slots` field "
            "as you learn from the user:\n" + "\n".join(slot_lines)
        )

    if lead_data:
        parts.append("Known lead data:\n" + json.dumps(lead_data, ensure_ascii=False, indent=2))

    parts.append(
        "On every turn you MUST respond with a single JSON object matching this schema:\n"
        + json.dumps(VOICEBOT_RESPONSE_SCHEMA, indent=2)
    )

    parts.append(
        "Rules:\n"
        "- Keep `response_text` concise (1-2 sentences) — this is voice, not chat.\n"
        "- Never invent facts about the company or its products.\n"
        "- If the user asks if you are AI, answer honestly using the `is_ai` objection response.\n"
        "- If the user asks to be removed, set action=close_negative and acknowledge.\n"
        "- Set action=end only when the conversation is genuinely over."
    )

    if extra_directives:
        parts.append("Additional directives:\n" + "\n".join(f"- {d}" for d in extra_directives))

    return "\n\n".join(parts)


def build_chatbot_system_prompt(
    company_name: str,
    language_default: str = "en",
    rag_context: Optional[str] = None,
    extra_directives: Optional[list[str]] = None,
) -> str:
    """System prompt for the RAG-powered ChatBot agent (Phase 4)."""
    parts: list[str] = []
    parts.append(
        f"You are a helpful assistant for {company_name}. Answer the user's question "
        "using only the provided sources. If the sources don't contain the answer, "
        "say you don't know — do not invent."
    )
    parts.append(f"Default language: {language_default}. Mirror the user's language.")

    if rag_context:
        parts.append("Reference sources:\n" + rag_context)

    parts.append(
        "Respond with a single JSON object matching this schema:\n"
        + json.dumps(CHATBOT_RESPONSE_SCHEMA, indent=2)
    )

    if extra_directives:
        parts.append("Additional directives:\n" + "\n".join(f"- {d}" for d in extra_directives))

    return "\n\n".join(parts)
